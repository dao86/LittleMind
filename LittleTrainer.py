import argparse
import json
import math
import os
import shutil
from contextlib import nullcontext
from datetime import datetime
from json import JSONDecoder


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch import optim, nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from LittleMind import modelConfig, LittleForCausalLM, Little_Lora
from LittleDataSet import dataset_pre, dataset_sft, dataset_dpo


class trainConfig(JSONDecoder):
    model_type = "xlm_model_config"

    def __init__(self,
                 id='pre_20250101001',  # 和checkpoint配合使用
                 lora_name='',
                 lora_target = None,
                 lora_rank=8,
                 data_path='./data',
                 out_path='./output/model',
                 log_out_path='./output/log',
                 device='cpu',
                 dtype='float16',  # bfloat16 GPU支持，float16 cpu和GPU都支持
                 epochs: int = 1,
                 batch_size=1,
                 accumulation_steps:int=1,
                 grad_clip=1.0,
                 checkpoint_interval:int=100,
                 log_interval:int=100,
                 save_interval:int =1000,
                 log_cnt:int=0,
                 num_checkpoint: int = 0,
                 save_cnt:int=0,
                 num_workers=1,
                 learning_rate=5e-5,
                 train_type='pretrain',  # pretrain,sft,dpo
                 save_name='',
                 # 多卡分布式
                 ddp=False,
                 world_size=1,
                 num_machine=1,
                 dpp_master_addr='127.0.0.1',
                 dpp_port:int=37001,
                 dpo_beta=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.lora_name=lora_name
        self.lora_target = lora_target
        self.lora_rank=lora_rank
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_path = data_path
        self.out_path = out_path
        self.log_out_path = log_out_path
        self.dtype = dtype
        self.accumulation_steps = accumulation_steps
        self.grad_clip = grad_clip
        self.num_checkpoint = num_checkpoint
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_interval  = checkpoint_interval
        self.log_cnt = log_cnt
        self.save_cnt = save_cnt
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.train_type = train_type
        self.save_name = save_name
        self.device = device
        self.ddp = ddp
        self.world_size = world_size
        self.num_machine = num_machine
        self.dpp_master_addr = dpp_master_addr
        self.dpp_port = dpp_port
        self.dpo_beta = dpo_beta

# -------------------------checkpoint------------------------

def pre_json(data):
    if isinstance(data, dict):
        return {k: pre_json(v) for k, v in data.items()}
    elif isinstance(data, str) or isinstance(data, int) or isinstance(data, float) or isinstance(data, bool):
        return data
    else:
        return '@no@'

def set_json(data,tar_obj):
    for k, v in data.items():
        if v != '@no@':
            tar_obj[k] = v

def save_config(config,file_path):
    data = pre_json(config.__dict__)
    config_json = json.dumps(data)
    with open(file_path, 'w', encoding='utf-8') as fw:
        fw.write(config_json)

def get_sub_dir_index(checkpoint_path):
    # 读取路径
    from pathlib import Path
    checkpoint_dir = Path(checkpoint_path)
    # 获取 root_dir 下的所有子目录
    sub_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    # 按照目录的创建时间排序（升序）
    sorted_dirs = sorted(sub_dirs, key=lambda x: x.stat().st_ctime)
    exits_num=0#返回0代表没有目录
    if sorted_dirs:
        exits_num = int(sorted_dirs[-1].name)
    return exits_num


def checkpoint_save(point_dir_path,config_train:trainConfig,config_model:modelConfig,model,logcontent):
    #根据trainconfig type+id+checkpoint 创建目录
    checkpoint_num = 1
    if os.path.exists(point_dir_path):
        exits_num = get_sub_dir_index(point_dir_path)
        checkpoint_num += exits_num
    point_dir_path = point_dir_path+f'/{checkpoint_num}'
    if not os.path.exists(point_dir_path):
        os.makedirs(point_dir_path,exist_ok=True)
    #保存trainconfig,modelconfig
    save_config(config_train,point_dir_path + '/train_config.json')
    save_config(config_model, point_dir_path + '/model_config.json')
    #保存tokenizer dir
    shutil.copytree(config_model.token_path, point_dir_path+'/tokenizer')
    #保存 log
    log_path = point_dir_path + f'/{config_train.train_type}-train_log.log'
    with open(log_path, 'w', encoding='utf-8') as log:
        log.writelines(logcontent)
    #保存 pth
    save_path = point_dir_path + f'/{config_train.train_type}-{config_model.num_attention_layers}-{config_model.hidden_dim}.pth'
    model.eval()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # 分布式
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
    torch.save(state_dict, save_path)
    model.train()

    print(f'checkpoint 保存成功---{point_dir_path}---{datetime.now()}')

def checkpoint_load(checkpoint_path):
    # 加载 trainconfig, modelconfig
    sub_dir = get_sub_dir_index(checkpoint_path)
    if sub_dir==0:
        raise "没有checkpoint"
    checkpoint_dir_path = checkpoint_path+f'/{sub_dir}'
    with open(checkpoint_dir_path+'/train_config.json', 'r', encoding='utf-8') as fw:
        json_train_config_data = fw.read()
    train_loaded_dict = json.loads(json_train_config_data)
    config_train = trainConfig()
    set_json(train_loaded_dict, config_train.__dict__)


    with open(checkpoint_dir_path+'/model_config.json', 'r', encoding='utf-8') as fw:
        json_model_config_data = fw.read()
    model_loaded_dict = json.loads(json_model_config_data)
    config_model = modelConfig()
    set_json(model_loaded_dict, config_model.__dict__)


    # 加载tokenizer dir
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir_path+'/tokenizer')
    # 加载 pth
    model_path = checkpoint_dir_path + f'/{config_train.train_type}-{config_model.num_attention_layers}-{config_model.hidden_dim}.pth'
    model=LittleForCausalLM(config_model)
    model.load_state_dict(torch.load(model_path, map_location=config_model.device))
    print(f'checkpoint 加载参数成功 {model_path}')
    if config_train.ddp:
        pass
    model.to(config_model.device)
    print(f'checkpoint 读取成功---{checkpoint_dir_path}---{datetime.now()}')
    return tokenizer,model,config_train,config_model

# -------------------------train------------------------
def get_grad(model):
    # 梯度最大最小比率
    ratio_embedding = 0.0
    ratio_p = 0.0
    cnt = 0
    # 计算梯度范数
    grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:

            if name == 'model.embedding.weight':
                grad = param.grad
                mean = grad.abs().mean().item()
                max_val = grad.abs().max().item()
                ratio_embedding = max_val / mean if mean > 0 else float('inf')
                a = 1
            else:
                grad = param.grad
                mean = grad.abs().mean().item()
                max_val = grad.abs().max().item()
                ratio_p += max_val / mean if mean > 0 else float('inf')
                cnt += 1
                # 计算梯度范数
                grad_norm += param.grad.data.norm(2).item() ** 2
                b = 1

    grad_norm = grad_norm ** 0.5
    ratio_p = ratio_p / cnt
    return grad_norm,ratio_p,ratio_embedding

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_model_tokenizer(config_model, model_file):
    tokenizer = AutoTokenizer.from_pretrained(config_model.token_path)
    print(f'vocab_size:{config_model.vocab_size},tokenizer:{len(tokenizer)}')
    try:

        model = LittleForCausalLM(config_model).to(config_model.device)

        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file, map_location=config_model.device))
            print(f'加载参数成功 {model_file}')
        else:
            print(f'没有可以加载的参数文件 {model_file}')
        print(f'训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.3f} B')
    except BaseException:
        print(f"Error: {BaseException}")
    return tokenizer, model

def init_data(tokenizer, config_model, config_train, local_rank):
    if config_train.train_type == 'pre':
        dataset = dataset_pre(config_train.data_path, tokenizer, config_model.max_seq_len)
    elif config_train.train_type == 'dpo':
        dataset = dataset_dpo(config_train.data_path, tokenizer, config_model.max_seq_len)
    elif config_train.train_type == 'orpo':
        dataset = dataset_dpo(config_train.data_path, tokenizer, config_model.max_seq_len)
    else:
        dataset = dataset_sft(config_train.data_path, tokenizer, config_model.max_seq_len)

    train_sampler = None
    if config_train.ddp:
        train_sampler = DistributedSampler(dataset, num_replicas=config_train.world_size, rank=local_rank)

    dataloader = DataLoader(
        dataset,
        batch_size=config_train.batch_size,
        pin_memory=False if config_train.device == 'cpu' else True,  # 有GPU 为true
        drop_last=False,
        shuffle=True,
        prefetch_factor=2,  # 提前预取因子
        persistent_workers=True,  # 保持工作进程持久化
        num_workers=config_train.num_workers,  # 进程数
        sampler=train_sampler
    )
    return dataloader


def save_model_log(model, config_model, config_train, is_bak: bool = False):
    def save_train_model(model, save_path):
        model.eval()

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # 分布式
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
        torch.save(state_dict, save_path)

        print(f'保存参数成功---{save_path}---{datetime.now()}')
        model.train()

    def save_lora_model(model,config_train):
        lora_model_file_path = f'{config_train.out_path}/{config_train.lora_name}.pth'
        Little_Lora.save_lora(model,lora_model_file_path,config_train.lora_name)
        print(f'保存 lora 参数成功---{lora_model_file_path}---{datetime.now()}')

    if config_train.lora_name != '':
        save_lora_model(model,config_train)
    else:
        bak = '_bak' if is_bak else ''
        save_path = f'{config_train.out_path}/{config_train.train_type}{config_train.lora_name}_{config_model.hidden_dim}{bak}.pth'
        save_train_model(model, save_path)
        log_content = f'"text":{save_path},"time":{datetime.now()}'
        log_path = f'{config_train.log_out_path}/{config_train.train_type}_save_log_{config_model.hidden_dim}.log'
        log_train(log_content, log_path)


def log_train(log_content, log_file):
    log_content = '{' + log_content + '}\n'
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(log_content)


def pre_train_env(config_train):
    # 当训练在CPU上进行时，使用nullcontext()。这是一个不做任何事情的上下文管理器，意味着不会对计算类型进行任何转换。
    # 它允许你在不支持或不需要混合精度的环境中（如仅使用CPU时），仍然可以使用相同的代码结构而不引发错误。
    ctx = nullcontext() if config_train.device == "cpu" else torch.cuda.amp.autocast()

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler(enabled=(config_train.dtype in ['float16', 'bfloat16']))
    else:
        scaler = torch.amp.GradScaler(device=config_train.device,
                                      enabled=(config_train.dtype in ['float16', 'bfloat16']))

    return ctx,scaler

def set_checkpoint_log_save_lossacc_interval(iter_per_epoch,config_train):
    #打印日志设置
    if config_train.log_cnt>0:
        config_train.log_interval = iter_per_epoch // config_train.log_cnt  # 打印 log_cnt 个日志
    if config_train.log_interval < 1:# log_interval <1代表不打日志，说明希望打很多次日志，设置为1
        config_train.log_interval = 1
    if config_train.log_interval > iter_per_epoch:# log_interval >没批次数据数 代表打不了日志，说明希望打很少次日志，设置为iter_per_epoch
        config_train.log_interval = iter_per_epoch

    # 保存模型设置
    if config_train.save_cnt > 0:
        config_train.save_interval = iter_per_epoch // config_train.save_cnt  # 保存 save_cnt 次
    if config_train.save_interval < 1:
        config_train.save_interval = 1
    if config_train.save_interval > iter_per_epoch:
        config_train.save_interval = iter_per_epoch

    # 保存checkpoint设置 因为save_cnt至少保存一次，如果num_checkpoint==0 实际就不保存
    if config_train.num_checkpoint > 0:
        config_train.checkpoint_interval = iter_per_epoch // config_train.num_checkpoint  # checkpoint num_checkpoint 个
    if config_train.checkpoint_interval < 1:
        config_train.checkpoint_interval = 1
    if config_train.checkpoint_interval > iter_per_epoch:
        config_train.checkpoint_interval = iter_per_epoch

    # 反向传播 至少一次，或者一个批次一次
    if config_train.accumulation_steps < 1:
        config_train.accumulation_steps = 1
    if config_train.accumulation_steps > iter_per_epoch:
        config_train.accumulation_steps = iter_per_epoch



def step_acc_loss(step,scaler,optimizer,model,config_train):
    ratio_embedding = 0.0
    ratio_p = 0.0
    grad_norm = 0.0

    if (step + 1) % config_train.accumulation_steps == 0:
        scaler.unscale_(optimizer)

        grad_norm, ratio_p, ratio_embedding = get_grad(model)

        torch.nn.utils.clip_grad_norm_(model.parameters(), config_train.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)
        return grad_norm, ratio_p, ratio_embedding
    return grad_norm, ratio_p, ratio_embedding

def step_log_savemodel_checkpoint(epoch,step,iter_per_epoch,start_time,loss,optimizer,
                                  ratio_embedding,ratio_p,grad_norm,model,config_train,config_model, checkpoint_logs):
    if (step) % config_train.log_interval == 0 or (step+1) % config_train.log_interval == 0:
        spend_time = datetime.now() - start_time
        print(
            'Epoch:[{}/{}],batch_step:({}/{}), loss:{:.3f}, lr:{:.12f},grad_ratio:{:.3f}-{:.3f}-{:.3f}'.format(
                epoch + 1,
                config_train.epochs,
                step + 1,
                iter_per_epoch,
                loss.item() * config_train.accumulation_steps,
                optimizer.param_groups[-1]['lr'],
                ratio_embedding,
                ratio_p,
                grad_norm
                # spend_time / (step + 1) * iter_per_epoch // 60,
                # spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                # trainfile
            )
        )

        log_content = ('"total_epoch":{},"epoch":{},"batch_step":{},'
                       '"total_step_per_epoch":{}, "loss":{:.3f}, "lr":{:.12f} ,"grad":{:.12f} ,'
                       '"total_time":"{}" ,'
                       '"use_time":"{}","trainfile":"{}","time":"{}"').format(
            config_train.epochs,
            epoch + 1,
            step + 1,
            iter_per_epoch,
            loss * config_train.accumulation_steps,
            optimizer.param_groups[-1]['lr'], grad_norm,
            spend_time / (step + 1) * iter_per_epoch // 60,
            spend_time // 60,
            config_train.data_path,
            datetime.now()
        )
        log_path = f'{config_train.log_out_path}/{config_train.train_type}_log_{config_model.hidden_dim}.log'
        log_train(log_content, log_path)
        checkpoint_logs.append('{' + log_content + '}\n')

    if (step + 1) % config_train.save_interval == 0 and (not config_train.ddp or dist.get_rank() == 0):
        save_model_log(model, config_model, config_train)

    #num_checkpoint== 0 不保存checkpoint
    if config_train.num_checkpoint>0 and (step + 1) % config_train.checkpoint_interval == 0:
        point_serial = (step + 1) // config_train.checkpoint_interval
        point_dir_path = f'{config_train.out_path}/checkpoint/{config_train.train_type}_{config_train.id}'
        checkpoint_save(point_dir_path, config_train, config_model, model, checkpoint_logs)
        checkpoint_logs = []


def train_file(model, dataloader, config_train: trainConfig, config_model: modelConfig, is_dir: bool):
    print(f'开始处理文件---{config_train.data_path}---{datetime.now()}')
    if is_dir:
        train(model, config_train, config_model, dataloader, 0)
    else:
        for epoch in range(config_train.epochs):
            if config_train.ddp:
                dataloader.sampler.set_epoch(epoch)
            train(model, config_train, config_model, dataloader, epoch)
    if config_train.lora_name =='':
        save_model_log(model, config_model, config_train)
        print(f'文件处理完成---{config_train.data_path}---{datetime.now()}')


def train_dir(model, tokenizer, config_train: trainConfig, config_model: modelConfig, local_rank):
    def dir_read_savelog(savelog_path):
        process_files = []
        if os.path.exists(savelog_path):
            with open(savelog_path, 'r', encoding='utf-8') as log:
                for i, line in enumerate(log):
                    data = json.loads(line.strip().replace('\\', '/'))
                    process_files.append(data["savefile"])
        return process_files

    # 读取训练日志
    last_log_file = f'{config_train.log_out_path}/{config_train.train_type}_lastfile_{config_model.hidden_dim}.log'
    if not last_log_file:
        print(f'无此文件---{last_log_file}')
        return
    process_files = dir_read_savelog(last_log_file)

    train_files = os.listdir(config_train.data_path)
    train_dir = config_train.data_path
    for epoch in range(config_train.epochs):
        for file in train_files:
            file_full_path = os.path.join(train_dir, file).strip().replace('\\', '/')

            if os.path.isfile(file_full_path):  # 确保是文件
                # 这个文件在处理日志中存在，则不处理
                if file_full_path in process_files:
                    continue
                config_train.data_path = file_full_path
                dataloader = init_data(tokenizer, config_model, config_train, local_rank)
                if config_train.ddp:
                    dataloader.sampler.set_epoch(epoch)
                train_file(model, dataloader, config_train, config_model, True)
                with open(last_log_file, 'a', encoding='utf-8') as log:
                    logtext = '"savefile":"{}","savetime":"{}"'.format(file_full_path, datetime.now())
                    log.write('{' + logtext + '}\n')
        # 全部处理完成，修改训练日志文件名
        os.rename(last_log_file,
                  last_log_file.replace('.log', f'_{datetime.now().strftime("%Y_%m%d_%H%M%S")}.log'))

    config_train.data_path = train_dir

def get_loss_optim(model,config_train: trainConfig):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    if config_train.train_type != '':
        params=[]
        for name,param in model.named_parameters():
            if config_train.lora_name in name:
                params.append(param)
            else:
                param.requires_grad = False
        optimizer = optim.AdamW(params, lr=config_train.learning_rate)
        lora_params_count = sum(p.numel() for name, p in model.named_parameters() if config_train.lora_name in name)  # LoRA 参数数量
        print(f'lora训练参数量：{lora_params_count / 1e7:.3f} 千万')
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config_train.learning_rate)
    return loss_fct,optimizer

def train(model, config_train: trainConfig, config_model: modelConfig, train_loader, epoch):
    loss_fct,optimizer = get_loss_optim(model,config_train)
    ctx, scaler = pre_train_env(config_train)
    # 批次数量
    iter_per_epoch = len(train_loader)
    set_checkpoint_log_save_lossacc_interval(iter_per_epoch,config_train)
    checkpoint_logs=[]
    start_time = datetime.now()
    # writer = SummaryWriter()
    for step, (input, lable, loss_mask) in enumerate(train_loader):
        input = input.to(config_train.device)
        lable = lable.to(config_train.device)
        loss_mask = loss_mask.to(config_train.device)
        lr = get_lr(epoch * iter_per_epoch + step, config_train.epochs * iter_per_epoch, config_train.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        with ctx:
            res = model(input)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), lable.view(-1)).view(lable.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # loss += res.aux_loss
            loss = loss / config_train.accumulation_steps
        scaler.scale(loss).backward()
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name + '_grad', param.grad.clone().cpu().data.numpy(), step)
        grad_norm, ratio_p, ratio_embedding = step_acc_loss(step,scaler,optimizer,model,config_train)
        step_log_savemodel_checkpoint(epoch, step, iter_per_epoch, start_time,
                           loss, optimizer, ratio_embedding, ratio_p,grad_norm,model,config_train,config_model,checkpoint_logs)

# -------------------------dpo------------------------

def dpo_init_model_tokenizer(config_model, model_file_path):
    tokenizer = AutoTokenizer.from_pretrained(config_model.token_path)
    print(f'vocab_size:{config_model.vocab_size},tokenizer:{len(tokenizer)}')
    model = LittleForCausalLM(config_model)
    model_ref = LittleForCausalLM(config_model)
    try:
        if os.path.exists(model_file_path):
            model.load_state_dict(torch.load(model_file_path))
            print(f'model 加载参数成功 {model_file_path}')
            model_ref.load_state_dict(torch.load(model_file_path))
            print(f'model_ref 加载参数成功 {model_file_path}')
        else:
            print(f'model 没有可以加载的参数文件 {model_file_path}')
        print(f'训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e7:.3f} 千万')
    except BaseException:
        print(f"Error: {BaseException}")
    model.to(config_model.device)
    model_ref.to(config_model.device)
    return tokenizer, model, model_ref


def get_prob(out, label, mask):
    # out.shape    batch,seq_len,vocab_size
    # label.shape  batch,seq_len
    # mask.shape   batch,seq_len
    # out_chose = softmax(out_chose)
    prob = nn.functional.log_softmax(out, dim=-1)
    # chose_prob = mean(out_chose.gather(y_chose))
    # label.unsqueeze(2).shape  batch,seq_len,new
    # torch.gather(prob, dim=1, index=label.unsqueeze(2)).squeeze(-1) shape = batch,seq_len
    prob = torch.gather(prob, dim=2, index=label.unsqueeze(2)).squeeze(-1)
    prob = (prob * mask)
    # prob = prob.sum(-1)#此处与mini不同，mini没有sum
    prob = prob.mean(-1)
    return prob


def dpo_loss(out, out_ref, y, mask, delta=0.1):
    half_batch = out.shape[0] // 2
    out_chose, out_chose_ref, y_chose, mask_chosen = out[:half_batch], out_ref[:half_batch], y[:half_batch], mask[
                                                                                                             :half_batch]
    out_reject, out_reject_ref, y_reject, mask_reject = out[half_batch:], out_ref[half_batch:], y[half_batch:], mask[
                                                                                                                half_batch:]
    # chose_prob.shape  batch,seq_len
    chose_prob = get_prob(out_chose, y_chose, mask_chosen)
    ref_chose_prob = get_prob(out_chose_ref, y_chose, mask_chosen)
    reject_prob = get_prob(out_reject, y_reject, mask_reject)
    ref_reject_prob = get_prob(out_reject_ref, y_chose, mask_reject)
    # a = (chosen-chosen_ref) - (reject-reject_ref)
    # loss = (chose_prob - reject_prob ) - (ref_chose_prob - ref_reject_prob)
    # pi_logratios = policy_chosen_logps - policy_rejected_logps
    loss = (chose_prob - reject_prob) - (ref_chose_prob - ref_reject_prob)
    # b = sigmod(a)
    # c = -log(b)
    loss = -nn.functional.logsigmoid(delta * loss).mean(-1)
    return loss


def dpo_train_file(epoch, model, model_ref, dataloader, config_train: trainConfig, config_model: modelConfig, is_dir=False):
    # 准备环境
    ctx = nullcontext() if config_train.device == "cpu" else torch.cuda.amp.autocast()

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler(enabled=(config_train.dtype in ['float16', 'bfloat16']))
    else:
        scaler = torch.amp.GradScaler(device=config_train.device,
                                      enabled=(config_train.dtype in ['float16', 'bfloat16']))
    # 损失函数，优化器
    optimizer = optim.AdamW(model.parameters(), lr=config_train.learning_rate)

    # 训练日志等配置
    step_per_epoch = len(dataloader)
    config_train.log_interval = step_per_epoch // config_train.log_cnt  # 打印 log_cnt 个日志
    # 如果日志次数太多,改为一step一print日志
    config_train.log_interval = 1 if config_train.log_interval < 1 else config_train.log_interval

    config_train.save_interval = step_per_epoch // config_train.save_cnt  # 保存 save_cnt 次
    # 如果保存次数太多,改为一step一存
    config_train.save_interval = 1 if config_train.save_interval < 1 else config_train.save_interval
    # 保证至少更新一次参数
    config_train.accumulation_steps = step_per_epoch if step_per_epoch < config_train.accumulation_steps else config_train.accumulation_steps

    # checkpoint
    # step_per_check = step_per_epoch // config_train.num_checkpoint
    # step_per_check = 1 if step_per_check < 1 else step_per_check
    # checkpoint_logs = []

    start_time = datetime.now()
    # 循环 dpoloss 日志 保存
    for step, (chosen_input, chosen_label, chosen_loss_mask, reject_input, reject_label, reject_loss_mask) in enumerate(
            dataloader):
        lr = get_lr(epoch * step_per_epoch + step, config_train.epochs * step_per_epoch, config_train.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        chosen_input = chosen_input.to(config_train.device)
        chosen_label = chosen_label.to(config_train.device)
        chosen_loss_mask = chosen_loss_mask.to(config_train.device)
        reject_input = reject_input.to(config_train.device)
        reject_label = reject_label.to(config_train.device)
        reject_loss_mask = reject_loss_mask.to(config_train.device)

        x = torch.cat((chosen_input, reject_input), dim=0)
        y = torch.cat((chosen_label, reject_label), dim=0)
        mask = torch.cat((chosen_loss_mask, reject_loss_mask), dim=0)
        with ctx:
            with torch.no_grad():
                out_ref = model_ref(x)
            out = model(x)
            loss = dpo_loss(out.logits, out_ref.logits, y, mask, delta=config_train.dpo_beta)
            # 每次积累 1/config_train.accumulation_steps 的梯度，在(step+1) % config_train.accumulation_steps == 0时更新一次参数
            # 与每次更新参数效果差不多
            loss = loss / config_train.accumulation_steps

        scaler.scale(loss).backward()
        ratio_embedding=0.0
        ratio_p=0.0
        grad_norm=0.0

        if (step + 1) % config_train.accumulation_steps == 0:
            scaler.unscale_(optimizer)

            grad_norm, ratio_p, ratio_embedding = get_grad(model)


            torch.nn.utils.clip_grad_norm_(model.parameters(), config_train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if step % config_train.log_interval == 0:
            spend_time = datetime.now() - start_time
            print(
                'Epoch:[{}/{}],batch_step:({}/{}), loss:{:.3f}, lr:{:.12f},grad_ratio:{:.3f}-{:.3f}-{:.3f}'.format(
                    epoch + 1,
                    config_train.epochs,
                    step + 1,
                    step_per_epoch,
                    loss.item() * config_train.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    ratio_embedding,
                    ratio_p,
                    grad_norm
                    # spend_time / (step + 1) * iter_per_epoch // 60,
                    # spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    # trainfile
                )
            )

            log_content = ('"total_epoch":{},"epoch":{},"batch_step":{},'
                           '"total_step_per_epoch":{}, "loss":{:.3f}, "lr":{:.12f} ,"grad":{:.12f} ,'
                           '"total_time":"{}" ,'
                           '"use_time":"{}","trainfile":"{}","time":"{}"').format(
                config_train.epochs,
                epoch + 1,
                step + 1,
                step_per_epoch,
                loss * config_train.accumulation_steps,
                lr, grad_norm,
                spend_time / (step + 1) * step_per_epoch // 60,
                spend_time // 60,
                config_train.data_path,
                datetime.now()
            )
            log_path = f'{config_train.log_out_path}/{config_train.train_type}_log_{config_model.hidden_dim}.log'
            log_train(log_content, log_path)

        if (step + 1) % config_train.save_interval == 0:
            save_model_log(model, config_model, config_train)

        if (step + 1) % (config_train.save_interval*2) == 0:
            save_model_log(model, config_model, config_train,is_bak=True)

# -------------------------orpo-----------------------

def get_batch_logps(logits,labels,mask):
    """根据 labels 去 gather 对应位置的 log probability。.
    Args:
        logits:  Shape: (batch_size, sequence_length, vocab_size)
        labels:  Shape: (batch_size, sequence_length)
    Returns:
        shape (batch_size)
    """
    # logits包含的是chosen和reject的训练结果
    # 根据label的索引拿到chosen和rejected的对应概率（logPx,logQx)，引用用的是log_softmax所以返回的数据不是P，是log(P)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    #使用掩码把问题部分去掉，只计算推理结果部分
    return (per_token_logps * mask).sum(-1) / mask.sum(-1)

def odds_ratio_loss(policy_chosen_logps,policy_rejected_logps,device,beta) :
        """
        Args:
            policy_chosen_logps:  Shape: (batch_size)
            policy_rejected_logps: Shape: (batch_size)
        """

        #log identities and exp(log(P(y|x)) = P(y|x)
        #因为前面使用的是log_softmax，因此对应公式表达-torch.exp(policy_chosen_logps) = -P
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
        )
        ratio = nn.functional.logsigmoid(log_odds)
        losses = beta * ratio

        chosen_rewards = beta * (policy_chosen_logps.to(device)).detach()
        rejected_rewards = beta * (policy_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards, torch.mean(ratio), torch.mean(log_odds)

def orpo_loss(policy_chosen_logps, policy_rejected_logps, policy_cross_loss, device,delta=0.1):
    adds_losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = odds_ratio_loss(policy_chosen_logps, policy_rejected_logps,device,delta)
    # 全部loss chosen的交叉熵 +  odds_ratio_loss
    loss = policy_cross_loss - adds_losses.mean()
    return loss

def orpo_train_file(epoch, model, dataloader, config_train: trainConfig, config_model: modelConfig,is_dir=False):
    loss_cross = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),lr=config_train.learning_rate)
    ctx = nullcontext() if config_train.device == "cpu" else torch.cuda.amp.autocast()

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler(enabled=(config_train.dtype in ['float16', 'bfloat16']))
    else:
        scaler = torch.amp.GradScaler(device=config_train.device,
                                      enabled=(config_train.dtype in ['float16', 'bfloat16']))

    step_per_epoch = len(dataloader)
    config_train.log_interval = step_per_epoch // config_train.log_cnt  # 打印 log_cnt 个日志
    # 如果日志次数太多,改为一step一print日志
    config_train.log_interval = 1 if config_train.log_interval < 1 else config_train.log_interval

    config_train.save_interval = step_per_epoch // config_train.save_cnt  # 保存 save_cnt 次
    # 如果保存次数太多,改为一step一存
    config_train.save_interval = 1 if config_train.save_interval < 1 else config_train.save_interval
    # 保证至少更新一次参数
    config_train.accumulation_steps = step_per_epoch if step_per_epoch < config_train.accumulation_steps else config_train.accumulation_steps

    start_time = datetime.now()

    for step,(chosen_input, chosen_label, chosen_loss_mask, reject_input, reject_label, reject_loss_mask) in enumerate(dataloader):
        lr = get_lr(epoch * step_per_epoch + step, config_train.epochs * step_per_epoch, config_train.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        chosen_input = chosen_input.to(config_train.device)
        chosen_label = chosen_label.to(config_train.device)
        chosen_loss_mask = chosen_loss_mask.to(config_train.device)
        reject_input = reject_input.to(config_train.device)
        reject_label = reject_label.to(config_train.device)
        reject_loss_mask = reject_loss_mask.to(config_train.device)
        #合并到batch_size维度,相当于第一个维度变成2倍，前一半是chosen,后一半是rejected
        x = torch.cat((chosen_input, reject_input), dim=0)
        y = torch.cat((chosen_label, reject_label), dim=0)
        mask = torch.cat((chosen_loss_mask, reject_loss_mask), dim=0)

        with ctx:
            #此处相当于把chosen和reject都推理了一次
            output = model(x)
            half_batch = x.shape[0] // 2
            #chose的label
            chosen_y = y[:half_batch]
            # chose的训练结果
            chosen_out = output.logits[:half_batch]
            # chose的交叉熵loss
            policy_cross_loss = loss_cross(chosen_out.view(-1,chosen_out.shape[-1]),chosen_y.view(-1))

            #获取orpo的loss,output.logits包含的是chosen和reject的训练结果
            chosen_reject_logits = get_batch_logps(output.logits,y,mask)
            policy_chosen_logps = chosen_reject_logits[:half_batch]
            policy_rejected_logps = chosen_reject_logits[half_batch:]
            loss = orpo_loss(policy_chosen_logps,policy_rejected_logps,policy_cross_loss,config_train.device,config_train.dpo_beta)

            loss = loss / config_train.accumulation_steps
            scaler.scale(loss).backward()

            grad_norm=0
            ratio_p=0
            ratio_embedding=0
            if (step + 1) % config_train.accumulation_steps == 0:
                scaler.unscale_(optimizer)

                grad_norm, ratio_p, ratio_embedding = get_grad(model)

                torch.nn.utils.clip_grad_norm_(model.parameters(), config_train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % config_train.log_interval == 0:
                spend_time = datetime.now() - start_time
                print(
                    'Epoch:[{}/{}],batch_step:({}/{}), loss:{:.3f}, lr:{:.12f},grad_ratio:{:.3f}-{:.3f}-{:.3f}'.format(
                        epoch + 1,
                        config_train.epochs,
                        step + 1,
                        step_per_epoch,
                        loss.item() * config_train.accumulation_steps,
                        optimizer.param_groups[-1]['lr'],
                        ratio_embedding,
                        ratio_p,
                        grad_norm
                        # spend_time / (step + 1) * iter_per_epoch // 60,
                        # spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                        # trainfile
                    )
                )

                log_content = ('"total_epoch":{},"epoch":{},"batch_step":{},'
                               '"total_step_per_epoch":{}, "loss":{:.3f}, "lr":{:.12f} ,"grad":{:.12f} ,'
                               '"total_time":"{}" ,'
                               '"use_time":"{}","trainfile":"{}","time":"{}"').format(
                    config_train.epochs,
                    epoch + 1,
                    step + 1,
                    step_per_epoch,
                    loss * config_train.accumulation_steps,
                    lr, grad_norm,
                    spend_time / (step + 1) * step_per_epoch // 60,
                    spend_time // 60,
                    config_train.data_path,
                    datetime.now()
                )
                log_path = f'{config_train.log_out_path}/{config_train.train_type}_log_{config_model.hidden_dim}.log'
                log_train(log_content, log_path)

            if (step + 1) % config_train.save_interval == 0:
                save_model_log(model, config_model, config_train)

            if (step + 1) % (config_train.save_interval * 2) == 0:
                save_model_log(model, config_model, config_train, is_bak=True)

#------------------------------------------------------

#------------------------lora---------------------------

def init_lora_model_tokenizer(config_model,config_train:trainConfig, model_file):
    tokenizer = AutoTokenizer.from_pretrained(config_model.token_path)
    print(f'vocab_size:{config_model.vocab_size},tokenizer:{len(tokenizer)}')
    try:

        model = LittleForCausalLM(config_model).to(config_model.device)

        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file, map_location=config_model.device))
            print(f'加载参数成功 {model_file}')
        else:
            print(f'没有可以加载的参数文件 {model_file}')
        print(f'基座模型参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.3f} B')

        if config_train.lora_name != '':
            lora_model_file_path = f'{config_train.out_path}/{config_train.lora_name}.pth'
            Little_Lora.add_lora(model,config_train.lora_name,config_train.lora_rank,config_train.lora_target)
            Little_Lora.load_lora(model,lora_model_file_path,config_model.device,config_train.lora_name)
            print(f'加载lora参数 {config_train.lora_name}')
    except BaseException:
        print(f"Error: {BaseException}")
    return tokenizer, model

#_________________________________________________________

def pre_train(model, tokenizer, config_train: trainConfig, config_model: modelConfig, local_rank):
    if os.path.isfile(config_train.data_path):  # 文件
        dataloader = init_data(tokenizer, config_model, config_train, local_rank)
        train_file(model, dataloader, config_train, config_model, is_dir=False)
    else:  # 目录
        train_dir(model, tokenizer, config_train, config_model, local_rank)


def sft_train(model, tokenizer, config_train: trainConfig, config_model: modelConfig, local_rank):
    if os.path.isfile(config_train.data_path):  # 文件
        dataloader = init_data(tokenizer, config_model, config_train, local_rank)
        train_file(model, dataloader, config_train, config_model, is_dir=False)
    else:  # 目录
        train_dir(model, tokenizer, config_train, config_model, local_rank)


def orpo_train(model,  tokenizer, config_train, config_model, local_rank):
    if os.path.isfile(config_train.data_path):  # 文件
        dataloader = init_data(tokenizer, config_model, config_train, local_rank)
        for epoch in range(config_train.epochs):
            orpo_train_file(epoch, model, dataloader, config_train, config_model, is_dir=False)
    else:  # 目录
        pass


def dpo_train(model, model_ref, tokenizer, config_train, config_model, local_rank):
    if os.path.isfile(config_train.data_path):  # 文件
        dataloader = init_data(tokenizer, config_model, config_train, local_rank)
        for epoch in range(config_train.epochs):
            dpo_train_file(epoch, model, model_ref, dataloader, config_train, config_model, is_dir=False)
    else:  # 目录
        pass

def lora_train(model, tokenizer,config_train: trainConfig, config_model: modelConfig, local_rank):
    if os.path.isfile(config_train.data_path):  # 文件
        dataloader = init_data(tokenizer, config_model, config_train, local_rank)
        train_file(model, dataloader, config_train, config_model, is_dir=False)
    else:  # 目录
        train_dir(model, tokenizer, config_train, config_model, local_rank)

#_________________________________________________________
def checkpoint(config_train):
    point_dir_path = f'{config_train.out_path}/checkpoint/{config_train.train_type}_{config_train.id}'
    if os.path.exists(point_dir_path):
        return point_dir_path
    else:
        return ''

def pre_train_wrapper(local_rank, model_file_path, config_model, config_train):
    print(f'pre_train_wrapper local_rank:{local_rank}')
    point_dir_path = checkpoint(config_train)
    #从checkpoint加载
    if point_dir_path:
        tokenizer,model,config_train,config_model = checkpoint_load(point_dir_path)
    else:#从output加载
        tokenizer, model = init_model_tokenizer(config_model, model_file_path)

    if config_train.ddp:
        dist.init_process_group(backend="nccl",
                                init_method=f"tcp://{config_train.dpp_master_addr}:{config_train.dpp_port}",  # 主节点地址和端口
                                rank=local_rank,
                                world_size=config_train.world_size)
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(device)
        config_model.device = device
        config_train.device = device
        model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank])

    pre_train(model, tokenizer, config_train, config_model, local_rank)

    if config_train.ddp:
        dist.destroy_process_group()


def sft_train_wrapper(local_rank, model_file_path, config_model, config_train):
    print(f'sft_train_wrapper local_rank:{local_rank}')
    point_dir_path = checkpoint(config_train)
    if point_dir_path:
        tokenizer, model, config_train, config_model = checkpoint_load(point_dir_path)
    else:
        tokenizer, model = init_model_tokenizer(config_model, model_file_path)

    if config_train.ddp:
        dist.init_process_group(backend="nccl",
                                init_method=f"tcp://{config_train.dpp_master_addr}:{config_train.dpp_port}",  # 主节点地址和端口
                                rank=local_rank,
                                world_size=config_train.world_size)
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(device)
        config_model.device = device
        config_train.device = device
        model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank])

    sft_train(model, tokenizer, config_train, config_model, local_rank)
    if config_train.ddp:
        dist.destroy_process_group()


def dpo_train_wrapper(local_rank, model_file_path, config_model, config_train):
    print(f'sft_train_wrapper local_rank:{local_rank}')

    tokenizer, model, model_ref = dpo_init_model_tokenizer(config_model, model_file_path)

    if config_train.ddp:
        dist.init_process_group(backend="nccl",
                                init_method=f"tcp://{config_train.dpp_master_addr}:{config_train.dpp_port}",  # 主节点地址和端口
                                rank=local_rank,
                                world_size=config_train.world_size)
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(device)
        config_model.device = device
        config_train.device = device
        model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank])
        model_ref.to(device)
        model_ref = DistributedDataParallel(model_ref, device_ids=[local_rank])

    dpo_train(model, model_ref, tokenizer, config_train, config_model, local_rank)
    if config_train.ddp:
        dist.destroy_process_group()


def orpo_train_wrapper(local_rank, model_file_path, config_model, config_train):
    print(f'orpo_train_wrapper local_rank:{local_rank}')

    tokenizer, model = init_model_tokenizer(config_model, model_file_path)

    if config_train.ddp:
        dist.init_process_group(backend="nccl",
                                init_method=f"tcp://{config_train.dpp_master_addr}:{config_train.dpp_port}",  # 主节点地址和端口
                                rank=local_rank,
                                world_size=config_train.world_size)
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(device)
        config_model.device = device
        config_train.device = device
        model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank])


    orpo_train(model, tokenizer, config_train, config_model, local_rank)
    if config_train.ddp:
        dist.destroy_process_group()


def lora_train_wrapper(local_rank, model_file_path,config_model, config_train):
    print(f'lora_train_wrapper local_rank:{local_rank}')
    tokenizer, model = init_lora_model_tokenizer(config_model, config_train,model_file_path)

    if config_train.ddp:
        dist.init_process_group(backend="nccl",
                                init_method=f"tcp://{config_train.dpp_master_addr}:{config_train.dpp_port}",  # 主节点地址和端口
                                rank=local_rank,
                                world_size=config_train.world_size)
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(device)
        config_model.device = device
        config_train.device = device
        model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank])

    lora_train(model, tokenizer, config_train, config_model, local_rank)
    if config_train.ddp:
        dist.destroy_process_group()

#-------------------------------------------------------------------------

def set_train_args(parser):
    parser.add_argument("--epochs", type=int, default=1, help="训练次数")
    parser.add_argument("--batch_size", type=int, default=1, help="训练批次")
    parser.add_argument("--num_workers", type=int, default=1, help="数据集dataloader加载进程")
    parser.add_argument("--acc_steps", type=int, default=1, help="梯度累计次数")
    parser.add_argument("--lr", type=float, default=8e-4, help="学习率")

    parser.add_argument("--logcnt", type=int, default=0, help="日志次数 为0时使用默认log_interval")
    parser.add_argument("--savecnt", type=int, default=0, help="保存次数 为0时使用默认save_interval")
    parser.add_argument("--num_checkpoint", type=int, default=0, help="checkpoint次数 为0时使用默认 checkpoint_interval")

    parser.add_argument("--log_interval", type=int, default=1, help="log_interval")
    parser.add_argument("--save_interval", type=int, default=1, help="save_interval")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="checkpoint_interval")

    parser.add_argument("--ddp", type=int, default=0, help="分布式 1开启 0关闭")
    parser.add_argument("--world_size", type=int, default=1, help="gpu卡数")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="dpo_beta")

    parser.add_argument("--dim", type=int, default=512, help="model参数 embedding维度")
    parser.add_argument("--num_layers", type=int, default=8, help="model参数注意力层数")

    args = parser.parse_args()
    # 训练配置
    config_train = trainConfig()

    config_train.data_path = args.data_path
    config_train.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_train.epochs = args.epochs
    config_train.batch_size = args.batch_size
    config_train.accumulation_steps = args.acc_steps
    config_train.num_workers = args.num_workers
    config_train.learning_rate = args.lr

    config_train.log_cnt = args.logcnt
    config_train.save_cnt = args.savecnt
    config_train.num_checkpoint = args.num_checkpoint

    config_train.log_interval = args.log_interval
    config_train.save_interval = args.save_interval
    config_train.checkpoint_interval = args.checkpoint_interval

    config_train.ddp = (args.ddp == 1)
    config_train.world_size = args.world_size
    config_train.dpo_beta = args.dpo_beta

    config_train.id = args.train_id



    # 模型配置
    config_model = modelConfig()

    config_model.hidden_dim = args.dim
    config_model.num_attention_layers = args.num_layers
    config_model.device = config_train.device
    config_model.dropout = 0.0

    print(f'--- device:{config_train.device}--{config_model.device}\n'
          f'--- gpu_size:{torch.cuda.device_count()}\n'
          f'--- ddp:{config_train.ddp}\n'
          f'--- world_size:{config_train.world_size}\n'

          f'--- epochs:{config_train.epochs}\n'
          f'--- batch_size:{config_train.batch_size}\n'
          f'--- accumulation_steps:{config_train.accumulation_steps}\n'
          f'--- num_workers:{config_train.num_workers}\n'
          f'--- lr:{config_train.learning_rate}\n'
          f'--- dpo_beta:{config_train.dpo_beta}\n'
          f'--- dropout:{config_model.dropout}\n'

          f'--- log_cnt:{config_train.log_cnt}\n'
          f'--- save_cnt:{config_train.save_cnt}\n'
          f'--- num_checkpoint:{config_train.num_checkpoint}\n'
          f'--- train_id:{config_train.id}\n'
          )
    return args,config_train,config_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trainer")

    parser.add_argument("--savename", type=str, default='', help="指定保存的模型名称，暂时还没用上")
    parser.add_argument("--mode", type=str, default='lora', help="lora:lora微调, pre:预训练,sft:微调,dpo:偏好dpo,orpo:sft+偏好orpo")
    parser.add_argument("--data_path", type=str, default='./data/lora_identity.jsonl', help="")  # './data/sft_test.jsonl'

    parser.add_argument("--lora_name", type=str, default='lora_aaa', help="lora名称")
    parser.add_argument("--train_id", type=str, default='20250629_lora_test', help="训练id")
    args,config_train, config_model = set_train_args(parser)
    args.train_id=args.mode+'_'+args.train_id

    if not os.path.exists(config_train.log_out_path):
        os.makedirs(config_train.log_out_path,exist_ok=True)
    if not os.path.exists(config_train.out_path):
        os.makedirs(config_train.out_path,exist_ok=True)
    print(f'Trainer start ---{datetime.now()}')
    if args.mode == 'pre':
        print(f'pre---开始')
        config_train.train_type = args.mode
        model_file_path = f'{config_train.out_path}/{config_train.train_type}_{config_model.hidden_dim}.pth'

        if config_train.ddp:
            import torch.multiprocessing as mp

            # 设置启动方式为 'spawn'（Windows 和 macOS 必须使用 spawn）
            mp.spawn(
                pre_train_wrapper,
                args=(model_file_path, config_model, config_train),
                nprocs=config_train.world_size,
                join=True
            )
        else:
            pre_train_wrapper(0, model_file_path, config_model, config_train)
        print(f'pre---结束')

    elif args.mode == 'sft':
        print(f'sft---开始')
        config_train.train_type = args.mode
        model_file_path = f'{config_train.out_path}/{config_train.train_type}_{config_model.hidden_dim}.pth'
        if not os.path.exists(model_file_path):
            model_file_path = f'{config_train.out_path}/pre_{config_model.hidden_dim}.pth'
        if config_train.ddp:
            import torch.multiprocessing as mp

            # 设置启动方式为 'spawn'（Windows 和 macOS 必须使用 spawn）
            mp.spawn(
                sft_train_wrapper,
                args=(model_file_path, config_model, config_train),
                nprocs=config_train.world_size,
                join=True
            )
        else:
            sft_train_wrapper(0, model_file_path, config_model, config_train)
        print(f'sft--结束')

    elif args.mode == 'dpo':
        print(f'dpo---开始')
        config_train.train_type = args.mode
        model_file_path = f'{config_train.out_path}/{config_train.train_type}_{config_model.hidden_dim}.pth'
        if not os.path.exists(model_file_path):
            model_file_path = f'{config_train.out_path}/sft_{config_model.hidden_dim}.pth'
        if not os.path.exists(model_file_path):
            model_file_path = f'{config_train.out_path}/pre_{config_model.hidden_dim}.pth'
        if config_train.ddp:
            import torch.multiprocessing as mp

            # 设置启动方式为 'spawn'（Windows 和 macOS 必须使用 spawn）
            mp.spawn(
                dpo_train_wrapper,
                args=(model_file_path, config_model, config_train),
                nprocs=config_train.world_size,
                join=True
            )
        else:
            dpo_train_wrapper(0, model_file_path, config_model, config_train)
        print(f'dpo--结束')

    elif args.mode == 'orpo':
        print(f'orpo---开始')
        config_train.train_type = args.mode
        model_file_path = f'{config_train.out_path}/{config_train.train_type}_{config_model.hidden_dim}.pth'
        if not os.path.exists(model_file_path):
            model_file_path = f'{config_train.out_path}/sft_{config_model.hidden_dim}.pth'
        if not os.path.exists(model_file_path):
            model_file_path = f'{config_train.out_path}/pre_{config_model.hidden_dim}.pth'
        if config_train.ddp:
            import torch.multiprocessing as mp

            # 设置启动方式为 'spawn'（Windows 和 macOS 必须使用 spawn）
            mp.spawn(
                orpo_train_wrapper,
                args=(model_file_path, config_model, config_train),
                nprocs=config_train.world_size,
                join=True
            )
        else:
            orpo_train_wrapper(0, model_file_path, config_model, config_train)
        print(f'orpo--结束')

    elif args.mode == 'lora':
        print(f'lora---开始')

        if args.lora_name != '':
            config_train.train_type = args.mode
            config_train.lora_name = args.lora_name
            config_train.lora_target = ['q_proj','k_proj','v_proj','o_proj']

            model_file_path = f'{config_train.out_path}/sft_{config_model.hidden_dim}.pth'
            if not os.path.exists(model_file_path):
                model_file_path = f'{config_train.out_path}/pre_{config_model.hidden_dim}.pth'

            if config_train.ddp:
                import torch.multiprocessing as mp
                # 设置启动方式为 'spawn'（Windows 和 macOS 必须使用 spawn）
                mp.spawn(
                    lora_train_wrapper,
                    args=(model_file_path, config_model, config_train),
                    nprocs=config_train.world_size,
                    join=True
                )
            else:
                lora_train_wrapper(0, model_file_path, config_model, config_train)
        print(f'lora--结束')

    elif args.mode == 'checkpoint':
        print(f'checkpoint---开始')
        config_train.train_type='pre'
        with open('./output/log/pre_log_512.log', 'r', encoding='utf-8') as fw:
            logs = fw.readlines()
        model_file_path = f'{config_train.out_path}/pre_{config_model.hidden_dim}.pth'
        config_train.num_checkpoint=100
        _,model = init_model_tokenizer(config_model,model_file_path)
        step=1000
        point_serial = step // config_train.num_checkpoint
        point_dir_path = f'{config_train.out_path}/checkpoint/{config_train.train_type}_{config_train.id}'
        checkpoint_save(point_dir_path,point_serial,config_train, config_model, model,logs)

        checkpoint_load(point_dir_path)
        print(f'checkpoint--结束')

    else:
        print(f'---不支持{args.mode}模式---')

    print(f'Trainer end ---{datetime.now()}')
