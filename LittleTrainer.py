import argparse
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch import optim, nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from LittleMind import modelConfig, LittleForCausalLM
from LittleDataSet import dataset_pre, dataset_sft


class trainConfig():
    model_type = "xlm_model_config"

    def __init__(self,
                 data_path='./data',
                 out_path='./output/model',
                 log_out_path='./output/log',
                 device='cpu',
                 dtype='float16',  # bfloat16 GPU支持，float16 cpu和GPU都支持
                 epochs: int = 1,
                 batch_size=1,
                 accumulation_steps=5,
                 grad_clip=1.0,
                 log_interval=100,
                 save_interval=2000,
                 log_cnt=100,
                 save_cnt=10,
                 num_workers=1,
                 learning_rate=5e-5,
                 train_type='pretrain',  # pretrain,sft,dpo
                 # 多卡分布式
                 ddp=False,
                 world_size=1,
                 num_machine=1,
                 dpp_master_addr='127.0.0.1',
                 dpp_port=37001,
                 **kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_path = data_path
        self.out_path = out_path
        self.log_out_path = log_out_path
        self.dtype = dtype
        self.accumulation_steps = accumulation_steps
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_cnt = log_cnt
        self.save_cnt = save_cnt
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.train_type = train_type
        self.device = device
        self.ddp = ddp
        self.world_size = world_size
        self.num_machine = num_machine
        self.dpp_master_addr = dpp_master_addr
        self.dpp_port = dpp_port


def train_file(model, dataloader, config_train: trainConfig, config_model: modelConfig, is_dir: bool):
    print(f'开始处理文件---{config_train.data_path}---{datetime.now()}')
    if is_dir:
        train(model, config_train, config_model, dataloader, 0)
    else:
        for epoch in range(config_train.epochs):
            if config_train.ddp:
                dataloader.sampler.set_epoch(epoch)
            train(model, config_train, config_model, dataloader, epoch)
    save_model(model, config_model, config_train)
    print(f'文件处理完成---{config_train.data_path}---{datetime.now()}')


def train_dir(model, tokenizer, config_train: trainConfig, config_model: modelConfig, local_rank):
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
                dataloader = init_data(tokenizer, config_model,config_train, local_rank)
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


def train(model, config_train: trainConfig, config_model: modelConfig, train_loader, epoch):

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    # 当训练在CPU上进行时，使用nullcontext()。这是一个不做任何事情的上下文管理器，意味着不会对计算类型进行任何转换。
    # 它允许你在不支持或不需要混合精度的环境中（如仅使用CPU时），仍然可以使用相同的代码结构而不引发错误。
    ctx = nullcontext() if config_train.device == "cpu" else torch.cuda.amp.autocast()

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler(enabled=(config_train.dtype in ['float16', 'bfloat16']))
    else:
        scaler = torch.amp.GradScaler(device=config_train.device,
                                      enabled=(config_train.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=config_train.learning_rate)

    # 批次数量
    iter_per_epoch = len(train_loader)
    config_train.log_interval = iter_per_epoch // config_train.log_cnt #打印500个日志
    config_train.save_interval = iter_per_epoch // config_train.save_cnt #保存50次
    config_train.log_interval = 1 if config_train.log_interval < 1 else config_train.log_interval
    config_train.save_interval = iter_per_epoch if config_train.save_interval < 1 else config_train.save_interval
    config_train.accumulation_steps = 1 if iter_per_epoch < config_train.accumulation_steps else config_train.accumulation_steps
    start_time = datetime.now()
    ratio_embed = 0.0
    ratio_p = 0.0
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

        if (step + 1) % config_train.accumulation_steps == 0:

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad = param.grad
            #         mean = grad.abs().mean().item()
            #         max_val = grad.abs().max().item()
            #         ratio = max_val / mean if mean > 0 else float('inf')
            #         print(f"grad1---{name}: Mean={mean:.6f}, Max={max_val:.6f}, Ratio={ratio:.2f}")

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config_train.grad_clip)
            ratio_embed = 0.0
            ratio_p = 0.0
            cnt = 0
            for name, param in model.named_parameters():
                if name == 'model.embedding.weight':
                    if param.grad is not None:
                        grad = param.grad
                        mean = grad.abs().mean().item()
                        max_val = grad.abs().max().item()
                        ratio_embed = max_val / mean if mean > 0 else float('inf')
                else:

                    if param.grad is not None:
                        grad = param.grad
                        mean = grad.abs().mean().item()
                        max_val = grad.abs().max().item()
                        ratio_p += max_val / mean if mean > 0 else float('inf')
                        cnt += 1

            ratio_p = ratio_p / cnt
            # print(f"grad2---{name}: Mean={mean:.6f}, Max={max_val:.6f}, Ratio={ratio:.2f}")

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % config_train.log_interval == 0:
            spend_time = datetime.now() - start_time
            # , Total_Time:{}min ,Remaining_Time:{},file:{}
            print(
                'Epoch:[{}/{}],batch_step:({}/{}), loss:{:.3f}, lr:{:.12f},grad_ratio:{:.3f}-{:.3f}'.format(
                    epoch + 1,
                    config_train.epochs,
                    step + 1,
                    iter_per_epoch,
                    loss.item() * config_train.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    ratio_embed,
                    ratio_p
                    # spend_time / (step + 1) * iter_per_epoch // 60,
                    # spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    # trainfile
                )
            )
            log_content = ('"total_epoch":{},"epoch":{},"batch_step":{},'
                           '"total_step_per_epoch":{}, "loss":{:.3f}, "lr":{:.12f} ,"total_time":"{}" ,'
                           '"use_time":"{}","trainfile":"{}","time":"{}"').format(
                config_train.epochs,
                epoch + 1,
                step + 1,
                iter_per_epoch,
                loss * config_train.accumulation_steps,
                lr,
                spend_time / (step + 1) * iter_per_epoch // 60,
                spend_time // 60,
                config_train.data_path,
                datetime.now()
            )
            log_path = f'{config_train.log_out_path}/{config_train.train_type}_log_{config_model.hidden_dim}.log'
            log_train(log_content, log_path)

        if (step + 1) % config_train.save_interval == 0 and (not config_train.ddp or dist.get_rank() == 0):
            save_model(model, config_model, config_train)
        if (step + 1) % (config_train.save_interval * 2) == 0 and (not config_train.ddp or dist.get_rank() == 0):
            save_model(model, config_model, config_train, is_bak=True)


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
        print(f'训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e7:.3f} 千万')
    except BaseException:
        print(f"Error: {BaseException}")
    return tokenizer, model


def init_data(tokenizer, config_model,config_train, local_rank):
    if config_train.train_type == 'pre':
        dataset = dataset_pre(config_train.data_path, tokenizer, config_model.max_seq_len)
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
        shuffle=False,
        prefetch_factor=2,  # 提前预取因子
        persistent_workers=True,  # 保持工作进程持久化
        num_workers=config_train.num_workers,  # 进程数
        sampler=train_sampler
    )
    return dataloader


def save_model(model, config_model, config_train, is_bak: bool = False):
    bak = '_bak' if is_bak else ''
    save_path = f'{config_train.out_path}/{config_train.train_type}_{config_model.hidden_dim}{bak}.pth'
    save_train_model(model, save_path)
    log_content = f'"text":{save_path},"time":{datetime.now()}'
    log_path = f'{config_train.log_out_path}/{config_train.train_type}_save_log_{config_model.hidden_dim}.log'
    log_train(log_content, log_path)


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


def log_train(log_content, log_file):
    log_content = '{' + log_content + '}\n'
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(log_content)


def dir_read_savelog(savelog_path):
    process_files = []
    if os.path.exists(savelog_path):
        with open(savelog_path, 'r', encoding='utf-8') as log:
            for i, line in enumerate(log):
                data = json.loads(line.strip().replace('\\', '/'))
                process_files.append(data["savefile"])
    return process_files


def pre_train(model, tokenizer, config_train: trainConfig, config_model: modelConfig, local_rank):
    if os.path.isfile(config_train.data_path):  # 文件
        dataloader = init_data(tokenizer, config_model,config_train, local_rank)
        train_file(model, dataloader, config_train, config_model, is_dir=False)
    else:  # 目录
        train_dir(model, tokenizer, config_train, config_model, local_rank)


def sft_train(model, tokenizer, config_train: trainConfig, config_model: modelConfig, local_rank):
    if os.path.isfile(config_train.data_path):  # 文件
        dataloader = init_data(tokenizer,config_model, config_train, local_rank)
        train_file(model, dataloader, config_train, config_model, is_dir=False)
    else:  # 目录
        train_dir(model, tokenizer, config_train, config_model, local_rank)


def pre_train_wrapper(local_rank, model_file_path, config_model, config_train):
    print(f'pre_train_wrapper local_rank:{local_rank}')
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--batch_size", type=int, default=1, help="训练批次")
    parser.add_argument("--num_workers", type=int, default=1, help="数据集dataloader加载进程")
    parser.add_argument("--acc_steps", type=int, default=2, help="梯度累计次数")
    parser.add_argument("--lr", type=float, default=7e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=1, help="训练次数")
    parser.add_argument("--logcnt", type=int, default=100, help="日志次数")
    parser.add_argument("--savecnt", type=int, default=10, help="保存次数")
    parser.add_argument("--ddp", type=int, default=0, help="分布式 1开启 0关闭")
    parser.add_argument("--world_size", type=int, default=1, help="gpu卡数")
    parser.add_argument("--dim", type=int, default=512, help="embedding维度")
    parser.add_argument("--num_layers", type=int, default=8, help="注意力层数")
    parser.add_argument("--mode", type=str, default='sft', help="pre:预训练,sft:微调")
    parser.add_argument("--data_path", type=str, default='./data/sft_test.jsonl', help="")  # './data/sft_test.jsonl'
    args = parser.parse_args()

    print(f'Trainer start ---{datetime.now()}')
    #训练配置
    config_train = trainConfig()
    config_train.train_type = args.mode
    config_train.data_path = args.data_path
    config_train.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_train.epochs = args.epochs
    config_train.batch_size = args.batch_size
    config_train.accumulation_steps = args.acc_steps
    config_train.num_workers = args.num_workers
    config_train.learning_rate = args.lr
    config_train.log_cnt = args.logcnt
    config_train.save_cnt = args.savecnt
    config_train.ddp = (args.ddp==1)
    config_train.world_size = args.world_size
    # 模型配置
    config_model = modelConfig()
    config_model.hidden_dim = args.dim
    config_model.num_attention_layers = args.num_layers
    config_model.device = config_train.device
    config_model.dropout = 0.0

    print(f'--- device:{config_train.device}--{config_model.device}\n'
          f'--- epochs:{config_train.epochs}\n'
          f'--- batch_size:{config_train.batch_size}\n'
          f'--- accumulation_steps:{config_train.accumulation_steps}\n'
          f'--- num_workers:{config_train.num_workers}\n'
          f'--- lr:{config_train.learning_rate}\n'
          f'--- ddp:{config_train.ddp}\n'
          f'--- gpu_size:{torch.cuda.device_count()}\n'
          f'--- world_size:{config_train.world_size}\n'
          f'--- dropout:{config_model.dropout}\n'
          f'--- log_cnt:{config_train.log_cnt}\n'
          f'--- save_cnt:{config_train.save_cnt}\n'
          )


    if config_train.train_type == 'pre':
        print(f'pre---开始')
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

    elif config_train.train_type == 'sft':
        print(f'sft---开始')
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

    else:
        print(f'---不支持{args.mode}模式---')

    print(f'Trainer end ---{datetime.now()}')
