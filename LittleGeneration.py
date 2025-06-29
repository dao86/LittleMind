import argparse
from datetime import datetime

import torch
from transformers import AutoTokenizer, TextStreamer

from LittleMind import LittleForCausalLM, modelConfig, RmsNorm, Little_Lora
from LittleTrainer import trainConfig


def init_train(config_model, model_file_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f'vocab_size:{config_model.vocab_size},tokenizer_len:{len(tokenizer)}')
    model = LittleForCausalLM(config_model)
    load_data = torch.load(model_file_path, map_location=config_model.device)
    model.load_state_dict(load_data)
    print(
        f'model:{model_file_path}---参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.3f} B')

    if config_train.lora_name!='':
        lora_model_file_path = f'{config_train.out_path}/{config_train.lora_name}.pth'
        Little_Lora.add_lora(model, config_train.lora_name, config_train.lora_rank, config_train.lora_target)
        Little_Lora.load_lora(model, lora_model_file_path, config_model.device, config_train.lora_name)
        lora_params_count = sum(p.numel() for name, p in model.named_parameters() if config_train.lora_name in name)  # LoRA 参数数量
        print(f'{lora_model_file_path}.lora参数量：{lora_params_count / 1e7:.3f} 千万')
    model.to(config_model.device)
    return tokenizer, model


def test_transformer(model_file_path, config_model: modelConfig, config_train: trainConfig):
    tokenizer, model = init_train(config_model, model_file_path, config_model.token_path)
    input_msg = input('请输入:')
    while input_msg != '退出':
        transformer_generate(model, input_msg, tokenizer, config_model, config_train)
        input_msg = input('请输入:')


def prompt(text, tokenizer, config_model):
    text = tokenizer.bos_token + text
    encoding = tokenizer(
        text,
        max_length=config_model.max_seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding.input_ids.squeeze()
    return input_ids


def transformer_generate(model, messages, tokenizer, config_model, config_train):
    if config_train.train_type == 'pre':
        new_prompt = tokenizer.bos_token + messages
    else:
        messages = [{"role": "user", "content": messages}]
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    print(f'new_prompt:{new_prompt}')
    inputs = tokenizer(
        new_prompt,
        return_tensors="pt",
        truncation=True
    ).to(config_model.device)

    print('回复: ', end='')
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generated_ids = model.generate(
        inputs["input_ids"].to(config_model.device),
        max_length=300,#config_model.max_seq_len,
        # max_new_tokens=config_model.max_seq_len,
        num_return_sequences=1,
        use_cache=False,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        do_sample=True,
        top_p=0.8,
        top_k=20,
        temperature=0.7,
        typical_p=0.8,
        epsilon_cutoff=0.8,
        repetition_penalty=1.5
        # 当 do_sample=True 时，你还可以结合以下参数来进一步控制生成行为， （可选，默认为False）：是否使用采样；否则使用贪婪解码。
        # 结合使用 do_sample=True, top_k=50, 和 top_p=0.95 可以得到既富有创意又相对合理的生成结果。
        # top_k 只从 top-k 个最高概率的词中进行抽样，增加了一定程度的选择性。
        # top_p 核采样（nucleussampling），只考虑累积概率之和(sum)达到某个阈值p的最小集合中的词进行抽样，有助于避免极端低概率事件的发生。
        # temperature: 控制概率分布的平滑度。较高的温度会使分布变得更平滑（增加随机性），而较低的温度会使分布更尖锐（减少随机性）
        # 这个值用于调整下一个令牌的概率。通过改变这个值，你可以控制生成的文本的随机性。
        # 较大的 temperature 值会导致生成的文本更加随机，而较小的 temperature 值则会生成更加确定性的文本。
        # num_beams（可选，默认为1）：束搜索的束数。1表示不使用束搜索。
        # num_beam_groups（可选，默认为1）：将num_beams分成若干组，以确保不同束组的多样性。更多详细信息请参考这篇论文(This Paper)。
        # penalty_alpha（可选）：在对比搜索解码中，平衡模型置信度和退化惩罚的值。
        # use_cache（可选，默认为True）：模型是否应使用过去的最后一个键/值注意力（如果适用于模型）来加速解码。
        # typical_p (浮点数，可选，默认为 1.0)： 局部典型性衡量在给定部分文本生成条件下，预测下一个令牌的概率与随机预测下一个令牌的概率的相似程度。如果设置为小于 1 的浮点数，那么只有最局部典型的令牌集合，其概率之和达到或超过 typical_p，才会在生成过程中保留。
        # epsilon_cutoff (浮点数，可选，默认为 0.0)：
        # 如果设置为在 0 和 1 之间的浮点数，那么只有条件概率大于 epsilon_cutoff 的令牌才会被采样。这个参数可以用来控制生成过程中令牌的选择。
        # eta_cutoff (浮点数，可选，默认为 0.0)：
        # eta 采样是一种局部典型采样和 epsilon 采样的混合。如果设置为在 0 和 1 之间的浮点数，那么一个令牌只有在它大于 eta_cutoff 或 sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))) 时才会被考虑。后者直观上是预期下一个令牌概率，乘以 sqrt(eta_cutoff)。有关更多详细信息，请参阅 Truncation Sampling as Language Model Desmoothing。
        # diversity_penalty (浮点数，可选，默认为 0.0)：
        # 如果生成的某个时间点的令牌与同一组其他束的令牌相同，将从束的分数中减去 diversity_penalty。请注意，只有当 group beam search 启用时，diversity_penalty 才有效。
        # repetition_penalty (浮点数，可选，默认为 1.0)：
        # 重复惩罚参数。1.0 表示没有惩罚。有关更多详细信息，请参阅 this paper。
        # encoder_repetition_penalty (浮点数，可选，默认为 1.0)：
        # 编码器重复惩罚参数。对不是原始输入中的序列施加指数惩罚。1.0 表示没有惩罚。
        # length_penalty (浮点数，可选，默认为 1.0)：
        # 用于基于束生成的指数惩罚。它作为序列长度的指数使用，进而用于除以序列的分数。因为分数是序列的对数似然（即负数），所以 length_penalty > 0.0 促进较长序列，而 length_penalty < 0.0 鼓励较短序列。
        # no_repeat_ngram_size (整数，可选，默认为 0)：
        # 如果设置大于 0，那么在生成过程中，不会重复任何长度为 no_repeat_ngram_size 的 n-gram。这个参数主要用于控制生成文本的多样性，避免重复的 n-gram 导致生成的文本过于单一。
        # bad_words_ids：一个列表，包含不允许生成的 token ID。如果你想获取不应该出现在生成文本中的单词的 token ID，可以使用 tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids。
        # force_words_ids：一个列表，包含必须生成的 token ID。如果给出的是一个 List[List[int]]，那么它被视为一个简单的必须包含的单词列表，与 bad_words_ids 相反。如果给出的是一个 List[List[List[int]]]，则会触发一个 析构约束，其中可以允许每个单词的不同形式。
        # renormalize_logits：一个布尔值，表示是否在应用所有 logits 处理器或 warpers（包括自定义的）后归一化 logits。建议将此标志设置为 True，因为搜索算法假定分数 logits 是归一化的，但一些 logits 处理器或 warpers 会破坏归一化。
        # constraints：一个包含自定义约束的列表，可以添加到生成中，以确保输出在最合适的方式包含由 Constraint 对象定义的某些 token。
        # forced_bos_token_id：一个整数，表示在 decoder_start_token_id 之后强制生成的第一个 token 的 ID。这对于多语言模型（如 mBART）很有用，因为第一个生成的 token 应该是目标语言的 token。
        # forced_eos_token_id：当达到 max_length 时强制生成的最后一个 token 的 ID。可以使用一个列表来设置多个 end-of-sequence token。
        # remove_invalid_values：一个布尔值，表示是否移除模型可能产生的 nan 和 inf 输出，以防止生成方法崩溃。需要注意的是，使用 remove_invalid_values 可能会降低生成速度。
        # exponential_decay_length_penalty：一个元组，用于在生成一定数量的 token 后添加一个指数增长的长度惩罚。元组应该是 (start_index, decay_factor) 的形式，其中 start_index 表示惩罚开始的位置，decay_factor 表示指数衰减因子。
        # suppress_tokens：一个列表，包含在生成过程中将被抑制的 token。SupressTokens logit 处理器会将这些 token 的 log 概率设置为 -inf，以便它们不会被采样。
        # begin_suppress_tokens：一个列表，包含在生成开始时将被抑制的 token。SupressBeginTokens logit 处理器会将这些 token 的 log 概率设置为 -inf，以便它们不会被采样。
        # forced_decoder_ids：一个列表，包含表示生成索引和 token 索引映射的整数对。例如，[[1, 123]] 表示第二个生成的 token 总是索引为 123 的 token。

    )

    # response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # print(f'res:{response}')


def stream_generate(config_train,model, messages, tokenizer, device, cache_use=False, sample=True, stream=True,
                    temperature = 0.75,repetition_penalty=1.5,top_p=0.8,top_k=5,max_len=100):
    model.eval()
    with torch.no_grad():
        if config_train.train_type == 'pre':
            new_prompt = tokenizer.bos_token + messages
        else:
            messages = [{"role": "user", "content": messages}]
            new_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(device)
        input_ids = inputs["input_ids"].to(device)
        is_not_end = True
        out_len = 0
        is_first = True
        past_key_values=None
        while is_not_end:
            if is_first:
                outputs = model(input_ids,None,cache_use)
                is_first=False
            else:
                if cache_use:
                    outputs = model(input_ids[:,-1:], past_key_values, cache_use)
                else:
                    outputs = model(input_ids)

            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=device)

            next_token_logits = next_token_logits / temperature

            #暂不用 next_token_logits[:,input_ids.tolist()[0]]/=repetition_penalty

            input_ids_tmp = input_ids.tolist()[0]
            # 创建惩罚掩码：对于已出现的token，根据penalty调整其logits
            for i, log_index in enumerate(input_ids_tmp):
                if next_token_logits[0][log_index] < 0:
                    # 负logit：惩罚系数应用为乘法（使值更小）
                    next_token_logits[0][log_index] *= repetition_penalty
                else:
                    # 正logit：惩罚系数应用为除法（使值更小）
                    next_token_logits[0][log_index] /= repetition_penalty

            if sample:
                if top_p>0 and top_p<1:#top_p
                    #按概率从高到低排序
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                    #计算累积概率
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    # 找出使其总概率 >= p的前n个索引 第一个true就是，把第一个满足条件的为true的也改为false
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    # 所有为true的索引都设置为-inf
                    next_token_logits[indices_to_remove] = -float('Inf')
                else:#top_k
                    #获取 top-k 的 logit 及其对应的索引
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    #创建一个与 logits 同形状的新张量，只保留 top-k 的位置，其余设为 -inf
                    filter = torch.full_like(next_token_logits, -float('Inf'))
                    filter[:,top_k_indices] = top_k_logits
                    next_token_logits=filter
                    # #将过滤后的 logits 转换为概率分布
                    # probs = torch.nn.functional.softmax(filter, dim=-1)
                    # # 根据概率分布采样一个 token
                    # next_token = torch.multinomial(probs, num_samples=1).item()

                probability = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probability, num_samples=1).squeeze(1)
            else:
                next_tokens = next_token_logits.argmax(-1)

            is_not_end = (next_tokens[0].item() != 2)
            if out_len > max_len:  # 输出大于 max_len 就停止输出
                is_not_end = False
            if stream and is_not_end:
                yield next_tokens[0]

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            out_len += 1

        # tokenizer.de
        # print(f'res : {tokenizer.decode(input_ids[0])}')
        # print(f'res : {input_ids[0]}')


def stream_little(tokenizer, model, config_model,config_train, msg, max_len):
    start_time = datetime.now()
    is_sample=True
    is_stream = True
    temperature = 1
    repetition_penalty = 1
    top_p=1
    top_k=5
    cache_use=True
    token = stream_generate(config_train,model, msg, tokenizer, config_model.device,cache_use, is_sample, is_stream,
                            temperature,repetition_penalty,top_p,top_k,max_len)
    print(f'回答:', end='', flush=True)
    len_out_token = 0
    out_decode_token = [] #输出的解码token数组缓冲区
    for y in token:
        if len(out_decode_token) == 0:#第一次解码
            res = tokenizer.decode(y.tolist(), skip_special_tokens=True)
        else:#第二次开始把输出放到 out_decode_token
            out_decode_token.append(y.tolist())
            res = tokenizer.decode(out_decode_token, skip_special_tokens=True)
        if res == '�':  # 如果解码后 out_decode_token 是乱码，需要等下一个输出，继续合并后一起decode
            out_decode_token.append(y.tolist())
            continue
        print(res, end='', flush=True)
        len_out_token = len(res)
        out_decode_token = []#如果不是乱码，代表解码成功，清空缓冲区
    spend_time = datetime.now()-start_time
    print(f'\ncache_use:{cache_use}, spend_time:{spend_time}, len_out_token:{len_out_token}')


def test_little(model_file_path, config_model,config_train, max_len):
    tokenizer, model = init_train(config_model, model_file_path, config_model.token_path)
    input_text = input('请输入:')
    while input_text != '退出':
        stream_little(tokenizer, model, config_model,config_train, input_text, max_len)
        input_text = input('请输入:')

def new_test_V(sft_path,pre_path,config_model):
    tokenizer, sft_model = init_train(config_model, sft_path, config_model.token_path)
    _, pre_model = init_train(config_model, pre_path, config_model.token_path)

if __name__ == "__main__":
    # rsm = RmsNorm(5)
    # a = torch.tensor([1,2,3,4,5],dtype=float)
    # a1 = rsm(a)
    # b = torch.tensor([[1, 2, 3, 4, 5],[10, 20, 30, 40, 50]],dtype=float)
    # b1 = rsm(b)
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--dim", type=int, default=512, help="")
    parser.add_argument("--layernum", type=int, default=8, help="")

    parser.add_argument("--lora_name", type=str, default='lora_aaa', help="")
    parser.add_argument("--mode", type=str, default='sft', help="")
    args = parser.parse_args()

    config_model = modelConfig()
    config_model.hidden_dim = args.dim
    config_model.num_attention_layers = args.layernum
    config_model.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_train = trainConfig()
    config_train.train_type = args.mode
    config_train.device = config_model.device
    config_train.lora_name = args.lora_name
    if config_train.lora_name!='':
        model_file_path = f'{config_train.out_path}/sft_{config_model.hidden_dim}.pth'
    else:
        model_file_path = f'{config_train.out_path}/{config_train.train_type}_{config_model.hidden_dim}.pth'

    test_little(model_file_path, config_model,config_train, 200)
    # test_transformer(model_file_path, config_model,config_train)
