from json import JSONDecoder

import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import Optional
import math


class modelConfig(PretrainedConfig,JSONDecoder):
    model_type = "xlm_model_config"

    def __init__(self,
                 dropout: float = 0.0,
                 pad_token_id: int = 0,  # 填充标记
                 bos_token_id: int = 1,  # 开始标记
                 eos_token_id: int = 2,  # 结束标记
                 hidden_act: str = 'silu',  # ffn激活函数 relu
                 hidden_dim: int = 512,  # 嵌入维度
                 intermediate_size: int = None,  # decoder 中ffn的隐藏层节点数量，必须被64和hidden_dim整除
                 max_position_embeddings: int = 512,  # 位置编码维度
                 vocab_size: int = 6400,  # 词表大小,暂时写死
                 token_path='./tokenizer',  # tokenizer地址
                 num_attention_heads: int = 8,  # 多头注意力数量
                 num_attention_layers: int = 8,  # decoder层数
                 num_key_value_heads: int = 2,  # gqa k v分组数
                 rms_norm_eps: float = 1e-05,  # rmsnorm层超参数
                 device='cpu',  # cuda cuda:0
                 # RoPE的思想就是让toekn q 和 token k之间旋转不同的角度来表达相对位置信息，
                 # 相近的token间旋转角度小、点乘得到的值更大，离得远的token间旋转角度大，点乘得到的值更小。
                 # 点乘具有的一个特性就是，当token q和token k同时旋转一样的角度，它们之间的夹角不变，
                 # 所以它们之间的关系保持不变，所以RoPE是维护相对位置信息而丢弃绝对位置信息的。
                 rope_theta: int = 100000.0,  # 旋转式位置编码 theta超参数
                 scaled_dot_product_attention=True,  # torch 是否支持缩放点积注意力机制 F.scaled_dot_product_attention
                 flash_attn: bool = True,  # 是否使用flash_attn  结合  F.scaled_dot_product_attention 使用
                 max_seq_len=256,

                 ####################################################
                 # Here are the specific configurations of MOE
                 # When use_moe is false, the following is invalid
                 ####################################################
                 use_moe: bool = False,
                 num_experts_per_tok: int = 2,
                 n_routed_experts: int = 4,
                 n_shared_experts: int = 1,
                 scoring_func: str = 'softmax',
                 aux_loss_alpha: float = 0.1,
                 seq_aux: bool = True,
                 norm_topk_prob: bool = True,

                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_attention_layers = num_attention_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.token_path = token_path
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.flash_attn = flash_attn
        self.scaled_dot_product_attention = scaled_dot_product_attention

        # moe
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


class PositionalEncoding_sincos(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding_sincos, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为缓冲区

    def forward(self, x):
        # 将位置编码添加到输入中 x格式为 batch,seqlen,dim
        return x + self.pe[:, :x.size(1)]


# 代码实现与网上公式有不同，但效果相同
class PositionalEncoding_Rope:
    def __init__(self):
        super(PositionalEncoding_Rope, self).__init__()

    @staticmethod
    def precompute_freqs_cis(dim: int, pos_row_end: int = int(32 * 1024), theta: float = 1e6):
        # 假设数据为dim=512维
        # (torch.arange(0, dim, 2)[: (dim // 2)] 创建一个长度256，步长2的数组，内容为（0，2，4，。。。 ，510）
        # [: (dim // 2)]这段其实没必要
        # 计算三角函数频率,实现三角函数中的值或者可以理解为三角函数图像的频率（一般三角函数sin(x)周期为2pi,频率为1/2pi）
        # [1/10000**(0/512),1/10000**(2/512).....,1/1000**(510/512)]
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        # 位置编码的行，相当于输入的seq_len  pos_row_end >= seq_len
        # [0,1,2,.....1023] 假设 pos_row_end =1024,
        t = torch.arange(pos_row_end, device=freqs.device)
        # 求外积
        # 结果为t变成行，freqs变成列
        # 二维矩阵 (t,freqs)
        # [0*freqs[0],0*freqs[1]....,0*freqs[255]]
        # ...
        # [1023*freqs[0],1023*freqs[1]....,1023*freqs[255]]
        freqs = torch.outer(t, freqs).float()
        # 拼接两个cos矩阵
        # cos[freqs[0],...,freqs[255],freqs[0],...,freqs[255]]
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        # 拼接两个sin矩阵
        # sin[freqs[0],...,freqs[255],freqs[0],...,freqs[255]]
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

        return freqs_cos, freqs_sin

    # 假设为512维度
    # cos[freqs[0],...,freqs[255],freqs[0],...,freqs[255]]
    # sin[freqs[0],...,freqs[255],freqs[0],...,freqs[255]]
    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # 数组后一半设为负数放前面，然后合并前一半
        # [1,2,...  255,    256,...,511] ==>
        # [-256,...-511    ,1,2,...,255]
        def rotate_half(x):
            return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

        # 公式为 0维*cos(freqs[0])- 1维sin(freqs[0]),1维*cos(freqs[1])+0维*sin(freqs[1])...,,,
        # 效果为 0维*cos(freqs[0])-256维sin(freqs[0]),...,256维*cos(freqs[0])+0维*sin(freqs[0])
        # 将1维和256维替换调，两者差不多是一样的
        q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
        # 效果为 0维*cos(freqs[0])-255维sin(freqs[0]),...,255维*cos(freqs[0])+0维*sin(freqs[0])
        k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
        return q_embed, k_embed

    def test(self, start_pos, seq_length, xq, xk):
        cos = self.freqs_cos[start_pos:start_pos + seq_length]
        sin = self.freqs_sin[start_pos:start_pos + seq_length]

        xq, xk = PositionalEncoding_Rope.apply_rotary_pos_emb(xq, xk, cos[:seq_length], sin[:seq_length])
        return xq, xk


class RmsNorm(nn.Module):
    def __init__(self, dim: int, ep: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.ep = ep
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x1 = x.pow(2)
        x2 = x1.mean(-1, keepdim=True)
        x3 = torch.rsqrt(x2 + self.ep)
        output = x * x3
        output = self.weight * output
        return output


from torch import nn


class FeedForward(nn.Module):
    def __init__(self, config: modelConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_dim * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = transformers.activations.ACT2FN[config.hidden_act]

    def forward(self, x):
        x_1 = self.gate_proj(x)  # 升intermediate_size
        y = self.act_fn(x_1)  # 激活

        a = self.up_proj(x)  # 升intermediate_size

        b = y * a  # 点积
        c = self.down_proj(b)  # 降hidden_dim

        return self.dropout(c)


class Attention(nn.Module):
    def __init__(self, config_args: modelConfig):
        super().__init__()
        self.num_attention_heads = config_args.num_attention_heads
        self.num_key_value_heads = config_args.num_key_value_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads

        self.hidden_dim = config_args.hidden_dim
        self.head_dim = self.hidden_dim // self.num_attention_heads

        self.flash_attn = config_args.flash_attn
        self.scaled_dot_product_attention = config_args.scaled_dot_product_attention

        self.rope_theta = config_args.rope_theta
        self.dropout = config_args.dropout

        self.q_proj = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim // self.n_rep, bias=False)
        self.o_proj = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)
        self.att_dropout = nn.Dropout(self.dropout)
        self.res_dropout = nn.Dropout(self.dropout)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        bs, slen, num_key_value_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
        )

    def forward(self, x, pos, att_mask=None):
        batch_size, seq_len, embed_dim = x.shape

        # 线性变换
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # 将输入张量分割为多个头
        # 输入形状: (batch_size, seq_length, hidden_dim)
        # 输出形状: (batch_size, num_heads, seq_length, hidden_dim/num_heads)
        xq = xq.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)  # .transpose(1, 2)
        xk = xk.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)  # .transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)  # .transpose(1, 2)

        cos, sin = pos
        xq, xk = PositionalEncoding_Rope.apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        xq, xk, xv = (
            xq.transpose(1, 2),
            self.repeat_kv(xk, self.n_rep).transpose(1, 2),
            self.repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 使用加速atten
        if self.flash_attn and self.scaled_dot_product_attention:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if att_mask is not None:
                attn_mask = att_mask.view(batch_size, 1, 1, -1).expand(batch_size, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if att_mask is not None else None

            # is_causal=True 代表会自动构建mask
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p,
                                                    is_causal=True)
        else:
            # Q * K^T  / 开平方 d
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 掩码  diagonal=1代表后移一位 inf
            mask_tmp = torch.full((seq_len, seq_len), float("-inf"), device=scores.device)
            mask_tmp = torch.triu(mask_tmp, diagonal=1).unsqueeze(0).unsqueeze(0)
            scores = scores + mask_tmp  # scores+mask

            if att_mask is not None:
                extended_attention_mask = att_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            # 注意力系数
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.att_dropout(scores)
            # 注意力系数 * V
            output = scores @ xv

        # 输入形状: (batch_size,num_heads,seq_length,d_model/hidden_dim)
        # 输出形状: (batch_size, seq_length, hidden_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)
        output = self.res_dropout(output)
        return output


class LittleBlock(nn.Module):
    def __init__(self, blockid, config_args: modelConfig):
        super().__init__()
        self.block_id = blockid
        self.att = Attention(config_args)
        self.att_rms_norm = RmsNorm(config_args.hidden_dim, config_args.rms_norm_eps)
        self.fnn_rms_norm = RmsNorm(config_args.hidden_dim, config_args.rms_norm_eps)
        self.fnn = FeedForward(config_args)

    def forward(self, x, pos):
        y = self.att_rms_norm(x)  # 归一化
        y = self.att(y, pos)
        y = y + x  # 残差

        z = self.fnn_rms_norm(y)  # 归一化
        z = self.fnn(z)
        z = z + y  # 残差

        return z


class LittleModel(nn.Module):
    def __init__(self, config_args: modelConfig):

        super().__init__()
        self.config_model = config_args
        self.num_attention_layers = config_args.num_attention_layers

        self.embedding = nn.Embedding(config_args.vocab_size, config_args.hidden_dim)
        # self.pos = PositionalEncoding_Rope(config_args)

        self.dropout = nn.Dropout(config_args.dropout)
        self.norm = RmsNorm(config_args.hidden_dim, ep=config_args.rms_norm_eps)
        self.laysers = nn.ModuleList([LittleBlock(a, config_args) for a in range(self.num_attention_layers)])

        freqs_cos, freqs_sin = PositionalEncoding_Rope.precompute_freqs_cis(
            dim=config_args.hidden_dim // config_args.num_attention_heads,
            pos_row_end=config_args.max_position_embeddings,
            theta=config_args.rope_theta)

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


    def forward(self, input_ids):
        x = self.embedding(input_ids)
        y = self.dropout(x)
        # y = self.pos(x)
        pos = (self.freqs_cos, self.freqs_sin)

        for layer in self.laysers:
            y = layer(y, pos)
        z = self.norm(y)

        return z


class LittleForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = modelConfig

    def __init__(self, config_args: modelConfig):

        self.config_args = config_args
        super().__init__(self.config_args)
        self.model = LittleModel(self.config_args)

        self.lm_head = nn.Linear(self.config_args.hidden_dim, self.config_args.vocab_size, bias=False)
        self.model.embedding.weight = self.lm_head.weight

        self.OUT = CausalLMOutputWithPast()


    def forward(self, input_ids: Optional[torch.Tensor] = None, **args):
        out = self.model(input_ids=input_ids)
        out1 = self.lm_head(out)
        # return out1
        # 用于缓存kvcache
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(h[:, slice_indices, :])
        # self.OUT.__setitem__('last_hidden_state', out)
        self.OUT.__setitem__('logits', out1)
        # self.OUT.__setitem__('aux_loss', 0)
        self.OUT.__setitem__('past_key_values', None)
        return self.OUT
