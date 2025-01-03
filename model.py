import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    """
    层归一化（Layer Normalization）模块，但带有可选的偏置。

    PyTorch 的标准 LayerNorm 不支持简单地设置 bias=False，因此这个类实现了带有可选偏置的层归一化。
    """

    def __init__(self, ndim, bias):
        """
        初始化 LayerNorm 模块。

        参数:
            ndim (int): 输入的维度，用于初始化权重和偏置。
            bias (bool): 是否使用偏置。如果为 True，则使用偏置；否则，不使用偏置。
        """
        super().__init__()
        # 初始化权重参数，形状为 (ndim,)
        self.weight = nn.Parameter(torch.ones(ndim))
        # 如果使用偏置，则初始化偏置参数；否则，设置为 None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        前向传播函数，执行层归一化操作。

        参数:
            input (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过层归一化处理后的张量。
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制（Causal Self-Attention）模块。

    该模块实现了多头自注意力机制，并确保每个位置只能关注到其左侧的序列元素，实现了因果关系。
    """

    def __init__(self, config):
        """
        初始化因果自注意力模块。

        参数:
            config: 配置对象，包含以下属性：
                - n_embd (int): 嵌入维度。
                - n_head (int): 多头注意力的头数。
                - block_size (int): 序列的最大长度。
                - bias (bool): 是否使用偏置。
                - dropout (float): Dropout 概率。
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # 线性层，用于计算键（key）、查询（query）和值（value），所有头的参数一起计算
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        # 线性层，用于输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        # Dropout 层，用于注意力权重和残差连接
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 多头注意力的头数
        self.n_head = config.n_head
        # 嵌入维度
        self.n_embd = config.n_embd
        # Dropout 概率
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # 创建因果掩码，确保注意力只应用于输入序列的左侧
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        前向传播函数，执行因果自注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, n_embd)。

        返回:
            torch.Tensor: 经过因果自注意力机制处理后的张量，形状为 (batch_size, sequence_length, n_embd)。
        """
        # 获取批量大小 (B)、序列长度 (T) 和嵌入维度 (C)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 计算键 (k)、查询 (q) 和值 (v) 对于所有头，并将头作为批量维度
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 因果自注意力机制
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # 使用 Flash Attention 加速注意力计算
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            # 手动实现注意力机制
            # 计算注意力得分
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # 应用因果掩码
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # 应用 softmax 激活函数
            att = F.softmax(att, dim=-1)
            # 应用 Dropout
            att = self.attn_dropout(att)
            # 计算最终输出
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 重塑输出张量
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # 输出投影
        # 应用残差连接和 Dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    多层感知机（MLP）模块，用于 Transformer 模型中的前馈神经网络部分。

    MLP 模块由两个线性层和一个 GELU 激活函数组成，应用于 Transformer 块的残差连接之后。
    """

    def __init__(self, config):
        """
        初始化 MLP 模块。

        参数:
            config: 配置对象，包含以下属性：
                - n_embd (int): 嵌入维度。
                - dropout (float): Dropout 概率。
                - bias (bool): 是否使用偏置。
        """
        super().__init__()
        # 第一个线性层，将维度扩展 4 倍
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # GELU 激活函数
        self.gelu    = nn.GELU()
        # 第二个线性层，将维度恢复为原始大小
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Dropout 层
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        前向传播函数，执行 MLP 操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过 MLP 处理后的张量。
        """
        # 第一个线性层
        x = self.c_fc(x)
        # GELU 激活函数
        x = self.gelu(x)
        # 第二个线性层
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer 块（Block），是 GPT 模型的基本构建单元。

    每个 Transformer 块包含两个子层：多头自注意力机制和前馈神经网络，并通过残差连接和层归一化进行增强。
    """

    def __init__(self, config):
        """
        初始化 Transformer 块。

        参数:
            config: 配置对象，包含以下属性：
                - n_embd (int): 嵌入维度。
                - dropout (float): Dropout 概率。
                - bias (bool): 是否使用偏置。
        """
        super().__init__()
        # 第一个层归一化层，用于注意力机制输入
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 多头自注意力机制
        self.attn = CausalSelfAttention(config)
        # 第二个层归一化层，用于前馈神经网络输入
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # 前馈神经网络
        self.mlp = MLP(config)

    def forward(self, x):
        """
        前向传播函数，执行 Transformer 块的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过 Transformer 块处理后的张量。
        """
        # 第一个残差连接：注意力机制 + 层归一化
        x = x + self.attn(self.ln_1(x))
        # 第二个残差连接：前馈神经网络 + 层归一化
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """
    GPT 模型的配置类，用于定义模型的各个参数。

    参数:
        block_size (int, 可选): 输入序列的最大长度，默认为 1024。
        vocab_size (int, 可选): 词汇表的大小，默认为 50304（GPT-2 的词汇表大小为 50257，为了效率填充到最近的 64 的倍数）。
        n_layer (int, 可选): Transformer 块的层数，默认为 12。
        n_head (int, 可选): 多头注意力的头数，默认为 12。
        n_embd (int, 可选): 嵌入维度，默认为 768。
        dropout (float, 可选): Dropout 概率，默认为 0.0。
        bias (bool, 可选): 是否使用偏置，默认为 True（与 GPT-2 相同）。设置为 False 时，性能略好且速度更快。
    """
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):
    """
    GPT 模型类，实现了基于 Transformer 的生成式预训练模型。

    GPT 模型由嵌入层、位置编码、多个 Transformer 块、层归一化以及语言模型头组成。
    通过权重绑定（Weight Tying）技术，将嵌入层和语言模型头的权重共享，以提高模型性能并减少参数量。
    """

    def __init__(self, config):
        """
        初始化 GPT 模型。

        参数:
            config: 配置对象，包含以下属性：
                - vocab_size (int): 词汇表的大小。
                - block_size (int): 输入序列的最大长度。
                - n_embd (int): 嵌入维度。
                - n_layer (int): Transformer 块的层数。
                - dropout (float): Dropout 概率。
                - bias (bool): 是否使用偏置。
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 定义 Transformer 模块字典，包括嵌入层、位置编码、Dropout、多个 Transformer 块和层归一化
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # 词嵌入层
            wpe = nn.Embedding(config.block_size, config.n_embd), # 位置编码嵌入层
            drop = nn.Dropout(config.dropout), # Dropout 层
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # 多个 Transformer 块
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # 最终层归一化层
        ))

        # 语言模型头（线性层），用于将 Transformer 块的输出映射到词汇表
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权重绑定：将嵌入层的权重与语言模型头的权重共享
        # 这意味着嵌入层和语言模型头共享相同的权重矩阵
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        # 初始化所有权重
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # 对残差投影应用特殊的初始化方法，根据 GPT-2 论文
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        # 输出模型参数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        返回模型中的参数数量。

        参数:
            non_embedding (bool, 可选): 是否排除嵌入层的参数。默认为 True。

        返回:
            int: 参数数量。
        """
        # 计算所有参数的数量
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # 排除位置编码嵌入层的参数
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        初始化模型权重。

        参数:
            module (nn.Module): 需要初始化的模块。
        """
        if isinstance(module, nn.Linear):
            # 线性层的权重初始化为正态分布
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # 线性层的偏置初始化为零
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层的权重初始化为正态分布
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播函数。

        参数:
            idx (torch.Tensor): 输入的索引张量，形状为 (batch_size, sequence_length)。
            targets (Optional[torch.Tensor], 可选): 目标张量，形状与 idx 相同。

        返回:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 模型输出和损失。
        """
        device = idx.device
        # 获取批量大小 (b) 和序列长度 (t)
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # 生成位置索引，形状为 (t)
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        # 前向传播 GPT 模型本身
        # 词嵌入，形状为 (b, t, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # 位置编码嵌入，形状为 (t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # 加上位置编码并应用 Dropout
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            # 通过多个 Transformer 块
            x = block(x)
        # 最终层归一化
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # 如果提供了目标，计算损失
            # 通过语言模型头计算 logits
            logits = self.lm_head(x)
            # 计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # 如果没有提供目标，只计算最后一个位置的 logits（推理时的优化）
            logits = self.lm_head(x[:, [-1], :]) # 注意：使用列表 [-1] 来保留时间维度
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """
        修改模型的块大小。

        参数:
            block_size (int): 新的块大小，必须小于或等于当前块大小。
        """
        assert block_size <= self.config.block_size
        # 更新块大小
        self.config.block_size = block_size
        # 裁剪位置编码嵌入层
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                # 裁剪注意力
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        从预训练的 Hugging Face GPT2 模型加载权重。

        参数:
            model_type (str): 预训练模型的类型，支持 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'。
            override_args (dict, 可选): 用于覆盖默认配置参数的字典。目前只支持覆盖 dropout 率。

        返回:
            GPT: 初始化后的 GPT 模型实例。
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # 默认覆盖参数为空字典
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        # 目前只支持覆盖 dropout 率
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        # 根据 model_type 确定模型的层数、头数和嵌入维度
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")

        # 对于 GPT 模型检查点，词汇表大小始终为 50257
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        # 对于 GPT 模型检查点，块大小始终为 1024
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # 对于 GPT 模型检查点，始终为 True
        config_args['bias'] = True # always True for GPT model checkpoints

        # we can override the dropout rate, if desired
        # 如果提供了覆盖参数，则覆盖 dropout 率
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # create a from-scratch initialized minGPT model
        # 创建一个从头初始化的 minGPT 模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # 丢弃这些掩码/缓冲区，不是参数
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # 初始化 Hugging Face/transformers 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # 复制权重，同时确保所有参数对齐，并且名称和形状匹配
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # 基本上，OpenAI 的checkpoints使用 "Conv1D" 模块，但我们只希望使用普通的线性层
        # 这意味着我们导入时必须转置这些权重
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对于需要转置的 Conv1D 权重进行特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 普通的复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        配置优化器。

        参数:
            weight_decay (float): 权重衰减率。
            learning_rate (float): 学习率。
            betas (tuple): AdamW 优化器的 beta 参数。
            device_type (str): 设备类型，例如 'cpu' 或 'cuda'。

        返回:
            Optimizer: 配置好的优化器。
        """
        # 收集所有候选参数
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        # 过滤掉不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 创建优化器组。所有 2D 参数将进行权重衰减，否则不进行。
        # 所有在矩阵乘法 + 嵌入中的权重张量进行衰减，所有偏置和层归一化不进行衰减。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        估计模型浮点运算利用率（MFU），单位为 A100 bfloat16 峰值 FLOPS。

        参数:
            fwdbwd_per_iter (int): 每次前向和后向传播中的前向传播次数。
            dt (float): 每次迭代的时间（秒）。

        返回:
            float: MFU 值。
        """
        # 首先估计每次迭代中执行的浮点运算次数。
        # 参考 PaLM 论文附录 B 作为参考:
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # 将我们的浮点运算吞吐量表示为 A100 bfloat16 峰值浮点运算的比例
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # A100 GPU bfloat16 峰值浮点运算为 312 TFLOPS
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        给定一个索引序列 idx（形状为 (b,t) 的 LongTensor），并完成序列 max_new_tokens 次，
        每次将预测结果重新输入到模型中。大多数情况下，你可能希望确保在 model.eval() 模式下运行此操作。

        参数:
            idx (torch.Tensor): 输入索引序列。
            max_new_tokens (int): 要生成的新的标记数。
            temperature (float, 可选): 温度参数，用于控制生成的多样性。默认为 1.0。
            top_k (int, 可选): 如果提供，则在生成过程中仅考虑前 k 个最可能的标记。

        返回:
            torch.Tensor: 生成后的索引序列。
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # 如果序列上下文增长过长，我们必须将其裁剪到块大小
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # 前向传播模型以获取序列中索引的 logits
            logits, _ = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            # 在序列的最后一步取 logits 并按所需温度缩放
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            # 可选地，仅裁剪 logits 到前 k 个选项
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # apply softmax to convert logits to (normalized) probabilities
            # 应用 softmax 将 logits 转换为（归一化）概率
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            # 将采样的索引追加到运行中的序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
