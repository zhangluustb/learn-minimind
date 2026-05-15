# L10 - RMSNorm 与完整 MiniMindBlock

> **"万丈高楼的每一层结构相同——理解了一层，就理解了整栋楼。"**

---

## 📌 本节目标

1. 理解 LayerNorm 和 RMSNorm 的区别，以及现代 LLM 为何选择 RMSNorm
2. 掌握 Pre-Norm vs Post-Norm 的差异及其对训练稳定性的影响
3. 能完整追踪一个 token 通过 MiniMindBlock 的数据流（shape 变化）
4. 理解残差连接如何解决深层网络的梯度消失问题

---

## 📚 前置知识

- 完成 L07（Attention）、L08（FFN）、L09（KV-Cache）
- 了解反向传播和梯度的基本概念

---

## 正文

### 1. 为什么需要归一化层

深度神经网络训练中存在一个严峻挑战：**内部协变量偏移（Internal Covariate Shift）**。

用大白话说：当网络的浅层参数更新后，深层的输入分布随之改变，深层网络不得不持续"适应"越来越不同的输入，导致训练缓慢甚至失稳。

归一化层的职责就是：**把每层的输入分布稳定下来**，让每一层可以独立地学习，不用担心上游分布的漂移。

#### BatchNorm 的局限

BatchNorm（批归一化）是最早被广泛使用的归一化方法，沿 batch 维度统计均值和方差：

$$\text{BN}(x) = \gamma \cdot \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma_{\text{batch}}^2 + \epsilon}} + \beta$$

问题：
- 序列长度可变时，batch 维度统计意义不明确
- 小 batch size 时统计不稳定
- 推理时需要维护全局均值/方差的移动平均

#### LayerNorm：沿 hidden 维度归一化

LayerNorm 抛弃了 batch 维度的统计，改为对每个 token 的 hidden 维度独立归一化：

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu = \frac{1}{d}\sum_{i=1}^d x_i$，$\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$

需要计算均值 $\mu$ 和方差 $\sigma^2$ 两个统计量，有 $\gamma$（scale）和 $\beta$（bias）两组可学习参数。

---

### 2. RMSNorm：更快的近似

研究者发现：LayerNorm 的"重缩放不变性"主要来自 scale 操作（除以方差），而"重中心化"（减去均值）对效果的贡献有限。

**RMSNorm（Root Mean Square Normalization）** 果断省去均值计算，只保留 RMS（均方根）缩放：

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$$

> **类比**：LayerNorm 是"先调零点再调增益"的放大器，RMSNorm 是"只调增益"的放大器。通常增益才是关键，零点偏移影响不大。

在 minimind 中的实现非常简洁：

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # γ：可学习的缩放因子，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算 RMS（用 rsqrt 代替 sqrt 提高效率）
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms_inv * self.weight
```

**三种归一化方式对比：**

| 特性 | BatchNorm | LayerNorm | RMSNorm |
|------|-----------|-----------|---------|
| 统计维度 | Batch 维 | Hidden 维 | Hidden 维 |
| 计算均值 | ✅ | ✅ | ❌（省略）|
| 计算方差 | ✅ | ✅ | ✅（近似）|
| 可学习参数 | γ, β | γ, β | γ（仅 scale）|
| 速度 | 中 | 中 | **快 ~15-20%** |
| 效果 | 好 | 好 | **相当或更好** |
| NLP 适用性 | ❌ | ✅ | ✅ |

---

### 3. Pre-Norm vs Post-Norm：归一化的"位置"之争

原始 Transformer 论文（2017 年）使用 **Post-Norm**：先做 sublayer 计算，再归一化：

$$\text{Post-Norm: } x_{out} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

现代大型 LLM（LLaMA、minimind、GPT-3 等）几乎全部转向 **Pre-Norm**：先归一化，再做 sublayer 计算：

$$\text{Pre-Norm: } x_{out} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

**为什么 Pre-Norm 训练更稳定？**

关键在于**残差路径**。Post-Norm 中，每次都对残差结果做 norm，这实际上截断了梯度的"直通路"。

Pre-Norm 中，残差路径（`x + ...`）的 `x` 部分不经过任何 norm，梯度可以从 loss 沿着残差加法**一路直传**到浅层，梯度消失问题大幅缓解。

```
Post-Norm 梯度路径：
loss → LN → (x + F(x)) → LN → ... → input
         ↑ norm 每次都会变换梯度的尺度

Pre-Norm 梯度路径：
loss → (x + F(LN(x))) → (x + F(LN(x))) → ... → input
         ↑ 残差路径的梯度可以直通，"1"这一项保证梯度不消失
```

Pre-Norm 允许使用更大的学习率（通常 2-5 倍），训练更快更稳定。

---

### 4. MiniMindBlock：把所有零件组装在一起

现在是"揭秘时刻"！把 L07（Attention）、L08（FFN）和本节的 RMSNorm + 残差连接组装起来，就得到了一个完整的 **MiniMindBlock**：

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        
        # Attention 子层
        self.attention = Attention(config)
        # FFN 子层（普通 FFN 或 MoE）
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        # 两个 Pre-Norm（每个子层前各一个）
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, past_key_value=None, use_cache=False):
        # ① 第一个子层：Pre-Norm + Attention + Residual
        attn_out, new_kv = self.attention(
            self.attention_norm(x),      # Pre-Norm
            freqs_cos, freqs_sin,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + attn_out                 # Residual Add

        # ② 第二个子层：Pre-Norm + FFN + Residual
        out = h + self.feed_forward(self.ffn_norm(h))  # Pre-Norm + Residual Add
        
        return out, new_kv
```

**完整数据流（以 minimind 默认配置为例：dim=512, n_heads=8, head_dim=64）**

```
输入 x:  shape [B, T, 512]
    │
    ├── identity (直通残差) ─────────────────────────────────────────────┐
    │                                                                    │
    ▼                                                                    │
attention_norm(x): RMSNorm → shape [B, T, 512]（值变，shape 不变）      │
    │                                                                    │
    ▼                                                                    │
Attention: Q[B,T,8,64], K/V[B,T,4,64] → output [B, T, 512]            │
    │                                                                    │
    ▼                                                                    │
attn_out: shape [B, T, 512]                                             │
    │                                                                    │
    └─ x + attn_out ──────────────────────────────────────────► h [B, T, 512]
                                                                         │
    ┌── identity (直通残差) ──────────────────────────────────────────────┘
    │                                                                    │
    ▼                                                                    │
ffn_norm(h): RMSNorm → shape [B, T, 512]                               │
    │                                                                    │
    ▼                                                                    │
FeedForward: hidden [B, T, 1366] → output [B, T, 512]                  │
    │                                                                    │
    └─ h + ffn_out ──────────────────────────────────────────► out [B, T, 512]
```

> **关键规律**：输入和输出的 shape 永远相同（都是 `[B, T, dim]`），这就是为什么可以简单地堆叠很多层。

---

### 5. 残差连接：为什么如此重要

残差连接（Residual Connection）来自 ResNet（2015 年何恺明等提出），解决了一个核心问题：**为什么深层网络比浅层网络效果还差？**

**直觉解释：学 "修正量" 比从头学更容易**

残差公式是 $h = x + F(x)$，等价于 $F(x) = h - x$。

sublayer 只需要学习"需要对输入做什么修正"，而不是"从头开始映射输入到输出"。就像批改作文，红笔画圈比从头重写容易得多。

**梯度角度：梯度的"高速公路"**

定义 $x_l$ 是第 $l$ 层的输入，$x_L$ 是第 $L$ 层的输入（$L > l$），在 Pre-Norm 中：

$$x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)$$

对 loss 关于 $x_l$ 的梯度：

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} + \frac{\partial L}{\partial x_L} \cdot \frac{\partial \sum F_i}{\partial x_l}$$

$\frac{\partial L}{\partial x_l}$ 中始终包含 $\frac{\partial L}{\partial x_L}$ 这一项——梯度可以**不经任何衰减**地从第 $L$ 层直达第 $l$ 层！

即使所有 sublayer 的梯度都因为激活饱和而趋近于 0，这条"直通路"也保证了梯度不为零。

---

### 6. 完整 MiniMind 模型结构

将 MiniMindBlock 组装成完整模型：

```python
class MiniMind(nn.Module):
    def __init__(self, config=MiniMindConfig()):
        super().__init__()
        self.config = config
        # 词表嵌入层
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        # n_layers 个 MiniMindBlock
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(config.n_layers)
        ])
        # 最终的 RMSNorm
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        # 输出头：映射到词表大小
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight Tying：共享嵌入矩阵和输出矩阵（节省 vocab_size × dim 参数）
        self.tok_embeddings.weight = self.output.weight
```

**Weight Tying 小知识**：输入嵌入矩阵和输出投影矩阵形状相同（都是 `vocab_size × dim`），让它们共享同一组参数，可以节省约 6400×512 ≈ 3.2M 参数，同时有正则化效果（语义相似的 token 既在输入空间相近，也在输出空间相近）。

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| BN vs LN vs RMSNorm | BN 用 batch 统计，LN 用 hidden 统计，RMSNorm 省去均值更快 |
| Pre-Norm | 先 norm 再计算，残差路径"干净"，训练更稳定 |
| 残差连接 | sublayer 只学"修正量"，梯度有直通路不消失 |
| MiniMindBlock | Pre-Norm + Attention + Residual → Pre-Norm + FFN + Residual |
| Weight Tying | 嵌入矩阵 = 输出矩阵，节省参数且有正则效果 |
| 输入输出 shape | 每个 Block 都是 `[B, T, dim]` 进，`[B, T, dim]` 出，可以任意堆叠 |

---

## 🧪 习题集 10

**题目 1（参数量计算）：** minimind 配置：dim=512, n_layers=8, vocab_size=6400。计算 RMSNorm 的参数量（所有层加总）。再计算如果用 LayerNorm 替换，参数量有何变化？

**题目 2（梯度分析）：** 一个 4 层 Post-Norm 网络，在初始化时每层 `F(x)` 的梯度范数为 0.5。从第 4 层到第 1 层，梯度能剩下多少（假设链式法则下梯度相乘）？如果换成 Pre-Norm + 残差连接，情况如何？

**题目 3（shape 追踪）：** 给定 minimind 超参：B=2, T=128, dim=512, n_heads=8, n_kv_heads=4, head_dim=64, hidden_dim=1366。完整追踪一个 token batch 通过 MiniMindBlock 时每个中间 tensor 的 shape（至少列出 8 个步骤）。

**题目 4（代码分析）：** 在 `MiniMind.__init__` 中有 `self.tok_embeddings.weight = self.output.weight` 这一行。如果删掉这行，模型的参数量会增加多少？模型行为会有什么变化（提示：考虑梯度更新方向）？

**题目 5（设计题）：** 有人提议在每个 MiniMindBlock 的**输出**再加一个 RMSNorm（共 3 个 norm：attention_norm, ffn_norm, output_norm），认为这样可以进一步稳定训练。请从梯度流和计算开销两个角度分析这个设计是否合理？

---

> 下一课我们将学习 **L11 - 预训练（Pretraining）——语言建模**，进入训练流程的第一个关键阶段。
