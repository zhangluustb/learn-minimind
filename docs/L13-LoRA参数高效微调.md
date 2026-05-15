# L13 - LoRA 参数高效微调

> **"不改造整栋房子，只在关键的几根支柱上加个支架——同样能改变结构。"**

---

## 📌 本节目标

1. 理解 LoRA（Low-Rank Adaptation）的核心思想和数学原理
2. 掌握低秩分解如何用 1% 的参数达到 90% 的效果
3. 能在 minimind 源码中找到 LoRA 的实现位置
4. 学会如何把 LoRA 权重合并回原模型

---

## 📚 前置知识

- 完成 L12（理解 SFT 的流程）
- 知道什么是"参数效率"

---

## 正文

### 1. LoRA 的动机与直觉

全参数 SFT（Fine-Tuning all parameters）面临一个问题：

- 64M minimind → 所有参数都要调整 → 需要 256MB 的梯度（FP32）+ 256MB 的模型 = 512MB 显存
- 大规模模型（如 70B）→ 几千 GB 显存！

LoRA 的关键观察：**训练过程中的权重更新不需要是满秩的**。

大多数参数变化其实只发生在一个小的子空间里。比如：

$$\Delta W \approx B \cdot A$$

其中 $W$ 是原始权重 `[d_out, d_in]`，$\Delta W$ 是更新量，$A$ 是 `[r, d_in]` 的矩阵（r ≪ d_in），$B$ 是 `[d_out, r]` 的矩阵。

**参数节省：**
- 原始参数：$d_{\text{out}} \times d_{\text{in}}$
- LoRA 参数：$r \times (d_{\text{out}} + d_{\text{in}})$
- 比例：$\frac{r(d_{\text{out}} + d_{\text{in}})}{d_{\text{out}} \times d_{\text{in}}} \approx \frac{r}{\min(d_{\text{out}}, d_{\text{in}})}$

对于 minimind（dim=512），如果 r=64：

$$\text{参数减少} = \frac{64}{512} = 12.5\% \quad (\text{即保留 87.5% 参数不动})$$

### 2. LoRA 的数学形式

对于一个线性层 $y = Wx + b$，LoRA 修改为：

$$y = (W + \Delta W)x + b = (W + BA)x + b$$

其中：
- $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 是预训练权重（冻结，不更新）
- $\Delta W = B \cdot A$ 是低秩更新
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$，$A \in \mathbb{R}^{r \times d_{\text{in}}}$
- $r$ 是秩（rank），通常 8-128

初始化方式很讲究：
- $A$ 用高斯分布初始化 $\mathcal{N}(0, 0.02)$
- $B$ 初始化为全 0，确保最开始 $\Delta W = 0$（不破坏预训练权重）

### 3. minimind 中的 LoRA 实现

minimind 的 `model_lora.py` 定义了一个 LoRA 线性层：

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, lora_rank=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        
        # 原始权重（冻结）
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        
        # LoRA 权重（可训练）
        self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
        
        # 初始化
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)  # 重要：从零开始！
        
        # LoRA 的缩放因子（防止"适配器遗忘"）
        self.lora_alpha = 16
        self.lora_scale = self.lora_alpha / self.lora_rank
    
    def forward(self, x):
        # 原始权重的贡献
        out = F.linear(x, self.weight)
        
        # LoRA 的贡献
        lora_out = x @ self.lora_A.t() @ self.lora_B.t() * self.lora_scale
        
        return out + lora_out

class MiniMindLoRA(MiniMind):
    def __init__(self, config):
        super().__init__(config)
        
        # 冻结所有原始参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 替换 Attention 和 FFN 中的线性层为 LoRA 线性层
        for layer in self.layers:
            # Attention 层的 Q, K, V, O 投影
            layer.attention.q_proj = LoRALinear(
                config.dim, config.n_heads * config.dim // config.n_heads, lora_rank=64
            )
            # ... 同样替换 k_proj, v_proj, o_proj ...
            
            # FFN 的第一和第二个投影
            layer.feed_forward.w1 = LoRALinear(
                config.dim, layer.feed_forward.w2.in_features, lora_rank=64
            )
            # ... 同样替换 w2, w3 ...
```

### 4. train_lora.py 与权重合并

LoRA 训练循环与 SFT 基本相同，区别在于：

```python
# 加载预训练模型
model = MiniMind.from_pretrained('./out/pretrain_768.pth')

# 转换为 LoRA 模型
lora_model = MiniMindLoRA(config)
lora_model.load_state_dict(model.state_dict(), strict=False)

# 只有 LoRA 参数参与梯度计算
for param in lora_model.parameters():
    if param.requires_grad:
        print(param.shape)  # 只打印 LoRA_A 和 LoRA_B 的参数

total_params = sum(p.numel() for p in lora_model.parameters())
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"Trainable: {trainable_params} / {total_params} = {100*trainable_params/total_params:.1f}%")
```

**训练后，如何用模型进行推理？**

两种方案：

#### 方案 A：在线合并（推理时）

```python
def merge_lora_weights(model, lora_model):
    """把 LoRA 的增量合并到原模型"""
    merged_model = copy.deepcopy(model)
    
    for name, param in merged_model.named_parameters():
        if 'attention.q_proj.weight' in name:
            # 找到对应的 LoRA 参数
            lora_a = lora_model.get_parameter(name.replace('q_proj.weight', 'q_proj.lora_A'))
            lora_b = lora_model.get_parameter(name.replace('q_proj.weight', 'q_proj.lora_B'))
            
            # 合并：W_new = W_old + (lora_B @ lora_A) * scale
            lora_update = lora_b @ lora_a * lora_scale
            param.data += lora_update
    
    return merged_model

merged = merge_lora_weights(model, lora_model)
# 现在可以用 merged 进行推理，相当于完整 SFT 的效果
```

#### 方案 B：仅保存 LoRA 部分（存储高效）

```python
# 只保存 LoRA 参数（几 MB）
torch.save(
    {k: v for k, v in lora_model.state_dict().items() if 'lora_' in k},
    './out/lora_weights.pth'
)

# 推理时先加载原模型，再加载 LoRA
model = MiniMind.from_pretrained('./out/pretrain.pth')
lora_weights = torch.load('./out/lora_weights.pth')
model.load_state_dict(lora_weights, strict=False)  # 只加载 LoRA 部分
```

这样优势是：原模型可以被多个 LoRA 适配器共享，每个 LoRA 适配器只需几 MB！

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| LoRA | 用低秩矩阵对 $\Delta W \approx BA$ 参数化，减 90% 参数 |
| 秩 | 通常 8-128，秩越小参数越少，秩越大灵活性越大 |
| 初始化 | $A$ 随机初，$B$ 从零初始化，保持"增量"的含义 |
| 缩放因子 | $\alpha/r$ 防止"适配器遗忘"（太大的 LoRA 初始化） |
| 权重合并 | 推理前把 LoRA 合并回原权重，或只保存 LoRA 部分（高效存储） |
| 应用场景 | 个性化微调、多任务适配、资源受限设备等 |

---

## 🧪 习题集 13

**题目 1（参数对比）：** minimind 的 Attention.q_proj 是 `Linear(512, 512)`。用 LoRA 秩=8 改造后，参数数减少了多少百分比？

**题目 2（权重形状）：** 一个 LoRA 线性层，输入=600，输出=2400，秩=64。计算 $A$ 和 $B$ 的 shape，以及总参数数。

**题目 3（初始化的重要性）：** 为什么 $B$ 要初始化为全 0，而 $A$ 不能？如果反过来会怎样？

**题目 4（内存计算）：** LoRA 微调一个 64M 模型，秩=64，batch_size=16, seq_len=512, 用 BF16。计算激活值、梯度、优化器状态的显存占用（只计 LoRA 部分）。

**题目 5（设计题）：** 现在你有 8 块 GPU，要同时对 8 个领域分别进行 LoRA 微调（医学、法律、代码等）。应该：
- A) 每块 GPU 加载完整模型 + 一个 LoRA 适配器
- B) 共享模型，每块 GPU 只跑一个 LoRA 的前向/反向
- C) 哪个更高效，为什么？

---

> 下一课我们将学习 **L14 - RLHF——DPO 算法原理与实践**，进入强化学习时代。
