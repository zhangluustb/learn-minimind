# L09 - KV-Cache 与推理加速

> **"如果你已经背会了课文，每次背诵都从头再来，那才是真正的低效。"**

---

## 📌 本节目标

1. 理解自回归解码的工作原理及其天然的计算瓶颈
2. 掌握 KV-Cache 的核心机制——用空间换时间
3. 能在 minimind 源码中找到 `past_key_value` 的完整流转路径
4. 了解影响推理速度的多种因素（量化、batch、投机解码等）

---

## 📚 前置知识

- 完成 L07（理解 Multi-Head Attention 的 Q/K/V 计算）
- 了解 K/V 的维度：`[batch, seq_len, n_kv_heads, head_dim]`

---

## 正文

### 1. 训练 vs 推理：两种截然不同的运行模式

初学者容易忽略一个关键点：**训练和推理的数据流完全不同**。

**训练阶段（Teacher Forcing）**

你把完整的输入序列 `[token_1, token_2, ..., token_T]` 一次性喂给模型，模型并行输出所有位置的预测，每个位置的 loss 都可以同时计算并反向传播。

```python
# 训练：一次前向，所有位置并行
outputs = model(input_ids)          # [B, T, vocab_size]
logits = outputs.logits             # 所有 T 个位置的预测
loss = cross_entropy(logits, labels)  # T 个 loss 一起算
```

**推理阶段（自回归解码）**

推理时，你只有一个起始 prompt，模型一次只输出**一个** token，然后把这个 token 拼上去，再预测下一个……如此循环：

```
step 1: 输入 [你好] → 预测 "，"
step 2: 输入 [你好，] → 预测 "世"
step 3: 输入 [你好，世] → 预测 "界"
...
```

> **类比**：训练就像在考场上做选择题，答案全在面前，可以一次全写完。推理就像在玩"接龙"游戏，每次只能说一个字，还得等对方接着往下说。

---

### 2. 自回归的计算瓶颈

问题来了：在第 $t$ 步生成 token 时，Attention 需要计算：

$$\text{Attention}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}$$

也就是新 token 的 $Q_t$ 要跟**所有历史** token 的 K、V 做点积。如果不做任何优化：

- 第 1 步：计算 K₁, V₁
- 第 2 步：重新计算 K₁, V₁, K₂, V₂
- 第 3 步：重新计算 K₁, V₁, K₂, V₂, K₃, V₃
- ……

总计算量：$O(T^2 \cdot d)$，T 是生成序列长度，d 是 hidden size。

生成 1024 个 token？每个 attention layer 都要做约 $1024^2 \approx 100$ 万次点积运算——而且大量计算是**完全重复的**！

---

### 3. KV-Cache：用空间换时间

核心洞察：**历史 token 的 K、V 完全没有变化！**

因为 K 和 V 只取决于 token 本身的 embedding（在自回归解码中，已经生成的 token 不会再改变），所以可以把它们**缓存**起来：

```
step 1: 计算 K₁, V₁，存入 cache
step 2: 只计算 K₂, V₂；从 cache 取 K₁, V₁，拼接后使用
step 3: 只计算 K₃, V₃；从 cache 取 K₁, V₁, K₂, V₂，拼接后使用
```

有了 KV-Cache，第 $t$ 步的计算量从 $O(t \cdot d)$ 降为 $O(d)$（只需要计算新 token 的 Q、K、V，三个矩阵乘法）。总计算量从 $O(T^2 d)$ 降为 $O(T d)$——线性复杂度！

**KV-Cache 的内存代价**

天下没有免费的午餐，KV-Cache 需要额外的显存：

$$\text{KV Cache 大小} = B \times L \times n_{kv} \times S \times d_h \times 2 \times \text{bytes}$$

其中：
- $B$ = batch_size
- $L$ = n_layers（层数）
- $n_{kv}$ = n_kv_heads（KV 头数）
- $S$ = max_seq_len（最大序列长度）
- $d_h$ = head_dim（每个头的维度）
- $\times 2$：K 和 V 各一份
- bytes：FP16 = 2，FP32 = 4

以 minimind 为例（4层, 4KV头, 1024 max_seq_len, head_dim=64, FP16）：

$$4 \times 4 \times 4 \times 1024 \times 64 \times 2 \times 2 \approx 4\text{ MB}$$

非常小！大模型（如 LLaMA-70B，80层，8KV头，4096 seq）的 KV Cache 可以达到 160GB，这正是 vLLM 等推理框架重点优化的地方。

---

### 4. `past_key_value` 在 minimind 源码中的流转

minimind 的 Attention 模块通过 `past_key_value` 参数传递缓存，整个流程如下：

```python
class Attention(nn.Module):
    def forward(self, x, freqs_cos, freqs_sin, past_key_value=None, use_cache=False):
        bsz, seq_len, _ = x.shape

        # 1. 计算当前输入的 Q, K, V
        xq = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 2. 施加 RoPE（只对新 token 的位置应用对应角度）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 3. KV-Cache：将历史缓存拼接到当前 KV 前面
        if past_key_value is not None:
            past_key, past_value = past_key_value
            xk = torch.cat([past_key, xk], dim=1)   # [B, past_len+1, kv_heads, head_dim]
            xv = torch.cat([past_value, xv], dim=1)

        # 4. 决定是否保存更新后的 KV
        new_kv = (xk, xv) if use_cache else None

        # 5. GQA：把 KV 头数扩展到与 Q 头数相同
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 6. Flash Attention（prefill 阶段需要 causal mask，decode 阶段不需要）
        output = F.scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2),
            is_causal=(past_key_value is None)
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(output), new_kv
```

> **注意**：decode 阶段不需要 causal mask！因为只有一个新 token 的 Q，它只能"看到"历史，历史缓存中所有 K/V 都在它"之前"，天然就是 causal 的。

**推理生成循环**

```python
@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.85, top_p=0.85):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    past_key_values = None  # 初始无缓存

    for _ in range(max_new_tokens):
        # decode 阶段：只传最新的 1 个 token（如果 past_kv 存在）
        cur_input = input_ids[:, -1:] if past_key_values is not None else input_ids

        outputs = model(cur_input, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits          # [1, 1, vocab_size]
        past_key_values = outputs.past_key_values  # 更新 cache（现在包含了新 token 的 KV）

        # 采样：temperature 控制随机性，top-p 进行核采样
        next_token = sample(logits[:, -1, :], temperature=temperature, top_p=top_p)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
```

生成过程分为两个阶段：

| 阶段 | 英文 | 工作内容 | 耗时特征 |
|------|------|---------|---------|
| **预填充** | Prefill | 处理完整 prompt，建立 KV Cache | 一次，受 prompt 长度影响 |
| **解码** | Decode | 逐步生成每个新 token | 每步相同，轻量 |

---

### 5. 推理速度的其他影响因素

KV-Cache 解决了重复计算的问题，但还有更多优化手段：

**量化（Quantization）**

把模型权重和 KV Cache 从 FP32/FP16 转为 INT8 或 INT4，减少内存带宽需求：

| 精度 | 每个参数大小 | 64M 模型显存 | 速度提升 |
|------|------------|-------------|---------|
| FP32 | 4 bytes | ~256 MB | 基准 |
| FP16 / BF16 | 2 bytes | ~128 MB | ~1.5-2x |
| INT8 | 1 byte | ~64 MB | ~2-3x |
| INT4 | 0.5 bytes | ~32 MB | ~3-5x |

**批量推理（Batching）**

吞吐量 ≠ 单请求延迟。把多个请求合并成一个 batch，GPU 利用率可从 30% 提升到 90%+。

**采样参数**

```python
# temperature：控制生成的"随机性"
# temperature=0 → 每次选概率最高的（贪婪解码）
# temperature=1 → 按原始概率采样
# temperature>1 → 更随机，更有创意但可能不连贯

# top-p（nucleus sampling）：累计概率阈值
# top_p=0.9 → 只从累计概率达到 90% 的 token 中采样
```

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| 自回归解码 | 逐步生成，每步预测一个 token，前面的不变 |
| KV-Cache | 缓存历史 K/V，避免重复计算，复杂度从 $O(T^2)$ 降为 $O(T)$ |
| Prefill 阶段 | 处理完整 prompt，建立 KV Cache，需要 causal mask |
| Decode 阶段 | 每步只输入 1 个 token，不需要 causal mask |
| past_key_value | 每层 Attention 的 (K, V) 缓存，逐步追加新 token 的 KV |
| 量化 | 减少内存占用和带宽，是推理加速的另一大武器 |

---

## 🧪 习题集 9

**题目 1（计算题）：** 给定一个模型：4 层、8 个 KV 头、head_dim=128、max_seq_len=2048、batch_size=4、FP16 精度。计算完整 KV Cache 占用的显存大小（MB）。

**题目 2（过程追踪）：** 假设模型已经生成了 3 个 token，当前 `past_key_value` 中每层 K 的 shape 是什么？再生成第 4 个 token 后，shape 变为多少？（假设 n_kv_heads=4, head_dim=64）

**题目 3（代码理解）：** 在 decode 阶段为什么 `is_causal=False`？如果错写成 `True` 会发生什么？

**题目 4（分析题）：** 温度参数 temperature=0 时，模型每次生成的内容是否完全确定？temperature→∞ 时会发生什么？

**题目 5（设计题）：** vLLM 的 Paged Attention 把 KV Cache 分成固定大小的"页"进行管理，类似操作系统的内存分页。请分析为什么这种设计对多用户并发推理服务特别有价值？（提示：考虑不同请求的序列长度差异）

---

> 下一课我们将学习 **L10 - RMSNorm 与完整 MiniMindBlock**，把 Attention 和 FFN 组装成完整的 Transformer 层。
