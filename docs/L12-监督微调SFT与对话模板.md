# L12 - 监督微调（SFT）与对话模板

> **"预训练的模型是个万事通，但听不懂你到底想要什么。SFT 就是教它读懂你的指令。"**

---

## 📌 本节目标

1. 理解 SFT（Supervised Fine-Tuning）与 Instruction Tuning 的概念
2. 掌握 chat_template 的作用及其格式规范
3. 理解 loss masking 为什么只计算答案部分的损失
4. 能对比 SFT 前后模型的行为差异

---

## 📚 前置知识

- 完成 L11（预训练的基本流程）
- 理解 chat_template 的概念（L05）

---

## 正文

### 1. 从语言建模到指令跟随

预训练完成后，模型已经学会了语言的统计规律——它能流畅地"接龙"。但问题来了：

```
Q: 什么是大语言模型？
模型回复：是指由大量数据训练的语言模型，如 GPT-3, GPT-4, ChatGPT...
你想要的：听起来不像在背百科
```

模型能说，但不知道什么时候该停。或者说出冗长的列举而不是精准的回答。

**监督微调（SFT）** 的用意就是：通过有标注的"指令-回答"对，教模型理解什么是好的、有帮助的回答。

$$L_{\text{SFT}} = -\frac{1}{|y|} \sum_{i \in \text{answer}} \log P(y_i | x, y_{<i})$$

注意：这里 loss 只计算**答案部分** $y$，不计算指令部分 $x$。

### 2. 指令数据的格式与构建

硅谷的大公司（OpenAI、Anthropic）会雇数千人标注"高质量"的指令数据。minimind 用的数据通常更简单：

```json
{
  "instruction": "什么是大语言模型？",
  "input": "",
  "output": "大语言模型（LLM）是由大量文本数据训练的神经网络模型。\n它能理解和生成自然语言，完成翻译、摘要、问答等任务。"
}
```

**或者用 chat 格式：**

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是大语言模型？"},
    {"role": "assistant", "content": "大语言模型（LLM）是..."}
  ]
}
```

### 3. chat_template 的秘密

你可能注意到，模型输入是纯 token，怎么知道哪些是指令哪些是回答呢？

就靠 **chat_template** 这个模板字符串！它定义了**哪部分计算损失、哪部分用于掩码**：

```python
chat_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""
```

当数据被喂入模型时，会应用这个模板：

```python
def apply_chat_template(messages):
    text = "<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n"
    text += "<|im_start|>user\n什么是大语言模型？<|im_end|>\n"
    text += "<|im_start|>assistant\n大语言模型（LLM）是...<|im_end|>"
    return text
```

然后编码成 tokens：`[1234, 5678, 9999, ...]`

**Loss Masking 的核心逻辑：**

```python
# 标记出 assistant 部分的 token 位置（损失相关的部分）
attention_mask = [0, 0, 0, ..., 0,      # system 部分：mask 掉，不参与 loss
                  0, 0, 0, ..., 0,      # user 部分：mask 掉
                  1, 1, 1, ..., 1]      # assistant 部分：参与 loss

# 计算 loss 时
loss = cross_entropy(logits, labels)  # 原生的 loss
loss = (loss * attention_mask).sum() / attention_mask.sum()  # 只取 assistant 部分的平均
```

minimind 的 `train_full_sft.py` 就是这样做的：

```python
def preprocess_sft_data(messages, tokenizer):
    # 应用 chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer.encode(text)
    
    # 计算 assistant 部分的起始位置
    template_text = tokenizer.apply_chat_template([
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""}
    ], tokenize=False)
    
    # 在完整文本中找到 assistant tag 的位置
    im_start_pos = text.rfind("<|im_start|>assistant\n")
    assistant_start = len(tokenizer.encode(text[:im_start_pos]))
    
    # 创建 loss_mask
    loss_mask = [0] * len(tokens)
    loss_mask[assistant_start:] = [1] * (len(tokens) - assistant_start)
    
    return {
        'input_ids': tokens,
        'labels': tokens,  # labels 和 input_ids 相同
        'loss_mask': loss_mask
    }

def train_step(batch, model, optimizer):
    outputs = model(batch['input_ids'])
    logits = outputs.logits
    
    # 计算原生的交叉熵
    loss = cross_entropy(
        logits.view(-1, vocab_size),
        batch['labels'].view(-1),
        reduction='none'
    ).view(batch['input_ids'].shape)
    
    # 应用 mask：只计算 assistant 部分的 loss
    masked_loss = (loss * batch['loss_mask']).sum() / batch['loss_mask'].sum()
    
    masked_loss.backward()
    optimizer.step()
```

### 4. SFT 的常见陷阱

**陷阱 1：过拟合**

SFT 数据相比预训练数据少得多（通常几千到几万对），模型容易记住答案而非真正理解。表现就是：

- 在 SFT 数据上 Acc 很高
- 但对稍微变化的问题给不出好回答

解决方案：混合预训练数据和 SFT 数据（比例 9:1），保持模型的泛化能力。

**陷阱 2：忘记预训练知识**

如果只用 SFT 数据训练，模型会"遗忘"预训练阶段学到的很多知识。

```
模型回复："我不知道如何计算 7+3"（但预训练中肯定见过）
```

解决方案：同样是混合数据，以及降低 SFT 学习率（通常是预训练的 1/10）。

**陷阱 3：Loss Masking 错误**

如果不小心把指令部分也算进 loss，模型会被迫记住每个问题的标准表述，而不是理解语义。

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| SFT | 用有标注的指令数据，教模型从"接龙能手"变成"指令跟随者" |
| chat_template | 定义系统/用户/助手各部分的分隔符，便于格式化和 loss masking |
| Loss Masking | 只在 assistant（答案）部分计算损失，指令部分跳过 |
| 数据混合 | 预训练 + SFT 数据混合训练，保持泛化能力，避免过拟合和遗忘 |
| 学习率 | SFT 学习率通常是预训练的 1/10，防止太激进的参数修改 |
| 特殊 Token | `<|im_start|>` 和 `<|im_end|>` 是角色分隔符，tokenizer 预留的特殊 token |

---

## 🧪 习题集 12

**题目 1（模板应用）：** 给定消息：
```json
[
  {"role": "user", "content": "1+1=？"},
  {"role": "assistant", "content": "2"}
]
```
用 minimind 的 chat_template 格式化后，哪部分需要 mask（loss_mask=0），哪部分需要计算损失（loss_mask=1）？写出完整的 token 及对应的 mask。

**题目 2（过拟合分析）：** SFT 数据只有 5000 对对话，模型有 64M 参数。为什么容易过拟合？计算一下"参数数/样本数"的比例。

**题目 3（学习率策略）：** 预训练使用 1e-4 的学习率，SFT 应该改成多少？为什么不能沿用更大的学习率（比如 1e-3）？

**题目 4（数据混合）：** 如果 SFT 训练集只有 1 万对对话，为了避免遗忘预训练知识，应该：
- A) 只用 SFT 数据，多跑几个 epoch
- B) 混合 9 万条预训练数据 + 1 万条 SFT 数据各一遍
- C) 70% 预训练 + 30% SFT，轮流交替

你选哪个，为什么？

**题目 5（设计题）：** 如果要给 minimind 加入"拒绝有害内容"的能力，应该怎么设计 SFT 数据和训练流程？

---

> 下一课我们将学习 **L13 - LoRA 参数高效微调**，用低秩分解把完整 SFT 的显存/参数需求降到 1/10。
