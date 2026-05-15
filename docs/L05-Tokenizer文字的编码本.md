# L05 - Tokenizer——文字的"编码本"

> **"语言模型看不见文字，它只看见数字。Tokenizer 就是把文字翻译成数字的那本字典。"**

---

## 📌 本节目标

1. 理解 BPE（字节对编码）分词算法的原理与训练过程
2. 了解 minimind 词表大小为 6400 的设计权衡
3. 掌握特殊 Token（`<s>`、`<|im_start|>` 等）和 `chat_template` 的用法
4. 读懂 `train_tokenizer.py` 的核心逻辑，能自己训练一个 Tokenizer
5. 动手运行编解码示例，真正感受 Token 化的效果

## 📚 前置知识

- 读完 L01~L04，理解 Token 是模型输入的基本单位
- 了解 Python 字典、列表的基本操作
- 知道"词表"是什么（一张 ID ↔ 字符串的映射表）

---

## 正文

### 1. 为什么模型不直接读文字？

你或许会纳闷：模型为什么不直接把"你好"这两个汉字输进去，非得先转成数字？

原因很简单：**神经网络的运算世界里只有矩阵和向量，没有文字**。就像电脑底层只认 0 和 1，所有的图片、音乐、文字都要先编码成二进制，语言模型也需要一套编码规则，把文本变成它能运算的整数序列。

但"用什么粒度切分文字"是个大学问，选择不同，模型的能力和效率会差异巨大：

| 切分策略 | 示例（"学习算法"） | 词表大小 | 优点 | 缺点 |
|---------|----------------|--------|------|------|
| **按字符** | `['学','习','算','法']` | 数千（中文有几万常用字） | 词表小，不会有未知词 | 序列太长，语义太碎 |
| **按单词/词语** | `['学习','算法']` | 数十万 | 语义完整 | 未登录词（OOV）问题严重 |
| **子词（BPE）** | `['学习','算','法']`（视频率而定） | 可控（几千到几十万） | 兼顾两者，泛化性好 | 概念理解有门槛 |

现代 LLM 几乎清一色使用 **子词（Subword）** 方案，而 BPE 是最流行的子词算法之一。minimind 也采用 BPE 训练了一套定制化的 Tokenizer。

---

### 2. BPE 分词原理：从字符出发，合并到词片段

**Byte-Pair Encoding（BPE）** 最初是数据压缩算法，2016 年被引入 NLP。它的核心思想极其朴素：

> **反复合并文本中出现最频繁的相邻字符对，直到词表达到目标大小。**

#### 2.1 训练过程（伪代码）

```python
# BPE 训练算法（伪代码）
def train_bpe(corpus: str, target_vocab_size: int) -> list:
    # 第一步：把语料拆成最小单元（字符），建立初始词表
    vocab = set(all_characters_in_corpus)
    # 把每个词初始表示为字符序列
    # "算法" -> ['算', '法']
    # "hello" -> ['h', 'e', 'l', 'l', 'o']

    while len(vocab) < target_vocab_size:
        # 第二步：统计所有相邻 Token 对的出现频率
        pair_freq = count_adjacent_pairs(corpus)

        # 第三步：找出频率最高的 Token 对
        best_pair = max(pair_freq, key=pair_freq.get)
        # 例如：('算', '法') 出现了 10000 次，是频率最高的对

        # 第四步：把这个对合并成新 Token，加入词表
        new_token = best_pair[0] + best_pair[1]  # "算法"
        vocab.add(new_token)

        # 第五步：把语料中所有该对的出现替换为新 Token
        corpus = merge_pair(corpus, best_pair, new_token)

    return vocab
```

#### 2.2 一个完整的示例

假设训练语料只有下面几行文字（简化示意）：

```
初始语料：
"算法很重要" × 1000
"法律知识" × 200
"算术运算" × 300
```

**第 1 次迭代：**

| Token 对 | 频率 |
|---------|------|
| ('算', '法') | 1000 |
| ('法', '很') | 1000 |
| ('法', '律') | 200 |
| ('算', '术') | 300 |

→ 合并频率最高的 `('算', '法')`，新词表新增 `"算法"`

**第 2 次迭代：**

| Token 对 | 频率 |
|---------|------|
| ('法', '很') | 1000 |
| ('算法', '很') | 1000 |  ← 注意：上一轮合并后，相邻关系更新了
| ... | ... |

→ 持续合并，高频词逐渐"凝固"成完整 Token

#### 2.3 BPE 的数学直觉

每次合并等价于在词表 $V$ 中新增一个元素，合并规则是：

$$
\text{merge}(a, b) \rightarrow ab \quad \text{if } \text{freq}(a, b) = \max_{(x,y)} \text{freq}(x, y)
$$

经过 $k$ 次合并后，词表从初始字符集 $V_0$ 增长到 $|V_0| + k$ 个元素。对于 minimind，初始字符集约有 400 个（常用汉字 + ASCII + 标点），训练约 6000 次合并，最终词表大小达到 6400。

#### 2.4 中英文切分对比

实际观察 minimind Tokenizer 对不同文本的切分：

| 输入文本 | BPE 切分结果（示意） | Token 数 |
|---------|-------------------|--------|
| `"算法"` | `['算法']`（高频词，整体保留） | 1 |
| `"算術"` | `['算', '術']`（繁体字频率低，拆开） | 2 |
| `"Transformer"` | `['Trans', 'former']` 或 `['▁Trans', 'former']` | 2 |
| `"minimind"` | `['mini', 'mind']` | 2 |
| `"2024年"` | `['2024', '年']` | 2 |

> **规律**：越常见的词片段，越有可能整体保留为一个 Token；生僻词、新词会被拆成更小的片段，但不会彻底变成"未知词"。这就是 BPE 对 OOV（Out-of-Vocabulary）问题的根本解法。

---

### 3. minimind Tokenizer：为什么只有 6400 个词？

打开 minimind 项目，你会发现 `model/minimind_tokenizer/` 目录保存着一个只有几十 KB 的 Tokenizer。它的词表大小是 **6400**——相比主流模型，这是一个非常小的数字：

| 模型 | 词表大小 |
|------|--------|
| GPT-2 | 50,257 |
| LLaMA-3 | 128,256 |
| Qwen3 | 151,643 |
| GPT-4 (tiktoken cl100k) | 约 100,000 |
| **minimind** | **6,400** |

#### 3.1 词表大小的三角权衡

词表大小并不是越大越好，它牵涉三个方向的权衡：

```
           词表大小
            ↑
   表达力强  │  存储/显存压力大
   序列更短  │  Embedding 矩阵更大
            │  训练数据稀疏问题加重
            └──────────────────→
         问题：如何平衡？
```

具体而言：

| 维度 | 词表小（6400） | 词表大（100k+） |
|-----|-------------|--------------|
| **Embedding 矩阵大小** | $6400 \times d_{model}$（小） | $100000 \times d_{model}$（大 15 倍） |
| **平均序列长度** | 长（更多 Token 表示同一句话） | 短（包含更完整的词） |
| **生僻词处理** | 拆碎处理，token 多 | 整词保留，token 少 |
| **训练数据需求** | 少即可学好每个 Token | 需海量数据才能覆盖生僻 Token |
| **适合场景** | 教学/小模型/特定领域 | 大规模通用 LLM |

**minimind 为什么选 6400？**

minimind 是一个**教学型小模型**，优先保证：
1. **显存友好**：Embedding 矩阵 $6400 \times 512 = 3.27M$ 个参数，而 LLaMA-3 的 Embedding 就有 $128256 \times 4096 \approx 525M$ 个参数
2. **训练高效**：词表小意味着输出层的 softmax 计算量小，在 pretrain Loss 计算时 cross-entropy 只有 6400 个类别
3. **中文友好**：6400 个词中包含了常用中文字符和高频子词，足够表达日常对话语料

#### 3.2 动手体验 minimind Tokenizer

```python
# 使用 minimind tokenizer 编解码示例
from transformers import AutoTokenizer

# 加载 minimind 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')

# 基本编解码
text = "今天天气怎么样？"
tokens = tokenizer.encode(text)
print("Token IDs:", tokens)         # [1234, 5678, ...]
print("解码结果:", tokenizer.decode(tokens))  # 今天天气怎么样？

# 查看词表大小
print(f"词表大小: {tokenizer.vocab_size}")     # 6400

# 查看 Token 到字符串的对应关系
print(tokenizer.convert_ids_to_tokens(tokens))  # ['今天', '天气', '怎么样', '？']

# 查看某个 ID 对应的 Token
print(tokenizer.convert_ids_to_tokens([0, 1, 2]))  # 特殊 token
```

```python
# 对比不同文本的 Token 数
sentences = [
    "你好",
    "Hello",
    "大语言模型 (Large Language Model) 的核心是 Transformer",
    "1+1=2",
]
for s in sentences:
    ids = tokenizer.encode(s, add_special_tokens=False)
    print(f"'{s}' → {len(ids)} tokens: {ids}")
```

---

### 4. 特殊 Token 与 chat_template

光有普通词是不够的，语言模型还需要一套"标点系统"来区分：这里是一段话的开头/结尾？这是系统提示还是用户输入还是模型输出？——这就是**特殊 Token** 的用途。

#### 4.1 特殊 Token 一览

| Token | ID | 用途 |
|-------|----|------|
| `<s>` | 1 | 序列开始（Begin of Sequence） |
| `</s>` | 2 | 序列结束（End of Sequence） |
| `<unk>` | 0 | 未知词（Unknown Token） |
| `<|im_start|>` | 自定义 | 对话轮次开始标记 |
| `<|im_end|>` | 自定义 | 对话轮次结束标记 |

这些特殊 Token 在词表中占据固定位置，模型通过学习大量对话数据，逐渐理解它们的语义：遇到 `<|im_end|>` 意味着"对方说完了"，遇到 `</s>` 意味着"整个对话结束了"。

#### 4.2 chat_template：对话的格式化模具

**问题**：当你要让模型扮演"AI 助手"角色时，你需要告诉它：谁说的话，谁是系统角色，谁是用户……模型怎么区分这些？

**解法**：`chat_template` 是一套 Jinja2 模板，把结构化的对话消息列表格式化成模型能理解的纯文本字符串。

minimind 采用 **ChatML 格式**（来自 OpenAI，被广泛采用）：

```
<|im_start|>system
你是一个有用的助手。<|im_end|>
<|im_start|>user
什么是大语言模型？<|im_end|>
<|im_start|>assistant
大语言模型是...（模型在这里生成）
```

```python
# chat_template 使用示例
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')

messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user",   "content": "什么是大语言模型？"},
]

# tokenize=False 先看格式化后的纯文本
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted)
# 输出：
# <|im_start|>system
# 你是一个有用的助手。<|im_end|>
# <|im_start|>user
# 什么是大语言模型？<|im_end|>
# <|im_start|>assistant

# tokenize=True 直接得到 Token IDs
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt"
)
print(input_ids.shape)  # torch.Size([1, N])
```

#### 4.3 ChatML 与 minimind 格式的对比

minimind 的对话格式与 ChatML 高度对齐，支持三种角色：

| 角色 | 标记方式 | 作用 |
|------|---------|------|
| `system` | `<\|im_start\|>system\n...<\|im_end\|>` | 设定 AI 的人格、能力边界 |
| `user` | `<\|im_start\|>user\n...<\|im_end\|>` | 人类用户的输入 |
| `assistant` | `<\|im_start\|>assistant\n...<\|im_end\|>` | 模型的回复 |

> **小细节**：在 SFT 训练（L12 详讲）时，模型只对 `assistant` 的内容计算 Loss，`system` 和 `user` 部分被 **Loss Mask** 掩掉，让模型学会"仅生成回复，不生成提问"。

#### 4.4 为什么不直接用换行符分隔？

你可能会想：`system:`、`user:`、换行符不就够了吗？为什么需要 `<|im_start|>` 这样奇怪的 Token？

有三个原因：

1. **唯一性**：`<|im_start|>` 这个 Token 不会出现在正常文本中，不会被误识别；而普通的 `system:` 有可能出现在普通句子里
2. **原子性**：特殊 Token 是一个完整的 Token ID，模型学习时对它有明确的"信号"感知
3. **可扩展性**：未来增加 `tool`、`retrieval` 等新角色只需新增特殊 Token，无需改变整体格式

---

### 5. train_tokenizer.py 源码精读

了解了"用什么"之后，我们来研究 minimind 是"怎么训练"自己的 Tokenizer 的。

#### 5.1 整体流程

```
训练数据（纯文本语料）
        ↓
  SentencePiece 库
        ↓
   BPE 训练算法
        ↓
  输出 tokenizer.model
        ↓
  转换为 HuggingFace 格式
        ↓
  tokenizer.json + special_tokens_map.json
```

#### 5.2 关键代码解析

```python
# train_tokenizer.py 核心逻辑（精简版）
import sentencepiece as spm

def train_tokenizer():
    # 关键参数说明
    spm.SentencePieceTrainer.train(
        # 训练语料路径（一行一句的纯文本文件）
        input='./data/tokenizer_train_data.txt',

        # 输出模型前缀
        model_prefix='minimind_tokenizer',

        # 词表大小：6400（核心参数！决定了最终词表的大小）
        vocab_size=6400,

        # 分词模型类型：BPE（Byte Pair Encoding）
        model_type='bpe',

        # 字符覆盖率：0.9995 意味着覆盖语料中 99.95% 的字符
        # 剩余罕见字符用 <unk> 代替（防止词表被生僻字占满）
        character_coverage=0.9995,

        # 用户自定义特殊 Token
        user_defined_symbols=['<|im_start|>', '<|im_end|>'],

        # 以下控制哪些字符不参与合并（保持原子性）
        split_digits=True,        # 数字单独处理
        byte_fallback=True,       # 遇到未知字符用 UTF-8 字节回退

        # 标准特殊 Token 设置
        bos_id=1, eos_id=2, unk_id=0,
        pad_id=3,
    )
```

#### 5.3 关键参数深度解析

**`vocab_size=6400`**

这是最重要的参数。训练结束后，会执行恰好 `6400 - |初始字符集|` 次合并操作，最终词表包含 6400 个条目（初始字符 + 合并生成的子词 + 特殊 Token）。

**`character_coverage=0.9995`**

如果把所有出现过的字符都加入初始词表，会有大量"一生只见一次"的生僻汉字，浪费词表槽位。`0.9995` 的覆盖率在实践中意味着：常见的约 4000 个汉字全部覆盖，其余罕见字符用 `<unk>` 处理。

**`byte_fallback=True`**

当遇到 `character_coverage` 没有覆盖的字符时，改用 UTF-8 字节级别的 Token 来表示，确保任何文本都能被无损编码（不会产生 `<unk>`）。这是 SentencePiece 的一个重要兜底机制。

**`split_digits=True`**

数字 `0-9` 作为独立字符处理，不参与 BPE 合并，这样 `"2024"` 会被切分为 `['2','0','2','4']` 而不是整体 `['2024']`，让模型对数字大小的泛化能力更好。

#### 5.4 训练数据要求

Tokenizer 的训练数据和模型预训练的数据可以不同，它只需要**代表性文本语料**：

```
数据格式（tokenizer_train_data.txt）：
每行一句话或一段话，纯文本，无需任何标注
```

理想的 Tokenizer 训练数据应该：
- 覆盖目标语言的主要领域（新闻、百科、对话、代码……）
- 中英文混合（如果要支持中英文双语）
- 通常 100MB~1GB 的文本就够了（比模型预训练数据小得多）

> **与模型预训练的区别**：Tokenizer 训练只统计字符共现频率，不涉及神经网络，几分钟内即可完成；而模型预训练要在 GPU 上跑数小时甚至数天。

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| **Tokenizer** | 把文字转换成整数序列（Token IDs）的编码器，是模型的"大门" |
| **BPE** | 反复合并语料中最高频相邻 Token 对，直到达到目标词表大小 |
| **词表大小 6400** | minimind 的轻量化选择：减少 Embedding 开销，适合教学和小模型 |
| **特殊 Token** | `<s>`/`</s>`/`<\|im_start\|>`/`<\|im_end\|>`，标识序列边界和对话角色 |
| **chat_template** | 将结构化对话消息（system/user/assistant）格式化为模型输入字符串 |
| **train_tokenizer.py** | 用 SentencePiece 库驱动 BPE 训练，核心参数：`vocab_size`、`character_coverage` |

---

## 🏋️ 习题集

**基础题：**

1. BPE 的原始数据压缩含义是什么？它被引入 NLP 时解决了什么问题？为什么按字符切分或按完整词切分都不是好方案？

2. 假设 minimind Tokenizer 把句子 `"深度学习是人工智能的核心技术"` 切分成了 6 个 Token。请问，同样的句子用一个词表大小为 100,000 的 Tokenizer（如 GPT-4 的 tiktoken）切分后，得到的 Token 数是更多还是更少？请说明原因。

**进阶题：**

3. 以下是一段使用 `apply_chat_template` 的代码，输出结果是什么格式？请手动还原出完整的输出字符串（包含所有特殊 Token 和换行符）：

    ```python
    messages = [
        {"role": "system", "content": "你是一位数学老师。"},
        {"role": "user",   "content": "1+1等于几？"},
        {"role": "assistant", "content": "等于2。"},
        {"role": "user",   "content": "那2+2呢？"},
    ]
    result = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ```

4. minimind 的 Embedding 矩阵形状为 $[6400, 512]$（词表大小 × 隐层维度）。如果把词表扩大到 32000（LLaMA 的规模），Embedding 矩阵的参数量会增加多少？这部分参数占 minimind-3 总参数量（64M）的比例会变化多少？（提示：$64M = 64 \times 10^6$）

**设计题：**

5. 假设你要为一家医院的 AI 问诊系统训练一个专用 Tokenizer，目标语料以简体中文医学术语为主（如"心肌梗死"、"磁共振成像"、"糖尿病酮症酸中毒"等），同时需要支持一定量的英文医学缩写（CT、MRI、ICU……）。请从以下三个维度给出设计方案：
    - **词表大小**选多大合适？给出理由。
    - **character_coverage** 参数如何设置？
    - 需要哪些**用户自定义特殊 Token**（`user_defined_symbols`）？

---

> 下一课：**L06 - Embedding 与旋转位置编码（RoPE）**
