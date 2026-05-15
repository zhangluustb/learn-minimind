# L04 - minimind 项目导览与环境搭建

> **"先看懂地图，再出发探险——磨刀不误砍柴工。"**

---

## 📌 本节目标

1. 掌握 minimind 项目的完整目录结构，知道每个文件是干什么的
2. 理解 `MiniMindConfig` 的关键参数及其含义
3. 完成环境搭建并成功运行第一次推理
4. 了解 minimind 的模型权重命名规范

## 📚 前置知识

- 完成 L01-L03
- 会使用命令行基本操作（`cd`、`pip install`、`python xxx.py`）
- 有 Python 环境（推荐 Python 3.10+）

---

## 正文

### 1. 项目目录结构全解

minimind 的目录组织得非常清晰，每个文件夹承担单一职责。让我们像解剖一辆汽车一样，逐层理解它的结构：

```
minimind/
├── model/                      ← 🧠 模型"大脑"（核心架构）
│   ├── model_minimind.py       # 完整 LLM 架构：Attention + FFN + Block + Model
│   ├── model_lora.py           # LoRA 低秩适配实现
│   └── LMConfig.py             # （部分版本）模型配置类
│
├── trainer/                    ← 🏋️ 训练"健身房"（各阶段训练脚本）
│   ├── train_pretrain.py       # 预训练：从文本学语言规律
│   ├── train_full_sft.py       # 全参数监督微调：学会按指令对话
│   ├── train_lora.py           # LoRA 微调：低成本垂域适配
│   ├── train_dpo.py            # RLHF-DPO：人类偏好对齐
│   ├── train_ppo.py            # RLAIF-PPO：AI 反馈强化学习
│   ├── train_grpo.py           # RLAIF-GRPO（includes CISPO）：组相对策略优化
│   ├── train_distillation.py   # 知识蒸馏：从大模型学习
│   └── train_agent.py          # Agentic RL：工具调用强化学习
│
├── dataset/                    ← 📦 数据"原材料库"
│   ├── lm_dataset.py           # Dataset 类实现
│   └── *.jsonl                 # 训练数据文件（pretrain/sft/dpo...）
│
├── scripts/                    ← 🔧 工具"瑞士军刀"（推理/部署/转换）
│   ├── serve_openai_api.py     # OpenAI 兼容 API 服务
│   ├── web_demo.py             # Streamlit 聊天 WebUI
│   ├── convert_model.py        # torch ↔ transformers 格式转换
│   └── chat_api.py             # API 客户端测试
│
├── out/                        ← 💾 模型权重输出目录（训练完成后生成）
│   ├── pretrain_768.pth        # 预训练权重（768=hidden_size）
│   ├── full_sft_768.pth        # SFT 权重
│   └── ...
│
├── eval_llm.py                 # 推理/测试脚本（最常用）
├── eval_toolcall.py            # Tool Use 功能测试
├── train_tokenizer.py          # Tokenizer 训练脚本（一般不重新训练）
├── requirements.txt            # 依赖列表
└── README.md                   # 项目说明文档
```

**类比理解各模块：**

- `model/model_minimind.py` = 汽车的"发动机设计图纸"，定义了模型的每一个零件
- `trainer/` = 汽车"测试跑道"的集合，每种跑道（训练方式）对应一个脚本
- `dataset/` = 汽车"燃油仓库"，存放所有训练数据
- `scripts/` = 汽车"服务站工具箱"，用于部署、转换、对外提供服务
- `out/` = 训练完成后存放"出厂成品"的仓库

---

### 2. model_minimind.py 深度导览

这是整个项目最核心的文件，定义了模型的完整结构。让我们从顶层到底层，理解它的层次架构：

```python
# 层次结构（从外到内）
MiniMindForCausalLM           ← 最外层：包含语言模型头（输出层）
  └── MiniMindModel            ← 核心模型：Embedding + N个Block + 最终归一化
        ├── Embedding           ← 词嵌入层：Token ID → 向量
        ├── MiniMindBlock × N   ← N 个 Transformer Block（重复堆叠）
        │     ├── RMSNorm       ← 归一化（Pre-Norm 风格）
        │     ├── Attention     ← 多头注意力（含 GQA + RoPE）
        │     ├── RMSNorm       ← 第二层归一化
        │     └── FeedForward   ← 前馈网络（SwiGLU 激活）或 MOEFeedForward
        └── RMSNorm             ← 最终归一化
  └── lm_head                   ← 语言模型头：向量 → Token 概率
```

**MiniMindConfig：模型的"设计参数表"**

```python
@dataclass
class MiniMindConfig:
    # 基础架构参数
    hidden_size: int = 768          # 每个 Token 的向量维度（"词的厚度"）
    num_hidden_layers: int = 8      # Transformer Block 的堆叠层数
    num_heads: int = 8              # 注意力 Q 头数
    num_kv_heads: int = 4           # 注意力 KV 头数（GQA）
    
    # 词表和序列长度
    vocab_size: int = 6400          # 词表大小（minimind 自定义 Tokenizer）
    max_position_embeddings: int = 32768  # 最大上下文长度
    
    # FFN 参数
    intermediate_size: int = None   # FFN 中间层维度（默认 hidden_size * 8/3）
    
    # MoE 参数（可选）
    num_experts: int = 4            # 专家数量（Dense 模型时为 1）
    num_experts_per_tok: int = 1    # 每个 Token 激活的专家数（top-k routing）
    
    # 位置编码参数
    rope_theta: float = 1e6         # RoPE 的 θ 参数（影响位置编码的频率）
    
    # 归一化参数
    rms_norm_eps: float = 1e-5      # RMSNorm 的数值稳定性 ε
    
    # Dropout
    dropout: float = 0.0            # 注意力 Dropout（训练时）
```

**参数规模对照表：**

| 配置名 | hidden_size | num_layers | 参数量 | 适用场景 |
|--------|-------------|------------|--------|---------|
| minimind-3 | 768 | 8 | **64M** | 主线学习版本 |
| minimind-3-moe | 768 | 8 | 198M-A64M | MoE 版本 |
| minimind2-small | 512 | 8 | 26M | 快速实验 |
| 自定义 | 256 | 4 | ~8M | 入门调试 |

> **💡 记住"黄金配置"：** `hidden_size=768, num_layers=8` 就是 minimind-3 的主线配置，整个教程默认使用这个。

---

### 3. 环境搭建：从零开始

#### Step 0：克隆项目

```bash
# 克隆仓库（--depth 1 只获取最新版本，加速下载）
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
```

#### Step 1：安装依赖

```bash
# 推荐使用国内镜像加速下载
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# 验证 PyTorch 安装（应该输出 True）
python -c "import torch; print(torch.cuda.is_available())"
```

**requirements.txt 关键依赖解析：**

```
torch>=2.1.0            # PyTorch 核心（计算引擎）
transformers>=4.40.0    # HuggingFace 格式支持（模型保存/加载）
tokenizers>=0.15.0      # 高效 Tokenizer（BPE 实现）
datasets                # 数据集工具
numpy                   # 数值计算
tqdm                    # 进度条
wandb                   # 训练监控（可选）
streamlit               # Web UI（可选，用于web_demo.py）
```

#### Step 2：下载预训练模型（可选，用于快速推理）

```bash
# 方式1：通过 ModelScope（推荐国内用户）
pip install modelscope
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3

# 方式2：通过 HuggingFace
git clone https://huggingface.co/jingyaogong/minimind-3
```

下载后的文件结构：

```
minimind-3/
├── config.json                 # 模型配置（JSON 格式）
├── generation_config.json      # 生成配置（温度、top_p 等）
├── model.safetensors           # 模型权重（safetensors 格式）
├── tokenizer.json              # Tokenizer（BPE 规则）
├── tokenizer_config.json       # Tokenizer 配置
└── special_tokens_map.json     # 特殊 Token 映射
```

#### Step 3：第一次运行

```bash
# 使用 transformers 格式的下载模型推理
python eval_llm.py --load_from ./minimind-3

# 使用本地 PyTorch 权重推理（需要先训练或下载 .pth 文件）
python eval_llm.py --load_from ./model --weight full_sft
```

---

### 4. eval_llm.py 推理流程详解

下面是第一次运行时内部发生的事情（伪代码注释版）：

```python
# eval_llm.py 核心流程

# Step 1: 加载 Tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./minimind-3')
# Tokenizer 会把"你好世界" → [101, 2345, 567, ...]

# Step 2: 加载模型
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
# 方式A：加载 transformers 格式
model = AutoModelForCausalLM.from_pretrained('./minimind-3')
# 方式B：加载 PyTorch 格式
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, ...)
model = MiniMindForCausalLM(config)
model.load_state_dict(torch.load('./out/full_sft_768.pth'))

model.eval()           # 推理模式（关闭 Dropout）
model.to('cuda:0')     # 移到 GPU

# Step 3: 构建对话输入（使用 chat_template）
messages = [
    {"role": "user", "content": "你好，请介绍一下自己"}
]
# chat_template 会自动加上 <|im_start|>...<|im_end|> 格式
input_ids = tokenizer.apply_chat_template(
    messages, 
    return_tensors='pt',
    add_generation_prompt=True  # 在末尾加上 <|im_start|>assistant\n
).to('cuda:0')

# Step 4: 自回归生成
# 每次生成一个 Token，再把它加入输入，再生成下一个...
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=512,      # 最多生成 512 个 Token
        temperature=0.7,         # 温度（越高越随机）
        top_p=0.9,               # 核采样（保留累计概率前 90% 的词）
        do_sample=True,          # 是否随机采样（False=贪心解码）
    )

# Step 5: 解码输出
response = tokenizer.decode(
    output_ids[0][input_ids.shape[1]:],  # 只取新生成的部分
    skip_special_tokens=True
)
print(f"🤖: {response}")
```

**一次完整对话的数据流：**

```
用户输入文本 "你好"
         ↓
Tokenizer 编码：["你", "好"] → [254, 189]
         ↓
chat_template 包装：
  "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
  → [1, 254, 189, 2, 3, 1, ...]  (Token IDs)
         ↓
模型自回归生成：
  step 1: 预测第一个输出 Token → "你"
  step 2: 预测第二个输出 Token → "好"
  step 3: 预测第三个输出 Token → "！"
  ...直到生成 <|im_end|> 或达到 max_new_tokens
         ↓
Tokenizer 解码：[254, 189, 15, ...] → "你好！很高兴认识你。"
```

---

### 5. 模型权重命名规范

minimind 的权重文件命名有清晰规律，理解它能帮你快速找对文件：

```
pretrain_768.pth        = 预训练权重，hidden_size=768
full_sft_768.pth        = 全参数 SFT 权重，hidden_size=768
lora_medical_768.pth    = LoRA 医疗微调权重，hidden_size=768
dpo_768.pth             = DPO 权重
ppo_actor_768.pth       = PPO 训练後的 Actor 权重
grpo_768.pth            = GRPO 权重
agent_768.pth           = Agentic RL 权重
```

**`eval_llm.py` 的 `--weight` 参数示例：**

```bash
python eval_llm.py --weight pretrain      # 测试预训练阶段（只会接龙，不会聊天）
python eval_llm.py --weight full_sft      # 测试 SFT 阶段（会正常对话）
python eval_llm.py --weight full_sft --lora_weight lora_medical  # 加载 LoRA
```

---

### 6. 常见问题速查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `CUDA out of memory` | 显存不足 | 减小 `batch_size` 或 `max_seq_len` |
| `ModuleNotFoundError: model` | 未在项目根目录运行 | `cd minimind/` 后再执行 |
| 训练 loss 不下降 | 学习率过大/过小 | 参考 README 的推荐设置 |
| 生成乱码 | 权重与 Tokenizer 不匹配 | 确保使用配套的 tokenizer.json |
| DDP 报错 `connection refused` | 端口被占用 | 加参数 `--master_port=29501` |

---

## 💡 本节小结

| 组件 | 位置 | 功能 |
|------|------|------|
| `model/model_minimind.py` | 模型定义 | LLM 完整架构（Attention + FFN + Block） |
| `trainer/train_*.py` | 各阶段训练 | pretrain/sft/lora/dpo/grpo/agent... |
| `dataset/lm_dataset.py` | 数据加载 | 按需读取 JSONL 格式数据 |
| `scripts/` | 工具脚本 | API 服务、WebUI、格式转换 |
| `eval_llm.py` | 推理测试 | 命令行对话接口 |
| MiniMindConfig | 配置类 | hidden_size=768, layers=8, vocab=6400 |

---

## 🏋️ 习题集

**基础题：**

1. minimind 中 `n_layers=8`，`hidden_size=768`，`vocab_size=6400`。不查代码，估算模型的大致参数量（提示：最大头部是 Embedding 层 `vocab_size × hidden_size` 和各个线性层）。
2. `out/pretrain_768.pth` 文件名中的 `768` 代表什么？如果你训练一个更小的 `hidden_size=256` 的模型，权重文件名会是什么？
3. 运行 `python eval_llm.py --weight pretrain` 和 `--weight full_sft` 分别会有什么不同表现？（提示：预训练模型只会接龙，SFT 模型才能理解指令）

**进阶题：**

1. 阅读 `model/model_minimind.py` 中的 `MiniMindForCausalLM.forward()` 方法，找到 loss 计算的代码。它使用了什么损失函数？如何处理 padding token（不参与 loss 计算的 token）？
2. minimind 的 `MoE` 版本（num_experts=4）与 Dense 版本的区别在哪里？在代码中找到 `MOEFeedForward` 类，解释其与普通 `FeedForward` 的差异。
3. 尝试修改 `MiniMindConfig` 创建一个更小的模型（`hidden_size=256, num_hidden_layers=4`），计算它的参数量，并估算在单张 3090 上训练一个 epoch 大约需要多少时间。

---

> 下一课：**L05 - Tokenizer——文字的"编码本"**
