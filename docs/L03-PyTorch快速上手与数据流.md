# L03 - PyTorch 快速上手与数据流

> **"工欲善其事，必先利其器——训练 LLM，先摸透 PyTorch 的流水线。"**

---

## 📌 本节目标

1. 掌握 PyTorch Tensor 的核心属性：shape、dtype、device
2. 理解 Dataset 与 DataLoader 的"流水线工厂"设计
3. 理解混合精度训练（bfloat16 / autocast）如何节省显存
4. 掌握梯度累积（accumulation_steps）技巧
5. 了解 DDP 多卡训练的基本思想
6. 通读 minimind `train_pretrain.py` 的训练循环骨架

## 📚 前置知识

- Python 基础（列表、函数、类）
- 了解什么是神经网络的前向传播和反向传播（概念即可）
- 完成 L01、L02

---

## 正文

### 1. Tensor：一切的基础

在 PyTorch 里，数据以 **Tensor（张量）** 的形式存在。Tensor 就是多维数组——它比 Python list 快得多，因为它可以运行在 GPU 上，并且支持自动求导。

用生活类比：如果 Python list 是手工记账本，那么 Tensor 就是 Excel 表格的 GPU 加速版——能存大量数字，并且自动帮你做所有数学运算。

#### Tensor 三要素

| 属性 | 含义 | 示例 |
|------|------|------|
| `shape` | 维度形状 | `(32, 512)` → 批大小 32，序列长度 512 |
| `dtype` | 数据类型 | `float32`（普通精度）、`bfloat16`（半精度） |
| `device` | 所在设备 | `cpu` 或 `cuda:0`（第 0 块 GPU） |

```python
import torch

# 创建一个形如 [batch=2, seq_len=5] 的 Token ID 张量
token_ids = torch.tensor([[1, 23, 456, 789, 2], 
                           [1, 42, 33, 11, 2]])
print(token_ids.shape)   # torch.Size([2, 5])
print(token_ids.dtype)   # torch.int64（整数 ID）

# 创建随机的浮点 Tensor，并移到 GPU
x = torch.randn(2, 5, 768)    # [batch, seq, hidden_size]
x = x.to('cuda')              # 移到 GPU
print(x.device)               # cuda:0
print(x.dtype)                # torch.float32

# 维度操作
x_T = x.transpose(1, 2)  # 交换 dim=1 和 dim=2: [2, 768, 5]
x_flat = x.view(2, -1)   # 展平后两维: [2, 3840]
```

**关键直觉：** 不同 `device` 的 Tensor 不能直接运算（就像不能把 Word 文档直接和 Excel 文件相加）。训练时需要确保模型参数和数据在同一设备上。

---

### 2. Dataset 与 DataLoader：流水线工厂

训练一个 LLM 需要处理数十亿 Token 的文本数据。如果每次训练前把所有数据加载进内存，内存会直接爆掉。

解决方案是**生产线模式（Pipeline）**：

- **Dataset** = 原材料仓库（知道怎么找到每一条样本，但"按需取用"）
- **DataLoader** = 传送带（自动打包成 Batch，协调后台预取）

想象一家汽车工厂：
- 原材料库（Dataset）里存放着所有零件，但不会一次性全搬出来
- 传送带（DataLoader）根据生产节奏，把零件自动打包成"一批次 32 辆车的零件"送到装配线

```python
from torch.utils.data import Dataset, DataLoader

class PretrainDataset(Dataset):
    def __init__(self, data_path, max_length=512):
        # 加载数据文件（mmap 模式，内存映射，不全量读入）
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.max_length = max_length

    def __len__(self):
        # 告诉 DataLoader 总共有多少条样本
        return len(self.data) // self.max_length

    def __getitem__(self, idx):
        # 按需获取第 idx 条样本（切片，不是全量加载）
        start = idx * self.max_length
        end = start + self.max_length + 1  # +1 因为需要 input 和 target 同时
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        
        x = chunk[:-1]  # 输入序列（除最后一个）
        y = chunk[1:]   # 目标序列（除第一个）—— 经典的"下一个 Token 预测"
        return x, y

# DataLoader 自动处理：批处理、随机打乱、多进程读取
dataset = PretrainDataset('./dataset/pretrain_t2t_mini.jsonl', max_length=512)
loader = DataLoader(
    dataset,
    batch_size=32,           # 每批 32 条样本
    shuffle=True,            # 随机打乱（防止过拟合）
    num_workers=4,           # 4 个后台进程并行读取
    pin_memory=True,         # 锁页内存，加速 CPU→GPU 传输
)
```

**注意 `x = chunk[:-1]` 和 `y = chunk[1:]`：** 这是语言模型训练的核心 trick——给定"序列的前 n 个词"，预测"后移一位的序列"。每一对 (x[i], y[i]) 就是一个（输入Token，目标下一个Token）的训练样本。

---

### 3. 混合精度训练：用 bfloat16 把显存砍一半

**为什么需要混合精度？**

默认情况下，PyTorch 使用 `float32`（32位浮点数）存储模型参数和激活值。对于 minimind-3 这样的 64M 参数模型：

$$
\text{模型参数显存} = 64 \times 10^6 \times 4 \text{ bytes} = 256 \text{ MB}
$$

如果改用 `bfloat16`（16位浮点数），显存占用减半：

$$
\text{bfloat16 参数显存} = 64 \times 10^6 \times 2 \text{ bytes} = 128 \text{ MB}
$$

**但为什么不全程用 bfloat16？**

bfloat16 精度较低，会导致某些计算（尤其是梯度更新、损失计算）出现数值不稳定。解决方案是**混合精度**：

- 前向传播（计算 loss）：用 `bfloat16`，速度快、省显存
- 反向传播（更新参数）：先把梯度转为 `float32`，用高精度更新

这就像：做草稿用铅笔（bfloat16，快但不精确），最终定稿用圆珠笔（float32，慢但精确）。

```python
from torch.cuda.amp import autocast, GradScaler

# 对于 bfloat16，不需要 GradScaler（它只对 float16 必要）
# 因为 bfloat16 的动态范围与 float32 相同，只是精度不同

# 训练循环中
with autocast(dtype=torch.bfloat16):
    # 整个前向传播在 bfloat16 下运行（自动转换）
    logits = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))

# loss 的值是 float32 精度（autocast 会自动处理）
loss.backward()      # 反向传播
optimizer.step()     # 参数更新（float32 精度）
optimizer.zero_grad()
```

**bfloat16 vs float16 的选择：**

| 类型 | 精度位 | 指数位（动态范围） | 说明 |
|------|--------|------------------|------|
| float32 | 23 位 | 8 位 | 最稳定，最慢 |
| **bfloat16** | **7 位** | **8 位** | 范围同 float32，精度略低 |
| float16 | 10 位 | 5 位 | 精度高但范围小，容易溢出 |

minimind 使用 `bfloat16` 而不是 `float16`，原因正是 bfloat16 保留了 float32 相同的指数位（动态范围），训练更稳定——这是 A100/H100/RTX4090 等新硬件的首选。

---

### 4. 梯度累积：用小显存模拟大 Batch

**问题：** 理想的批大小（Batch Size）应该很大（比如 1024 个样本），因为大 Batch 训练更稳定、收敛更快。但一张消费级 GPU 的显存只有 24GB，同时处理 1024 个长序列会爆显存。

**解决方案：梯度累积（Gradient Accumulation）**

类比：你想搬一批很重的箱子，但每次只能搬一个。你不用每次搬完都立刻整理（更新参数），而是连续搬 K 次（积累梯度），搬完一批后再统一整理（执行一次参数更新）。

数学上：积累 $K$ 步的梯度后再更新，等价于用 $K$ 倍大的 Batch Size 训练（近似等价）：

$$
\text{有效 Batch Size} = \text{actual\_batch\_size} \times \text{accumulation\_steps}
$$

```python
accumulation_steps = 8  # 积累 8 步的梯度再更新一次
optimizer.zero_grad()

for step, (x, y) in enumerate(loader):
    with autocast(dtype=torch.bfloat16):
        logits = model(x)
        # 注意：loss 要除以 accumulation_steps，保持梯度幅度不变
        loss = criterion(logits.view(-1, vocab_size), y.view(-1)) / accumulation_steps
    
    loss.backward()  # 梯度累积（不清空）
    
    if (step + 1) % accumulation_steps == 0:
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()    # 执行参数更新
        optimizer.zero_grad()  # 清空梯度，开始新一轮累积
```

**注意 `loss / accumulation_steps`：** 这很关键！如果不除，8 步累积的梯度幅度是单步的 8 倍，相当于学习率放大了 8 倍，训练会不稳定。

---

### 5. DDP：多卡并行的基本思想

当一张 GPU 不够快时，可以使用多张 GPU。**DDP（Distributed Data Parallel，分布式数据并行）** 是 PyTorch 中最常用的多卡训练方式。

**核心思想：**

把模型复制到每张 GPU，每张 GPU 处理不同的数据 Batch（数据并行）：
```
GPU 0：处理 Batch 0-31（计算梯度 g0）
GPU 1：处理 Batch 32-63（计算梯度 g1）
GPU 2：处理 Batch 64-95（计算梯度 g2）
GPU 3：处理 Batch 96-127（计算梯度 g3）
     ↓
所有 GPU 通信，将梯度取平均：g_avg = (g0 + g1 + g2 + g3) / 4
     ↓
每张 GPU 用 g_avg 更新自己的模型参数（保持同步）
```

类比：4 组工人同时搬箱子，搬完后统一汇报进度，再开始下一轮。

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# DDP 初始化（通常在脚本入口）
dist.init_process_group(backend='nccl')  # GPU 通信用 nccl
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 把模型包装成 DDP
model = MiniMind(config).to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 启动方式（命令行）
# torchrun --nproc_per_node=4 train_pretrain.py
```

minimind 支持单卡和多卡两种训练方式：
```bash
# 单卡训练
python train_pretrain.py

# 多卡训练（4 卡）
torchrun --nproc_per_node 4 train_pretrain.py
```

---

### 6. train_pretrain.py 训练循环：把所有知识串联

下面是 minimind `trainer/train_pretrain.py` 的核心训练循环骨架，加上详细注释：

```python
def train_epoch(model, loader, optimizer, lr_scheduler, scaler, args):
    """一个 epoch 的训练循环"""
    model.train()  # 开启训练模式（启用 Dropout 等）
    
    # 在多卡训练中，只有 rank=0 的进程打印日志
    if args.local_rank == 0:
        pbar = tqdm(total=len(loader), desc=f'[Epoch {args.current_epoch}]')
    
    total_loss = 0.0
    optimizer.zero_grad()  # 初始化梯度为 0
    
    for step, (X, Y) in enumerate(loader):
        # X: 输入 Token IDs [batch, seq_len]
        # Y: 目标 Token IDs [batch, seq_len]（X 右移一位）
        X = X.to(args.device)
        Y = Y.to(args.device)
        
        # ---- 前向传播（在 bfloat16 下运行）----
        with autocast(dtype=torch.bfloat16):
            logits, loss = model(X, Y)
            # logits: [batch, seq_len, vocab_size]
            # loss: 交叉熵损失（平均到每个 Token）
            loss = loss / args.accumulation_steps  # 梯度累积缩放
        
        # ---- 反向传播 ----
        loss.backward()  # 计算梯度（自动混合精度处理）
        
        # ---- 梯度累积：每 accumulation_steps 步才更新参数 ----
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()       # 更新参数
            lr_scheduler.step()    # 更新学习率（余弦退火等）
            optimizer.zero_grad()  # 清空梯度
        
        total_loss += loss.item() * args.accumulation_steps
        
        # ---- 定期保存检查点 ----
        if step % args.save_interval == 0 and args.local_rank == 0:
            torch.save(model.state_dict(), f'out/pretrain_{args.dim}.pth')
    
    return total_loss / len(loader)


def main():
    # ---- 1. 配置参数 ----
    args = get_args()
    
    # ---- 2. 初始化分布式（多卡时）或单卡 ----
    setup_distributed(args)
    
    # ---- 3. 构建模型 ----
    config = MiniMindConfig(
        hidden_size=args.dim,            # 例如 768
        num_hidden_layers=args.n_layers, # 例如 8
        num_heads=8,
        num_kv_heads=4,
        vocab_size=6400,
    )
    model = MiniMindForCausalLM(config).to(args.device)
    
    # ---- 4. 构建数据集和 DataLoader ----
    dataset = PretrainDataset(args.data_path, max_length=args.max_seq_len)
    loader = build_dataloader(dataset, args)
    
    # ---- 5. 优化器（AdamW）和学习率调度 ----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,           # 初始学习率，例如 5e-4
        weight_decay=0.1,     # L2 正则化
        betas=(0.9, 0.95),    # AdamW 动量系数
    )
    lr_scheduler = get_cosine_schedule(optimizer, args)
    
    # ---- 6. 训练循环 ----
    for epoch in range(args.epochs):
        args.current_epoch = epoch
        loss = train_epoch(model, loader, optimizer, lr_scheduler, None, args)
        if args.local_rank == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")
```

**几个关键点的汇总：**

| 组件 | 作用 | minimind 中的具体值 |
|------|------|-------------------|
| Adam 优化器 | 自适应学习率，比 SGD 收敛更稳定 | AdamW，lr=5e-4 |
| 余弦退火调度 | 学习率从高到低逐步降低，避免震荡 | warmup + cosine decay |
| 梯度裁剪 | 限制梯度最大范数，防止梯度爆炸 | max_norm=1.0 |
| 检查点保存 | 定期保存模型，支持断点续训 | 每 save_interval 步保存 |

---

### 7. 整体数据流回顾

让我们把整个训练过程可视化为一个流水线：

```
原始文本文件
     │ (PretrainDataset.__getitem__)
     ↓
Token ID 序列 [seq_len+1]
     │ → x = seq[:-1], y = seq[1:]
     ↓
DataLoader 打包
     │ → [batch, seq_len] tensor
     ↓
.to('cuda') + autocast(bfloat16)
     │
     ↓
模型前向传播 MiniMind
  Embedding → N × TransformerBlock → LM Head → logits
     │
     ↓
交叉熵损失 CrossEntropy(logits, y)
     │
     ↓
loss.backward() 反向传播
     │
     ↓
（累积 K 步后）optimizer.step() 参数更新
     │
     ↓
下一个 Batch...
```

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| Tensor | PyTorch 的多维数组，支持 GPU 和自动求导 |
| Dataset / DataLoader | 按需读取数据的流水线工厂，不会全量加载进内存 |
| bfloat16 / autocast | 前向用半精度省显存，梯度计算自动保持精度 |
| 梯度累积 | 连续多步积累梯度后再更新，模拟大 Batch Size |
| DDP | 多 GPU 数据并行，梯度平均后同步所有副本 |
| 训练循环 | 前向→计算 loss→反向→梯度裁剪→优化器更新 |

---

## 🏋️ 习题集

**基础题：**

1. 一个 `shape=(32, 512, 768)` 的 Tensor，三个维度分别代表什么（假设是 LLM 的隐状态）？
2. 为什么 `y = chunk[1:]` 而不是 `y = chunk[:-1]`？语言模型的训练目标是什么？
3. 梯度累积 4 步后更新，有效 Batch Size 是原来的几倍？loss 为什么要除以 accumulation_steps？

**进阶题：**

1. 如果不使用 `pin_memory=True`，数据从 CPU 传到 GPU 的速度会更慢。尝试解释其原因（提示：锁页内存 vs 可换页内存）。
2. bfloat16 有 7 位精度位，float16 有 10 位精度位，但训练 LLM 时 bfloat16 通常更稳定。解释为什么（提示：动态范围 vs 精度的权衡）。
3. 在 minimind 的 `train_pretrain.py` 中找到学习率调度的代码，理解 warmup 阶段的作用（为什么不从一开始就用最高学习率？）。

---

> 下一课：**L04 - minimind 项目导览与环境搭建**
