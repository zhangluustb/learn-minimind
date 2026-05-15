# L11 - 预训练（Pretraining）——语言建模

> **"小孩子学说话，不需要老师教语法——只需要听大人说话，然后模仿。"**

---

## 📌 本节目标

1. 理解语言建模的核心目标：Next-Token Prediction
2. 掌握预训练数据的格式和构建方式
3. 完整理解 `train_pretrain.py` 的执行流程
4. 学会读懂 loss 曲线，判断训练是否有效

---

## 📚 前置知识

- 完成 L10（理解完整 MiniMindBlock）
- 了解 PyTorch 的基本训练循环（L03）

---

## 正文

### 1. Next-Token Prediction：最朴素的学习目标

大语言模型的训练目标看似简单得不能再简单——**给定前面的 tokens，预测下一个 token**。

想象你在读一本书：

```
我今天 → [模型预测] → 去
去公园 → [模型预测] → 散步
散步很 → [模型预测] → 开心
```

每个预测都是从一个分布中采样。模型的目标就是让**真实 token 的概率尽可能高**——用交叉熵损失函数衡量：

$$L = -\sum_{i=1}^{T} \log P(x_{i+1} | x_1, x_2, ..., x_i)$$

其中 $x_i$ 是第 $i$ 个 token，$P(x_{i+1} | x_1, ..., x_i)$ 是模型输出的第 $i+1$ 个位置对真实 token 的概率。

> **类比**：这就像在做**无限接龙**游戏。你读过越多文本，就越能准确预测下一个词。通过重复玩这个游戏，模型会学到语言中隐含的所有规律：语法、常识、逻辑和创意。

### 2. 预训练数据：互联网的"碎片化"拼图

预训练数据不需要标注，直接从互联网、书籍、代码库爬取即可。minimind 的 `pretrain_t2t` 文件通常包含大量文本，格式特别简单：

```
# pretrain_t2t.jsonl
{"text": "今天天气很好，我们一起去公园玩。"}
{"text": "深度学习是机器学习的一个分支。"}
{"text": "Python 是一门流行的编程语言。"}
...（百万级数据行）
```

**数据处理流程**

```
原始文本 → 分词器编码 → Token 序列 → 分块（chunk_size=512）→ 小批量加载
```

minimind 的 `lm_dataset.py` 做的就是这个转换：

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokens = []
        
        # 逐行读文件，把所有文本都转成 token 序列
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = data['text']
                # 编码成 token
                encoded = tokenizer.encode(text)
                self.tokens.extend(encoded)
        
        # tokens 数量必须能被 chunk_size 整除
        self.tokens = self.tokens[:len(self.tokens) // max_length * max_length]
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx * max_length : (idx + 1) * max_length]
        # input_ids 是前 511 个 token，labels 是后 511 个 token（错开一位）
        return {
            'input_ids': chunk[:-1],  # [token1, token2, ..., token510]
            'labels': chunk[1:]       # [token2, token3, ..., token511]
        }
```

> **关键细节**：input_ids 和 labels 是错开一个位置的。这样模型预测 token_1 时用 token_0，预测 token_2 时用 token_0+token_1，以此类推。

### 3. train_pretrain.py 的核心流程

一个完整的预训练脚本包括以下关键步骤：

```python
def train_pretrain():
    # ① 配置超参数
    config = MiniMindConfig(
        dim=512, n_heads=8, n_kv_heads=4, n_layers=4,
        vocab_size=6400, max_seq_len=512,
        use_moe=False,  # 预训练不用 MoE
    )
    
    # ② 加载 tokenizer 和数据
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    dataset = PretrainDataset('./data/pretrain_t2t.jsonl', tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # ③ 初始化模型
    model = MiniMind(config)
    model.to(device)
    
    # ④ 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=1e-4)
    # Warm up: 前 1000 步从 0 线性增加到最大学习率，然后指数衰减
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=total_steps
    )
    
    # ⑤ 混合精度训练（节省显存，加快速度）
    scaler = GradScaler()
    
    # ⑥ 训练循环
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度前向传播
            with autocast():
                outputs = model(input_ids)
                logits = outputs.logits  # [B, T, vocab_size]
                
                # 计算 loss（对所有 token 位置求平均）
                loss = cross_entropy(
                    logits.view(-1, config.vocab_size),  # [B*T, vocab_size]
                    labels.view(-1)                       # [B*T]
                )
            
            # 反向传播 + 梯度累积
            scaler.scale(loss).backward()
            
            # 梯度裁剪：防止爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # 定期记录和保存
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item():.4f}")
                writer.add_scalar('Loss/train', loss.item(), global_step)
    
    # 保存模型权重
    torch.save(model.state_dict(), f'./out/pretrain_{config.dim}.pth')
```

### 4. Loss 曲线的解读

当模型开始训练时，loss 应该呈现以下特征：

```
Loss 曲线演进
^
|    ★ 理想曲线：平稳下降，最后收敛
|   /\
|  /  \___
| /       \___
|/           \___
+----+----+----+-----> 训练步数
```

| 症状 | 原因 | 解决方案 |
|------|------|--------|
| Loss 不下降 | 学习率太小 | 增大学习率（1e-4 → 1e-3） |
| Loss 振荡爆炸 | 学习率太大 | 减小学习率，启用 Warm-up |
| Loss 卡住不动 | 数据问题或模型容量不足 | 检查数据是否重复，增加模型深度 |
| Loss 先暴跌后停滞 | 过拟合（模型记住了训练集） | 增加数据量，启用 Dropout |

**为什么最后 loss 不能降到 0？**

因为每个 token 的预测都是概率性的，同样的前文可能对应多个合理的后续词。完美的损失 $L_{\text{min}}$ 取决于数据的**熵**（多样性）：

$$L_{\text{min}} = H(\text{data}) \approx \log(\text{有效词汇数})$$

对于通常的中文文本，optimal loss 约为 2-3（意味着平均每个 token 从约 10-20 个合理的后续词中选择）。

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| Next-Token Prediction | 给定前面 N-1 个 token，预测第 N 个 token 的概率分布 |
| 预训练目标 | 最小化交叉熵损失，让真实 token 的概率最大化 |
| 数据格式 | JSONL 的纯文本，无需标注，可直接从互联网爬取 |
| Token 对齐 | Input_ids 比 labels 早一个位置，形成自回归目标 |
| 混合精度 | BF16 或 FP16 计算，减少显存占用和提升速度 |
| Loss 曲线 | 平稳下降是好信号，振荡/停滞提示需要调超参 |

---

## 🧪 习题集 11

**题目 1（目标函数）：** 已知一条文本序列 `[我, 今天, 去, 公园]`，tokenizer 后为 `[1024, 2048, 3072, 4096]`。如果用这条序列进行预训练，input_ids 和 labels 分别是什么？请写出完整的 Token 列表。

**题目 2（损失计算）：** 模型在某个 batch 中输出的 logits shape 为 `[16, 512, 6400]`（batch_size, seq_len, vocab_size）。计算一步 backward pass 所需的显存（FP16，考虑梯度）。

**题目 3（数据处理）：** 假设你有 1GB 的原始文本文件，平均每行 500 characters，tokenizer 效率为 1 token ≈ 3 characters。如果 max_length=512，chunk_size=32，需要多少个 batch 来遍历整个数据集一遍？

**题目 4（学习率调度）：** Warm-up 阶段为什么必要？如果直接用 1e-4 学习率而不 warm-up，会发生什么？

**题目 5（设计题）：** minimind 的预训练中，是否需要给模型输入"我今天"就让它预测"去"，还是需要随机 mask 一些 token 然后填空（类似 BERT）？各有什么优劣？

---

> 下一课我们将学习 **L12 - 监督微调（SFT）与对话模板**，把预训练好的语言模型教会"听指令"。
