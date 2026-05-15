（交流可以用英文，本文档中文，保留这句）

# MiniMind 学习教程项目说明

## 项目目标

编写一份 minimind 的由浅入深学习教程，包含 20 节课程、面试八股文、
STAR 面试稿、简历模板和哆啦A梦漫画图解。

每章初版：Now update todo list, foreach chapterX.md, read @CLAUDE.md and @index.md to get context (do this every time), then write chapterX.md

丰富每章内容：Now update todo list, foreach chapterX.md, read @CLAUDE.md and @index.md to get context (do this every time), then enrich chapterX.md section by section with textual content

## 工具说明

当需要时，可以通过深度研究得到想要的答案。

## 源项目信息

- 项目名：minimind
- 地址：https://github.com/jingyaogong/minimind
- Star 数：30k+
- 技术栈：Python / PyTorch / Transformers
- 核心功能：用约 3 元、2 小时从零训练出 64M 参数语言模型，覆盖预训练→SFT→LoRA→RLHF→蒸馏全流程

## 教程大纲

### Phase 1：零基础入门（L01-L04）🟢

1. **大语言模型简介与 minimind 概览**
   - 1.1 什么是大语言模型（LLM）
   - 1.2 GPT / ChatGPT / DeepSeek 对比
   - 1.3 minimind 能做什么，为什么值得学
   - 1.4 项目 Star 历史与影响力概览
   - 习题集 1

2. **Transformer 核心原理——注意力机制**
   - 2.1 Seq2Seq 的局限与 Attention 的诞生
   - 2.2 Self-Attention 公式推导
   - 2.3 Multi-Head Attention
   - 2.4 为什么 Transformer 能"记住"上下文
   - 习题集 2

3. **PyTorch 快速上手与数据流**
   - 3.1 张量（Tensor）基础
   - 3.2 Dataset & DataLoader
   - 3.3 AutoCast 混合精度训练
   - 3.4 分布式训练 DDP 入门
   - 习题集 3

4. **minimind 项目导览与环境搭建**
   - 4.1 目录结构全解
   - 4.2 requirements.txt 关键依赖
   - 4.3 Tokenizer 文件解读
   - 4.4 第一次运行：对 minimind 提问
   - 习题集 4

### Phase 2：核心组件拆解（L05-L10）🔵

5. **Tokenizer——文字的"编码本"**
   - 5.1 BPE 分词原理
   - 5.2 minimind Tokenizer（6400 词表）
   - 5.3 chat_template 与特殊 token
   - 5.4 train_tokenizer.py 源码精读
   - 习题集 5

6. **Embedding 与旋转位置编码（RoPE）**
   - 6.1 词向量 Embedding
   - 6.2 绝对位置编码 vs 相对位置编码
   - 6.3 RoPE 数学原理与代码实现
   - 6.4 YaRN——长文本外推技巧
   - 习题集 6

7. **Multi-Head Attention（含 GQA/QK-Norm）**
   - 7.1 从标准 MHA 到 GQA（分组查询注意力）
   - 7.2 KV 头数 < Q 头数的工程意义
   - 7.3 QK-Norm 稳定训练
   - 7.4 Flash Attention 加速原理
   - 习题集 7

8. **FFN、SwiGLU 与 MoE**
   - 8.1 FFN 的"放大-激活-收缩"
   - 8.2 SwiGLU 激活函数
   - 8.3 MoE（混合专家）路由机制
   - 8.4 minimind MoE 源码解析（MOEFeedForward）
   - 习题集 8

9. **KV-Cache 与推理加速**
   - 9.1 自回归解码原理
   - 9.2 KV-Cache 工作机制
   - 9.3 past_key_value 在代码中的流转
   - 9.4 推理速度对比实验
   - 习题集 9

10. **RMSNorm 与完整 MiniMindBlock**
    - 10.1 LayerNorm vs RMSNorm
    - 10.2 Pre-Norm vs Post-Norm
    - 10.3 MiniMindBlock 整体数据流
    - 10.4 Residual Connection 为什么重要
    - 习题集 10

### Phase 3：训练流程串联（L11-L15）🟣

11. **预训练（Pretraining）——语言建模**
    - 11.1 Next-Token Prediction 目标函数
    - 11.2 pretrain_t2t 数据集格式
    - 11.3 train_pretrain.py 全流程解析
    - 11.4 Loss 曲线解读
    - 习题集 11

12. **监督微调（SFT）与对话模板**
    - 12.1 Instruction Tuning 原理
    - 12.2 chat_template 与 <|im_start|>/<|im_end|>
    - 12.3 train_full_sft.py 与 loss masking
    - 12.4 SFT 前后效果对比
    - 习题集 12

13. **LoRA 参数高效微调**
    - 13.1 为什么需要 LoRA
    - 13.2 低秩分解原理：ΔW = BA
    - 13.3 model_lora.py 源码精读
    - 13.4 train_lora.py 与 convert_model.py
    - 习题集 13

14. **RLHF——DPO 算法原理与实践**
    - 14.1 RLHF 三步走：SFT→RM→PPO
    - 14.2 DPO 绕过奖励模型的思路
    - 14.3 train_dpo.py 代码解析
    - 14.4 偏好数据集构建
    - 习题集 14

15. **RLAIF——PPO / GRPO / CISPO**
    - 15.1 RLAIF 与 RLHF 的区别
    - 15.2 PPO：Actor-Critic 强化学习
    - 15.3 GRPO：组相对策略优化
    - 15.4 rollout_engine.py 解耦设计
    - 习题集 15

### Phase 4：高级特性与面试（L16-L20）🟠

16. **模型蒸馏——知识传承**
    - 16.1 知识蒸馏原理（Teacher-Student）
    - 16.2 软标签 vs 硬标签
    - 16.3 train_distillation.py 源码解析
    - 16.4 minimind2-DeepSeek-R1 蒸馏案例
    - 习题集 16

17. **Tool Use & Agentic RL**
    - 17.1 Function Calling 协议设计
    - 17.2 <tool_call>/<tool_response> token
    - 17.3 train_agent.py 多轮工具调用训练
    - 17.4 web_demo.py WebUI 演示
    - 习题集 17

18. **部署与生态——llama.cpp / vLLM / OpenAI API**
    - 18.1 serve_openai_api.py 兼容接口
    - 18.2 llama.cpp 量化部署
    - 18.3 vLLM / ollama 推理框架
    - 18.4 与 FastGPT / Open-WebUI 集成
    - 习题集 18

19. **简历撰写指南**
    - 19.1 STAR 法则在简历中的应用
    - 19.2 minimind 项目的 4 种简历版本
    - 19.3 量化数据对照表
    - 19.4 不同岗位方向调整
    - 习题集 19

20. **STAR 面试法完整稿**
    - 20.1 STAR 法则介绍
    - 20.2 自我介绍三版本（30s/1min/3min）
    - 20.3 技术难点 STAR 场景 × 7
    - 20.4 模拟面试 12 轮
    - 习题集 20

## 面试材料大纲

### 基础面试（interview/01-05）
- 01-项目介绍话术.md（30秒/1分钟/3分钟）
- 02-模型架构面试题.md（Transformer/Attention/MoE）
- 03-训练流程面试题.md（预训练/SFT/RLHF）
- 04-工程实践面试题.md（分布式/推理优化/部署）
- 05-综合追问与深挖题.md

### 深度八股文（interview/06-10）
- 06-Transformer与Attention深度拷问30题.md
- 07-训练技巧与优化器面试50题.md
- 08-RLHF与强化学习面试30题.md
- 09-LLM工程实践面试30题.md
- 10-minimind专属面试50题.md

## 哆啦A梦漫画规划

| 编号 | 文件名 | 对应课程 | 漫画主题 |
|------|--------|---------|---------|
| 01 | 01-llm-overview.png | L01/L02 | LLM 全景概览 |
| 02 | 02-attention.png | L02 | Self-Attention 工作原理 |
| 03 | 03-transformer-block.png | L07/L10 | Transformer Block 结构 |
| 04 | 04-tokenizer.png | L05 | Tokenizer 编码流程 |
| 05 | 05-rope.png | L06 | RoPE 旋转位置编码 |
| 06 | 06-moe.png | L08 | MoE 混合专家路由 |
| 07 | 07-kv-cache.png | L09 | KV-Cache 推理加速 |
| 08 | 08-pretrain.png | L11 | 预训练数据流 |
| 09 | 09-sft.png | L12 | SFT 对话微调 |
| 10 | 10-lora.png | L13 | LoRA 低秩分解 |
| 11 | 11-rlhf-dpo.png | L14 | RLHF vs DPO |
| 12 | 12-grpo.png | L15 | GRPO 强化学习 |
| 13 | 13-distillation.png | L16 | 知识蒸馏 |
| 14 | 14-tooluse.png | L17 | Tool Use 工具调用 |
| 15 | 15-deploy.png | L18 | 部署生态全景 |

## 教程特色

- 每章节包含理论讲解、实例分析、代码示例
- 大量习题覆盖概念理解、设计实践、问题解决
- 渐进式学习路径，从基础到高级
- 面试八股文 190+ 题，覆盖项目所有知识点
- STAR 面试稿 + 简历模板，直接用于求职
- 哆啦A梦漫画图解，降低学习门槛

---

## 任务清单（Todo List）

### Phase 1: 基础设施 🏗️
- [x] 创建项目目录结构
- [x] 创建 CLAUDE.md（本文件）
- [ ] 创建 README.md

### Phase 2: 课程内容 📚
- [ ] 编写 L01-大语言模型简介与minimind概览.md
- [ ] 编写 L02-Transformer核心原理注意力机制.md
- [ ] 编写 L03-PyTorch快速上手与数据流.md
- [ ] 编写 L04-minimind项目导览与环境搭建.md
- [ ] 编写 L05-Tokenizer文字的编码本.md
- [ ] 编写 L06-Embedding与旋转位置编码RoPE.md
- [ ] 编写 L07-MultiHeadAttention含GQA和QKNorm.md
- [ ] 编写 L08-FFN与SwiGLU和MoE.md
- [ ] 编写 L09-KV-Cache与推理加速.md
- [ ] 编写 L10-RMSNorm与完整MiniMindBlock.md
- [ ] 编写 L11-预训练Pretraining语言建模.md
- [ ] 编写 L12-监督微调SFT与对话模板.md
- [ ] 编写 L13-LoRA参数高效微调.md
- [ ] 编写 L14-RLHF之DPO算法原理与实践.md
- [ ] 编写 L15-RLAIF之PPO和GRPO和CISPO.md
- [ ] 编写 L16-模型蒸馏知识传承.md
- [ ] 编写 L17-ToolUse与AgenticRL.md
- [ ] 编写 L18-部署与生态llamacpp和vLLM和OpenAI-API.md
- [ ] 编写 L19-简历撰写指南.md
- [ ] 编写 L20-STAR面试法完整稿.md

### Phase 3: 面试材料 🎯
- [ ] 编写 interview/01-项目介绍话术.md
- [ ] 编写 interview/02-模型架构面试题.md
- [ ] 编写 interview/03-训练流程面试题.md
- [ ] 编写 interview/04-工程实践面试题.md
- [ ] 编写 interview/05-综合追问与深挖题.md
- [ ] 编写 interview/06-Transformer与Attention深度拷问30题.md
- [ ] 编写 interview/07-训练技巧与优化器面试50题.md
- [ ] 编写 interview/08-RLHF与强化学习面试30题.md
- [ ] 编写 interview/09-LLM工程实践面试30题.md
- [ ] 编写 interview/10-minimind专属面试50题.md

### Phase 4: 哆啦A梦漫画 🎨
- [ ] 生成漫画 Prompt 文件 assets/comics/prompts.md

### Phase 5: Web 应用 🌐
- [ ] 初始化 Next.js + package.json
- [ ] 创建 lib/lessons-data.ts
- [ ] 创建 lib/interview-data.ts
- [ ] 创建 lib/readMarkdown.ts
- [ ] 创建 lib/mdTitle.ts
- [ ] 创建 app/layout.tsx + globals.css
- [ ] 创建 app/page.tsx（首页）
- [ ] 创建 app/learn/page.tsx（课程列表）
- [ ] 创建 app/lesson/[slug]/page.tsx（课程详情）
- [ ] 创建 app/interview/page.tsx（面试列表）
- [ ] 创建 app/interview/[slug]/page.tsx（面试详情）
- [ ] 创建 components/SiteNav.tsx
- [ ] 创建 components/Hero.tsx
- [ ] 创建 components/LearningPath.tsx
- [ ] 创建 components/MarkdownArticle.tsx
- [ ] 创建 components/InterviewPreview.tsx
- [ ] 创建 components/Footer.tsx
- [ ] 验证 npm run build

### Phase 6: 构建脚本 🔧
- [ ] 创建 scripts/build_html.py
- [ ] 创建 scripts/build_pdf.py

### Phase 7: 收尾 ✅
- [ ] 创建 README.md
- [ ] 质量检查
- [ ] git init && commit
