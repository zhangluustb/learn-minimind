# L18 - 部署与生态——llama.cpp / vLLM / OpenAI API

> **"训练完模型只是开始，怎么让它跑得快、跑到所有地方，才是真本事。"**

---

## 📌 本节目标

1. 了解量化技术（INT8, INT4）如何减少显存占用和加快推理
2. 掌握 llama.cpp 的离线部署方案
3. 理解 vLLM 的批量推理优化
4. 能用 serve_openai_api.py 对外提供 OpenAI 兼容的接口

---

## 📚 前置知识

- 完成 L09（理解 KV-Cache）
- 了解什么是量化和精度转换

---

## 正文

### 1. 量化：用更少的 bit 表示权重

训练好的 minimind 模型用 FP32（32 位浮点数）存储，每个参数 4 字节，总共 64M × 4 = 256 MB。

**量化**就是把这些数字压缩到更少的 bit：

| 精度 | 范围 | 存储 | 与原模型的表现 |
|------|------|------|--------------|
| FP32 | -3.4e38 ~ 3.4e38 | 4 字节 | 100%（基准） |
| FP16 / BF16 | 较小 | 2 字节 | 99.5% ✅ |
| INT8 | -128 ~ 127 | 1 字节 | 98% ✅ |
| INT4 | -8 ~ 7（4 bit） | 0.5 字节 | 90-95% ⚠️ |

一个 64M 的模型：
- FP32：256 MB
- FP16：128 MB
- INT8：64 MB
- INT4：32 MB

#### 量化的原理

假设权重范围是 $[-a, a]$，要映射到 INT8 的 $[-128, 127]$：

$$w_{\text{int8}} = \text{round}\left(\frac{w}{\text{scale}}\right), \quad \text{scale} = \frac{a}{127}$$

推理时再把 INT8 映射回来：

$$w = w_{\text{int8}} \times \text{scale}$$

实际应用通常用 **per-channel 量化**——每个权重矩阵的行或列用各自的 scale，减少信息损失。

### 2. llama.cpp：离线推理的标配

[llama.cpp](https://github.com/ggerganov/llama.cpp) 是一个 C++ 推理引擎，特别适合 CPU 推理或资源受限的场景。

**安装与转换**

```bash
# 1. 把 PyTorch 模型转换为 GGML 格式
python convert-hf-to-ggml.py ./out/final_model --outfile model.ggml

# 2. 量化到 INT4（体积从 256MB → 32MB）
./quantize model.ggml model-q4.gguf q4_0

# 3. 推理（不需要 GPU，纯 CPU）
./main -m model-q4.gguf -n 128 -p "什么是大语言模型？"
```

**在 Python 中调用**

```python
from llama_cpp import Llama

# 加载量化模型
llm = Llama(
    model_path="./model-q4.gguf",
    n_gpu_layers=0,  # 0 表示 CPU，可设大于 0 使用 GPU
    n_threads=8,     # CPU 线程数
)

response = llm(
    "什么是大语言模型？",
    max_tokens=128,
    temperature=0.7,
    top_p=0.9
)

print(response["choices"][0]["text"])
```

### 3. vLLM：高吞吐推理框架

vLLM 是一个为大规模推理优化的框架，核心是 **Paged Attention**——类似操作系统的内存分页。

**vLLM vs 普通推理：**

- 普通推理：KV-Cache 连续分配 → 碎片 → 显存浪费
- vLLM：分页管理 → 多个请求共享 → 吞吐提升 2-4 倍

**部署 vLLM 服务**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="./out/final_model/",
    tensor_parallel_size=1,  # 单 GPU
    gpu_memory_utilization=0.9
)

prompts = [
    "什么是深度学习？",
    "解释注意力机制",
    "什么是 Transformer？"
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=128
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### 4. serve_openai_api.py：兼容 OpenAI 的接口

minimind 提供了一个 Flask 服务器，用 OpenAI API 格式对外暴露模型：

```python
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

app = Flask(__name__)
model = AutoModelForCausalLM.from_pretrained('./out/final_model/')
tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completion():
    data = request.json
    messages = data.get('messages', [])
    
    # 应用 chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # 生成
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=1024,
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9)
        )
    
    response_text = tokenizer.decode(outputs[0])
    
    # OpenAI 兼容格式
    return jsonify({
        'model': 'minimind',
        'object': 'text_completion',
        'choices': [{
            'message': {'content': response_text},
            'finish_reason': 'stop'
        }]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

**客户端调用**

```python
import openai

openai.base_url = "http://localhost:8000/v1"
openai.api_key = "not-needed"

response = openai.chat.completions.create(
    model="minimind",
    messages=[
        {"role": "user", "content": "什么是大语言模型？"}
    ]
)

print(response.choices[0].message.content)
```

### 5. 与生态集成

**ollama**：傻瓜式部署

```bash
# 自动下载、量化、部署
ollama run minimind
```

**FastGPT/Open-WebUI**：低代码前端

把 minimind 的 OpenAI API 端口添加到这些平台，即刻获得专业 WebUI！

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| 量化 | 把 FP32 压缩到 INT8/INT4，减 4-8 倍体积，牺牲 5-10% 性能 |
| Per-Channel Quantization | 每行/列单独计算 scale，减少信息损失 |
| llama.cpp | C++ 推理引擎，完美支持 CPU 和量化模型 |
| vLLM | 分页 attention，吞吐提升 2-4 倍 |
| OpenAI API 兼容 | 标准接口，方便与其他系统集成 |
| 生态工具 | ollama（部署）、WebUI（前端）等生态丰富 |

---

## 🧪 习题集 18

**题目 1（量化的权衡）：** INT4 量化可以把模型从 256MB 缩到 32MB，为什么不直接全用 INT4？

**题目 2（llama.cpp 的优势）：** 为什么 llama.cpp 在 CPU 上的推理速度能接近或超过某些 GPU 推理框架？

**题目 3（Paged Attention）：** vLLM 的 Paged Attention 相比普通 KV-Cache，在多请求场景下的优势是什么？

**题目 4（API 兼容性）：** 为什么要让 minimind 的 API 兼容 OpenAI 格式？有什么好处？

**题目 5（部署选型）：** 给定场景：
- 场景 A：单用户，本地 GPU
- 场景 B：多用户，数据中心
- 场景 C：移动设备

分别应选 llama.cpp / vLLM / 其他？为什么？

---

> 下一课我们将学习 **L19 - 简历撰写指南**，用 minimind 项目打动面试官。
