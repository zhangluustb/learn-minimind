# L17 - Tool Use & Agentic RL

> **"自己算不出答案？问 Wolfram Alpha。找不到图片？问 Google Images。模型也可以。"**

---

## 📌 本节目标

1. 理解 Function Calling 的协议设计
2. 掌握特殊 token（`<tool_call>` 和 `<tool_response>`）的意义
3. 能设计一个完整的工具调用强化学习流程
4. 了解 minimind 的 web_demo.py 如何整合所有功能

---

## 📚 前置知识

- 完成 L12（SFT）、L15（RLAIF）
- 了解 JSON 格式和 API 调用

---

## 正文

### 1. Function Calling 的动机

纯语言模型有个根本局限：**信息是 offline 的**。

```
问：现在几点？
模型回答：我的训练数据到 2023 年 12 月，我不知道现在是几点。

想要的：
问：现在几点？
模型思考：我需要查询当前时间
模型调用：[call get_current_time()]
得到结果：2024-05-14 14:30:00
模型回答：现在是下午 2 点 30 分。
```

Function Calling 就是支持这个能力。

### 2. 特殊 Token 与对话格式

minimind 预留了两个特殊 token：`<tool_call>` 和 `<tool_response>`。

首先扩大词表，添加这两个 token：

```python
tokenizer.add_tokens(['<tool_call>', '<tool_response>'])
```

然后修改 chat_template：

```
<|im_start|>system
你是一个有帮助的助手。你可以使用以下工具：
1. get_current_time() - 获取当前时间
2. search_web(query: str) - 网络搜索
<|im_end|>
<|im_start|>user
现在几点？
<|im_end|>
<|im_start|>assistant
<tool_call>
{
  "function": "get_current_time",
  "arguments": {}
}
<tool_call>
<|im_end|>
<|im_start|>tool
<tool_response>
{
  "result": "2024-05-14 14:30:00"
}
<tool_response>
<|im_end|>
<|im_start|>assistant
现在是下午 2 点 30 分。
<|im_end|>
```

### 3. Agentic RL 的训练流程

与 SFT 不同，agentic RL 需要：
1. 模型生成工具调用
2. 执行工具
3. 回读结果
4. 继续推理

```python
class ToolUseEnvironment:
    def __init__(self, tools_dict):
        self.tools = tools_dict  # {'get_time': func, 'search': func}
    
    def execute_tool_call(self, tool_name, arguments):
        """解析模型的工具调用，执行真实工具"""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}
        
        try:
            result = self.tools[tool_name](**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def step(self, prompt, model, tokenizer, max_steps=3):
        """
        智能体的推理循环
        """
        messages = [{"role": "user", "content": prompt}]
        history = []
        
        for step in range(max_steps):
            # 生成：可能是普通回答或工具调用
            response = model.generate(
                tokenizer.apply_chat_template(messages),
                temperature=0.7, max_new_tokens=256
            )
            response_text = tokenizer.decode(response)
            history.append(response_text)
            
            # 检测是否有工具调用
            if '<tool_call>' in response_text:
                # 解析工具调用（通常在 JSON 块中）
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    tool_call_json = json.loads(response_text[json_start:json_end])
                    
                    # 执行工具
                    tool_result = self.execute_tool_call(
                        tool_call_json['function'],
                        tool_call_json.get('arguments', {})
                    )
                    
                    # 把结果回读给模型
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_result)
                    })
                except:
                    # 解析失败，提示模型
                    messages.append({"role": "user", "content": "格式错误，请重试"})
            else:
                # 没有工具调用，这是最终答案
                break
        
        return history[-1]  # 返回最后一个回答
```

### 4. RL 的奖励设计

工具使用能否提升性能取决于奖励函数的设计：

```python
def compute_tool_use_reward(prompt, response, ground_truth, executed_tools):
    """
    衡量一个 agentic 推理过程的好坏
    """
    reward = 0.0
    
    # ① 答案正确性（主要奖励）
    correctness = evaluate_correctness(response, ground_truth)  # 0-1
    reward += 0.7 * correctness
    
    # ② 工具调用的必要性（辅助奖励）
    # 如果问题确实需要工具（如查询时间），应该用工具
    needs_tools = requires_external_knowledge(prompt)
    if needs_tools and len(executed_tools) > 0:
        reward += 0.2  # 正确地调用了工具
    elif not needs_tools and len(executed_tools) == 0:
        reward += 0.1  # 不需要工具时没有瞎调用
    else:
        reward -= 0.1  # 错误地调用/不调用工具
    
    # ③ 推理效率（步数惩罚）
    num_steps = len(executed_tools) + 1
    efficiency_penalty = 0.05 * (num_steps - 1)
    reward -= efficiency_penalty
    
    return reward

def train_tool_use(model, train_data, optimizer, num_epochs=3):
    """
    用强化学习优化工具使用能力
    """
    env = ToolUseEnvironment(TOOLS)
    
    for epoch in range(num_epochs):
        for prompt, ground_truth in train_data:
            # 前向：生成推理过程
            with torch.no_grad():
                trajectory = env.step(prompt, model, tokenizer)
            
            # 计算奖励
            reward = compute_tool_use_reward(
                prompt, trajectory, ground_truth, 
                env.executed_tools
            )
            
            # 反向：优化（这里用 REINFORCE 算法）
            loss = -trajectory_log_prob * reward
            loss.backward()
            optimizer.step()
```

### 5. minimind 的 web_demo.py

minimind 提供了一个完整的 WebUI 演示：

```python
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained('./out/final_model/')
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    return model, tokenizer

model, tokenizer = load_model()

st.title("MiniMind - 智能助手")

# 工具定义
TOOLS = {
    'get_current_time': lambda: datetime.now().isoformat(),
    'search_web': lambda query: mock_search(query),  # 模拟搜索
    'calculator': lambda expr: eval(expr)  # 计算
}

user_input = st.text_input("你：", placeholder="问我任何问题…")

if user_input:
    env = ToolUseEnvironment(TOOLS)
    response = env.step(user_input, model, tokenizer)
    st.write(f"助手：{response}")
```

只需 `streamlit run web_demo.py` 就能启动一个交互式的聊天应用！

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| Function Calling | 让模型学会"我不知道，让我问一下" |
| 特殊 Token | `<tool_call>` 和 `<tool_response>` 标记工具交互的边界 |
| Agentic RL | 强化学习驱动的多步推理，每步可以调用工具 |
| 环境设计 | ToolUseEnvironment 模拟真实工具的执行 |
| 奖励塑造 | 正确答案 + 合理调用工具 - 不必要的工具调用 |

---

## 🧪 习题集 17

**题目 1（格式设计）：** 为什么要用特殊 token `<tool_call>` 而不是让模型输出"我要调用功能 X"这样的自然文本？

**题目 2（JSON 解析）：** 模型输出的工具调用 JSON 可能格式错误或不完整（如缺少 arguments）。应该怎么处理？

**题目 3（Tool Use 奖励）：** 如果你把 "不乱调用工具" 的奖励设得太高，会发生什么？反之呢？

**题目 4（多轮对话）：** 在工具调用的流程中，如何处理用户的后续问题（如"刚才的答案能详细说明吗"）？

**题目 5（设计题）：** 现在有 10 个工具（计算、搜索、天气、股票等）。怎么设计系统提示，让模型知道什么时候该用哪个工具？

---

> 下一课我们将学习 **L18 - 部署与生态**，让 minimind 跑在各种推理框架上。
