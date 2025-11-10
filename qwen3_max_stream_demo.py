import os
from openai import OpenAI
from dotenv import load_dotenv  # 👈 新增：加载环境变量

# 初始化 OpenAI 客户端（使用 DashScope 兼容模式）
load_dotenv()  # 👈 从 .env 文件加载环境变量

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量获取 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # DashScope OpenAI 兼容 API 地址
)

# 创建流式对话请求
completion = client.chat.completions.create(
    model="qwen3-max",  # Qwen3-Max 在 DashScope 的模型名（注意：不是 qwen3-max）
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    stream=True  # 关键：启用流式响应
)

# 处理流式响应
print("Assistant: ", end="", flush=True)
for chunk in completion:
    # 检查 chunk 是否包含内容
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

print("\n")  # 换行结束