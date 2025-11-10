import os
from openai import OpenAI
from dotenv import load_dotenv
import sys

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 客户端（使用 DashScope 兼容模式）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def main():
    print("="*50)
    print("Qwen3-Max 交互式对话 (流式输出)")
    print("输入 'exit' 或 'quit' 退出对话")
    print("="*50)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\nYou: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("Goodbye! 👋")
                break
                
            # 创建流式对话请求
            print("\nAssistant: ", end="", flush=True)
            completion = client.chat.completions.create(
                model="qwen3-max",  # 注意：正确模型名是 qwen-max (不是 qwen3-max)
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input},
                ],
                stream=True
            )
            
            # 处理流式响应
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            
            print("\n")  # 换行结束本次回答
            
        except KeyboardInterrupt:
            print("\n\nManual exit. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please check your API key and network connection.")

if __name__ == "__main__":
    main()