# qwen_tts_integration.py
import os
import asyncio
import logging
import time
import sys
from dotenv import load_dotenv
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
import pyaudio
import wave
import threading
from queue import Queue

from tts_realtime_client import TTSRealtimeClient, SessionMode

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")
TTS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime"

if not API_KEY:
    raise ValueError("❌ DASHSCOPE_API_KEY environment variable is not set. Please create a .env file with your API key.")

# 音频播放参数
AUDIO_SAMPLE_RATE = 24000
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_BUFFER_SIZE = 2048

# 全局变量
audio_chunks = []
audio_pyaudio = None
audio_stream = None
text_queue = Queue()
is_playing = False

# 时间统计变量
text_start_time = None
first_token_time = None  # 新增：记录第一个token生成时间
last_token_time = None  # 新增：记录上一个token的时间
first_audio_time = None
first_audio_logged = False

def audio_callback(audio_bytes: bytes):
    """TTS音频回调函数：实时播放并缓存音频数据"""
    print(f"DEBUG: Audio callback triggered with {len(audio_bytes)} bytes")
    global audio_stream, first_audio_time, first_audio_logged, text_start_time, first_token_time
    
    # 缓存音频数据（无论是否能播放）
    audio_chunks.append(audio_bytes)
    logging.info(f"Received audio chunk: {len(audio_bytes)} bytes")

    # 尝试播放音频
    if audio_stream is not None:
        try:
            audio_stream.write(audio_bytes)
        except Exception as exc:
            logging.error(f"PyAudio playback error: {exc}")
            # 即使播放失败也要继续缓存数据

    # 记录首次音频到达时间并计算首包延迟
    if not first_audio_logged and text_start_time is not None:
        first_audio_time = time.time()
        latency = (first_audio_time - text_start_time) * 1000  # 毫秒
        logging.info(f"[METRIC] Time to first audio: {latency:.2f} ms")
        
        # 如果已经记录了第一个token的时间，则计算token到音频播放的时间间隔
        if first_token_time is not None:
            token_to_audio_latency = (first_audio_time - first_token_time) * 1000  # 毫秒
            logging.info(f"[METRIC] Time from first token to first audio: {token_to_audio_latency:.2f} ms")
        
        first_audio_logged = True

def save_audio_to_file(filename: str = "qwen_tts_output.wav", sample_rate: int = 24000) -> bool:
    """保存音频数据到文件"""
    if not audio_chunks:
        logging.warning("No audio data to save")
        return False
    try:
        audio_data = b"".join(audio_chunks)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(AUDIO_CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        logging.info(f"Audio saved to: {filename}")
        return True
    except Exception as exc:
        logging.error(f"Failed to save audio: {exc}")
        return False




async def generate_text(prompt: str):
    """使用Qwen3-Max模型流式生成文本"""
    global text_start_time, first_token_time, last_token_time

    # 显式设置API密钥
    import dashscope
    dashscope.api_key = API_KEY
    
    messages = [
        {'role': Role.SYSTEM, 'content': '你是一个有帮助的助手'},
        {'role': Role.USER, 'content': prompt}
    ]
    
    text_start_time = time.time()
    full_response = ""
    
    try:
        response = Generation.call(
            model="qwen3-max",
            messages=messages,
            result_format='message',
            stream=True,
            incremental_output=True
        )
        
        print("🤖 AI助手: ", end="", flush=True)
        
        for chunk in response:
            if chunk.status_code == 200:
                content = chunk.output.choices[0].message.content
                if content:
                    current_time = time.time()
                    
                    # 记录第一个token的时间
                    if first_token_time is None:
                        first_token_time = current_time
                        latency = (first_token_time - text_start_time) * 1000  # 毫秒
                        logging.info(f"[METRIC] Time to first token: {latency:.2f} ms")
                    # 记录后续token的时间间隔
                    elif last_token_time is not None:
                        token_interval = (current_time - last_token_time) * 1000  # 毫秒
                        logging.info(f"[METRIC] Time since last token: {token_interval:.2f} ms")
                    
                    # 更新上一个token的时间
                    last_token_time = current_time
                    
                    full_response += content
                    # 实时打印输出
                    print(content, end="", flush=True)
                    # 将生成的文本放入队列供TTS消费
                    text_queue.put(content)
                    # 添加小延迟以确保TTS能及时处理
                    await asyncio.sleep(0.01)
            else:
                print(f"\n❌ 文本生成错误: {chunk.code} {chunk.message}")
                break
                
        print()  # 换行
        
        # 确保所有文本都已处理完后再发送结束标记
        await asyncio.sleep(0.2)
        # 文本生成完成后发送结束标记
        text_queue.put(None)
        print("📝 文本生成完成")
        logging.info("Text generation completed")
        
    except Exception as e:
        print(f"\n❌ 文本生成异常: {e}")
        logging.error(f"Text generation error: {e}")
        text_queue.put(None)  # 确保发送结束标记


# 替换 text_to_speech_producer 函数中的代码
async def text_to_speech_producer(client: TTSRealtimeClient):
    """从文本队列中获取文本并发送给TTS客户端"""
    print("🎧 TTS生产者已启动")
    text_count = 0
    
    while True:
        try:
            # 使用非阻塞方式获取队列内容
            text = text_queue.get(timeout=30)  # 增加超时时间
            if text is None:
                # 文本生成完成，结束会话
                print("📝 文本流结束，正在结束TTS会话...")
                await client.finish_session()
                print("✅ TTS会话已结束")
                break
            else:
                text_count += 1
                print(f"📤 发送第{text_count}段文本到TTS: {text[:50]}{'...' if len(text) > 50 else ''}")
                await client.append_text(text)
                # 添加小延迟以避免发送过于频繁
                await asyncio.sleep(0.05)
        except Exception as e:
            print(f"⚠️ TTS生产者异常: {e}")
            # 检查是否是连接问题，如果是则退出循环
            if "keepalive ping timeout" in str(e) or "connection closed" in str(e).lower():
                break
            # 不立即退出，继续等待可能的文本
            await asyncio.sleep(0.1)
            # 如果长时间没有新文本，可以考虑退出
            continue
    
    print(f"🎧 TTS生产者已完成，共处理 {text_count} 段文本")
    
    
async def run_integration_demo(prompt: str = None):
    """运行集成演示：文本生成 + TTS"""
    global audio_stream, text_start_time, first_audio_logged, audio_chunks, audio_pyaudio
    # 重置全局状态
    audio_chunks = []
    first_audio_logged = False
    text_start_time = None
    
    # 重新初始化 PyAudio
    if audio_pyaudio is not None:
        audio_pyaudio.terminate()
    audio_pyaudio = pyaudio.PyAudio()
    
    try:
        # 初始化音频流
        try:
            audio_stream = audio_pyaudio.open(
                format=AUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SAMPLE_RATE,
                output=True,
                frames_per_buffer=AUDIO_BUFFER_SIZE
            )
        except Exception as e:
            print(f"⚠️ 音频设备初始化失败: {e}")
            # 可以选择继续运行但不播放音频，或使用虚拟设备
            audio_stream = None

        # 初始化TTS客户端
        tts_client = TTSRealtimeClient(
            base_url=TTS_URL,
            api_key=API_KEY,
            voice="Cherry",
            language_type="Chinese",
            mode=SessionMode.SERVER_COMMIT,
            audio_callback=audio_callback
        )

        session_start = time.time()

        # 连接到TTS服务
        print("🔌 正在连接到TTS服务...")
        await tts_client.connect()
        print("✅ TTS服务连接成功")

        # 启动消息处理任务
        consumer_task = asyncio.create_task(tts_client.handle_messages())

        # 稍微等待确保连接建立
        await asyncio.sleep(0.1)

        # 获取用户输入
        if prompt is None:
            prompt = input("💬 请输入您的问题: ")

        print(f"🤔 正在处理问题: {prompt}")

        # 启动文本生成和TTS任务
        text_generation_task = asyncio.create_task(generate_text(prompt))
        tts_producer_task = asyncio.create_task(text_to_speech_producer(tts_client))

        # 等待所有任务完成
        await asyncio.gather(text_generation_task, tts_producer_task, return_exceptions=True)

        # 等待一段时间确保所有音频播放完毕
        print("⏳ 等待音频播放完成...")
        await asyncio.sleep(5)
    
    except Exception as e:
        print(f"❌ 运行时错误: {e}")
    finally:
        # 确保资源清理
        if 'tts_client' in locals():
            await tts_client.close()
        if 'consumer_task' in locals():
            consumer_task.cancel()

        # 清理音频资源
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
        if audio_pyaudio is not None:
            audio_pyaudio.terminate()

        total_time = (time.time() - session_start) * 1000  # 毫秒
        logging.info(f"[METRIC] Total session time: {total_time:.2f} ms")

        if not first_audio_logged and text_start_time is not None:
            logging.warning("[METRIC] No audio received at all!")

        # 保存音频文件
        os.makedirs("outputs", exist_ok=True)
        save_audio_to_file(os.path.join("outputs", "qwen_tts_integration_output.wav"))

def interactive_mode():
    """交互式模式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("🚀 启动 Qwen 文本生成 + TTS 交互式演示...")
    print("输入 'quit' 或 'exit' 退出程序\n")
    
    while True:
        try:
            prompt = input("💬 请输入您的问题: ").strip()
            if prompt.lower() in ['quit', 'exit']:
                print("👋 再见!")
                break
            
            if prompt:
                asyncio.run(run_integration_demo(prompt))
                print("-" * 50)  # 分隔线
            else:
                print("⚠️  请输入有效问题")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

def main():
    # 配置日志（减少干扰）
    logging.basicConfig(
        level=logging.INFO,  # 只显示警告及以上级别日志
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if len(sys.argv) > 1:
        # 命令行参数模式
        prompt = " ".join(sys.argv[1:])
        logging.getLogger().setLevel(logging.INFO)  # 恢复详细日志
        asyncio.run(run_integration_demo(prompt))
    else:
        # 交互式模式
        interactive_mode()

if __name__ == "__main__":
    main()