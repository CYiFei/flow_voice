import os
import asyncio
import logging
import wave
from dotenv import load_dotenv  # 👈 新增：加载环境变量
from tts_realtime_client import TTSRealtimeClient, SessionMode
import pyaudio
import time

# ======================
# 安全加载 API Key
# ======================
load_dotenv()  # 👈 从 .env 文件加载环境变量
API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 👈 从环境变量获取
URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime"

if not API_KEY:
    raise ValueError("❌ DASHSCOPE_API_KEY environment variable is not set. "
                     "Please create a .env file with your API key.")

# ======================
# 其余代码保持不变（以下为完整代码）
# ======================
# 收集音频数据
_audio_chunks = []
# 实时播放相关
_AUDIO_SAMPLE_RATE = 24000
_audio_pyaudio = pyaudio.PyAudio()
_audio_stream = None

# 时间统计变量
_text_start_time = None      # 第一个文本发送时间
_first_audio_time = None     # 首次收到音频时间
_first_audio_logged = False  # 是否已记录首包延迟

def _audio_callback(audio_bytes: bytes):
    """TTSRealtimeClient 音频回调: 实时播放并缓存"""
    global _audio_stream, _first_audio_time, _first_audio_logged, _text_start_time
    if _audio_stream is not None:
        try:
            _audio_stream.write(audio_bytes)
        except Exception as exc:
            logging.error(f"PyAudio playback error: {exc}")
    _audio_chunks.append(audio_bytes)
    logging.info(f"Received audio chunk: {len(audio_bytes)} bytes")

    # 记录首次音频到达时间并计算首包延迟
    if not _first_audio_logged and _text_start_time is not None:
        _first_audio_time = time.time()
        latency = (_first_audio_time - _text_start_time) * 1000  # 毫秒
        logging.info(f"[METRIC] Time to first audio: {latency:.2f} ms")
        _first_audio_logged = True

def _save_audio_to_file(filename: str = "output.wav", sample_rate: int = 24000) -> bool:
    if not _audio_chunks:
        logging.warning("No audio data to save")
        return False
    try:
        audio_data = b"".join(_audio_chunks)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        logging.info(f"Audio saved to: {filename}")
        return True
    except Exception as exc:
        logging.error(f"Failed to save audio: {exc}")
        return False

async def _produce_text(client: TTSRealtimeClient):
    global _text_start_time
    text_fragments = [
        "阿",
        "里",
        "云",
        "的",
        "大",
        "模",
        "型",
        "服",
        "务",
        "平",
        "台",
        "百",
        "炼,",
        "是",
        "一站式的大模型开",
        "发及应用构建平台。",
        "不论是开发者还是业务人员，都能深入参与",
        "大模型应用的设计和构建。"
        "您可以通过简",
        "单的界面操作，在5分钟内开发出一款大模型应用，",
        "或在几小时内训练出一个专",
        "属模型，从而将更多精力专注于应用创新。"
        "模型训练和应用构",
        "建，只需几步，即可实现高效、精准的模型应用。"
    ]

    logging.info("Sending text fragments…")
    _text_start_time = time.time()
    for i, text in enumerate(text_fragments):
        logging.info(f"Sending fragment: {text}")
        await client.append_text(text)
        await asyncio.sleep(0.1)

    await asyncio.sleep(1.0)
    await client.finish_session()

async def _run_demo():
    global _audio_stream, _text_start_time, _first_audio_logged
    _audio_stream = _audio_pyaudio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=_AUDIO_SAMPLE_RATE,
        output=True,
        frames_per_buffer=1024
    )

    client = TTSRealtimeClient(
        base_url=URL,
        api_key=API_KEY,
        voice="Cherry",
        language_type="Chinese",
        mode=SessionMode.SERVER_COMMIT,
        audio_callback=_audio_callback
    )

    session_start = time.time()
    await client.connect()

    consumer_task = asyncio.create_task(client.handle_messages())
    producer_task = asyncio.create_task(_produce_text(client))

    await producer_task
    await asyncio.sleep(5)

    await client.close()
    consumer_task.cancel()

    if _audio_stream is not None:
        _audio_stream.stop_stream()
        _audio_stream.close()
    _audio_pyaudio.terminate()

    total_time = (time.time() - session_start) * 1000  # 毫秒
    logging.info(f"[METRIC] Total session time: {total_time:.2f} ms")

    if not _first_audio_logged and _text_start_time is not None:
        logging.warning("[METRIC] No audio received at all!")

    os.makedirs("outputs", exist_ok=True)
    _save_audio_to_file(os.path.join("outputs", "qwen_tts_output.wav"))

    # 重置全局状态
    global _audio_chunks
    _audio_chunks = []
    _first_audio_logged = False
    _text_start_time = None

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("🚀 Starting QwenTTS Realtime Client demo with secure API key loading...")
    asyncio.run(_run_demo())

if __name__ == "__main__":
    main()