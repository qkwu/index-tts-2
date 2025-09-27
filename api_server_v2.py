import os
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from pathlib import Path
import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import argparse
import json
import numpy as np
import soundfile as sf
import uuid
from fastapi import BackgroundTasks
import time
from concurrent.futures import ThreadPoolExecutor
import librosa
import torch
import torchaudio

from indextts.infer_v2 import IndexTTS2

tts = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("index-tts2-api")

# 目录配置
OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "outputs")
REFERENCE_DIR = os.environ.get("TTS_REFERENCE_DIR", "references")
MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "checkpoints")
TEMP_DIR = os.environ.get("TTS_TEMP_DIR", "temp_uploads")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 定义请求模型
class TTSRequest(BaseModel):
    text: str
    reference_id: str
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 1500
    emo_alpha: float = 1.0
    emo_vector: Optional[List[float]] = None
    use_emo_text: bool = False
    emo_text: Optional[str] = None
    use_random: bool = False
    interval_silence: int = 200
    max_text_tokens_per_segment: int = 120
    stream: bool = False

class TTSRequestV2(BaseModel):
    text: str
    spk_audio_path: str  # 说话人参考音频路径
    emo_audio_path: Optional[str] = None  # 情感参考音频路径，可选
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 1500
    emo_alpha: float = 1.0
    emo_vector: Optional[List[float]] = None
    use_emo_text: bool = False
    emo_text: Optional[str] = None
    use_random: bool = False
    interval_silence: int = 200
    max_text_tokens_per_segment: int = 120

# 定义响应模型
class TTSResponse(BaseModel):
    id: str
    audio_url: str
    duration: float
    text: str
    sampling_rate: int

class SimpleInferenceQueue:
    """简单的推理队列"""

    def __init__(self, max_workers=1):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.queue_size = 0
        self.lock = asyncio.Lock()

    async def submit(self, func, *args, **kwargs):
        """提交任务到队列"""
        async with self.lock:
            self.queue_size += 1
            position = self.queue_size

        logger.info(f"Task submitted to queue. Position: {position}")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, func, *args, **kwargs)
            return result
        finally:
            async with self.lock:
                self.queue_size -= 1

    def get_queue_size(self):
        """获取当前队列大小"""
        return self.queue_size

    def shutdown(self):
        """关闭队列"""
        self.executor.shutdown(wait=True)

# 全局队列实例
inference_queue = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts, inference_queue, args  # 添加 args

    # 初始化推理队列 (只允许1个并发任务)
    inference_queue = SimpleInferenceQueue(max_workers=1)

    # 模型加载代码
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    tts = IndexTTS2(
        model_dir=args.model_dir,
        cfg_path=cfg_path,
        use_fp16=args.use_fp16,
        use_deepspeed=args.use_deepspeed
    )

    # 说话人注册代码
    logger.info(f"正在从 '{REFERENCE_DIR}' 目录扫描并注册音色...")
    if not os.path.exists(REFERENCE_DIR):
        os.makedirs(REFERENCE_DIR)
        logger.warning(f"参考音频目录 '{REFERENCE_DIR}' 不存在，已自动创建。")

    speaker_count = 0
    for speaker_dir in Path(REFERENCE_DIR).iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            audio_files = [str(p) for p in speaker_dir.glob("*")
                           if p.suffix.lower() in ['.wav', '.mp3', '.flac']]

            if audio_files:
                spk_audio_path = audio_files[0]
                emo_audio_path = audio_files[1] if len(audio_files) > 1 else None
                tts.registry_speaker(speaker_id, spk_audio_path, emo_audio_path)
                speaker_count += 1

    if speaker_count == 0:
        logger.warning(f"警告: '{REFERENCE_DIR}' 中没有找到任何可用的音色。")

    logger.info("Application startup complete.")
    yield

    # 清理资源
    if inference_queue:
        inference_queue.shutdown()

app = FastAPI(lifespan=lifespan)

def convert_audio_format(audio_tuple):
    """将IndexTTS2返回的音频格式转换为numpy数组"""
    sampling_rate, wav_data = audio_tuple

    print(f"DEBUG: Input audio shape: {wav_data.shape}, dtype: {wav_data.dtype}")

    # 确保是numpy数组
    if not isinstance(wav_data, np.ndarray):
        wav_data = np.array(wav_data)

    # 处理维度转换
    if wav_data.ndim == 2:
        # 检查是 [channels, samples] 还是 [samples, channels]
        if wav_data.shape[0] == 2 and wav_data.shape[1] > wav_data.shape[0]:
            # [2, samples] - 立体声，转置后取平均
            wav_data = wav_data.mean(axis=0)
        elif wav_data.shape[1] == 2:
            # [samples, 2] - 立体声，取平均
            wav_data = wav_data.mean(axis=1)
        elif wav_data.shape[0] == 1:
            # [1, samples] - 单声道，展平
            wav_data = wav_data.flatten()
        elif wav_data.shape[1] == 1:
            # [samples, 1] - 单声道，展平
            wav_data = wav_data.flatten()
        else:
            # 如果不确定格式，假设第一维是通道数
            if wav_data.shape[0] < wav_data.shape[1]:
                wav_data = wav_data.mean(axis=0)  # [channels, samples]
            else:
                wav_data = wav_data.mean(axis=1)  # [samples, channels]

    # 确保是1维数组
    if wav_data.ndim != 1:
        raise ValueError(f"Cannot convert audio to 1D array, shape: {wav_data.shape}")

    # 数据类型转换
    if wav_data.dtype == np.int16:
        # 如果是int16，转换为float32范围[-1,1]
        wav_data = wav_data.astype(np.float32) / 32767.0
    elif wav_data.dtype == np.int32:
        # 如果是int32，转换为float32范围[-1,1]
        wav_data = wav_data.astype(np.float32) / 2147483647.0
    else:
        # 确保是float32
        wav_data = wav_data.astype(np.float32)

    # 限制在合理范围内
    wav_data = np.clip(wav_data, -1.0, 1.0)

    print(f"DEBUG: Output audio shape: {wav_data.shape}, dtype: {wav_data.dtype}, range: [{wav_data.min():.3f}, {wav_data.max():.3f}]")

    return sampling_rate, wav_data

@app.post("/tts_v2", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_v2(request: Request):
    """使用队列的TTS V2接口"""
    global tts, inference_queue
    try:
        data = await request.json()
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_audio_path = data.get("emo_audio_path")
        emo_alpha = data.get("emo_alpha", 1.0)
        emo_vector = data.get("emo_vector")
        use_emo_text = data.get("use_emo_text", False)
        emo_text = data.get("emo_text")
        use_random = data.get("use_random", False)
        interval_silence = data.get("interval_silence", 200)
        max_text_tokens_per_segment = data.get("max_text_tokens_per_segment", 120)

        generation_kwargs = {
            "temperature": data.get("temperature", 0.8),
            "top_p": data.get("top_p", 0.8),
            "top_k": data.get("top_k", 30),
            "repetition_penalty": data.get("repetition_penalty", 10.0),
            "max_mel_tokens": data.get("max_mel_tokens", 1500),
        }

        # 定义推理函数
        def inference_task():
            start_time = time.time()
            logger.info(f"开始处理V2推理任务: '{text[:30]}...'")

            result = tts.infer(
                spk_audio_prompt=spk_audio_path,
                text=text,
                output_path=None,
                emo_audio_prompt=emo_audio_path,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
                emo_text=emo_text,
                use_random=use_random,
                interval_silence=interval_silence,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                **generation_kwargs
            )

            inference_time = time.time() - start_time
            logger.info(f"V2推理任务完成，耗时: {inference_time:.2f}秒")
            return result

        # 提交到队列
        logger.info(f"提交V2推理任务到队列，当前队列长度: {inference_queue.get_queue_size()}")
        audio_result = await inference_queue.submit(inference_task)

        sr, wav = convert_audio_format(audio_result)

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV', subtype='PCM_16')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        logger.error(f"TTS V2 generation failed: {tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )

@app.post("/tts", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api(request: Request):
    """使用队列的TTS接口"""
    global tts, inference_queue
    try:
        data = await request.json()
        text = data["text"]
        character = data["character"]

        if character not in tts.speaker_dict:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "error": f"Speaker '{character}' not found in registry. Available speakers: {list(tts.speaker_dict.keys())}"
                }
            )

        # 准备参数
        emo_alpha = data.get("emo_alpha", 1.0)
        emo_vector = data.get("emo_vector")
        use_emo_text = data.get("use_emo_text", False)
        emo_text = data.get("emo_text")
        interval_silence = data.get("interval_silence", 200)
        max_text_tokens_per_segment = data.get("max_text_tokens_per_segment", 120)

        generation_kwargs = {
            "temperature": data.get("temperature", 0.8),
            "top_p": data.get("top_p", 0.8),
            "top_k": data.get("top_k", 30),
            "repetition_penalty": data.get("repetition_penalty", 10.0),
            "max_mel_tokens": data.get("max_mel_tokens", 1500),
        }

        # 定义推理函数
        def inference_task():
            start_time = time.time()
            logger.info(f"开始处理推理任务: '{text[:30]}...'")

            result = tts.infer_with_speaker_id(
                speaker_id=character,
                text=text,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
                emo_text=emo_text,
                interval_silence=interval_silence,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                **generation_kwargs
            )

            inference_time = time.time() - start_time
            logger.info(f"推理任务完成，耗时: {inference_time:.2f}秒")
            return result

        # 提交到队列并等待结果
        logger.info(f"提交推理任务到队列，当前队列长度: {inference_queue.get_queue_size()}")
        audio_result = await inference_queue.submit(inference_task)

        sr, wav = convert_audio_format(audio_result)

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV', subtype='PCM_16')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        logger.error(f"TTS generation failed: {tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )

# 添加队列状态查询接口
@app.get("/queue/status")
async def get_queue_status():
    """获取队列状态"""
    global inference_queue
    if inference_queue:
        return {
            "queue_size": inference_queue.get_queue_size(),
            "max_workers": inference_queue.max_workers,
            "status": "running"
        }
    return {
        "queue_size": 0,
        "max_workers": 0,
        "status": "not_initialized"
    }

# 兼容性接口 - 现在也使用队列！
@app.post("/v1/tts", response_model=TTSResponse, tags=["Compatibility Endpoints"])
async def compatible_generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """[兼容旧版] 异步生成TTS，返回包含音频URL的JSON响应 - 使用队列"""
    global tts, inference_queue
    task_id = str(uuid.uuid4())
    logger.info(f"Received compatible request {task_id} for speaker '{request.reference_id}'")

    if tts is None:
        raise HTTPException(status_code=503, detail="TTS model is not ready.")

    try:
        character = request.reference_id.split(',')[0].strip()

        if character not in tts.speaker_dict:
            available_speakers = list(tts.speaker_dict.keys())
            raise HTTPException(status_code=404, detail=f"Speaker '{character}' not found. Available speakers: {available_speakers}")

        # 生成参数
        generation_kwargs = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "max_mel_tokens": request.max_mel_tokens,
        }

        # 定义推理任务 - 使用队列！
        def inference_task():
            start_time = time.time()
            logger.info(f"开始处理兼容性推理任务: '{request.text[:30]}...'")

            result = tts.infer_with_speaker_id(
                speaker_id=character,
                text=request.text,
                emo_alpha=request.emo_alpha,
                emo_vector=request.emo_vector,
                use_emo_text=request.use_emo_text,
                emo_text=request.emo_text,
                use_random=request.use_random,
                interval_silence=request.interval_silence,
                max_text_tokens_per_segment=request.max_text_tokens_per_segment,
                **generation_kwargs
            )

            inference_time = time.time() - start_time
            logger.info(f"兼容性推理任务完成，耗时: {inference_time:.2f}秒")
            return result

        # 使用队列执行推理
        logger.info(f"提交兼容性推理任务到队列，当前队列长度: {inference_queue.get_queue_size()}")
        audio_result = await inference_queue.submit(inference_task)

        sr, wav = convert_audio_format(audio_result)

        # 保存音频文件
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
        sf.write(output_path, wav, sr, subtype='PCM_16')

        duration = len(wav) / sr

        response_data = TTSResponse(
            id=task_id,
            audio_url=f"/v1/audio/{task_id}",
            duration=round(duration, 2),
            text=request.text,
            sampling_rate=sr
        )

        # 清理文件的后台任务
        background_tasks.add_task(lambda: os.remove(output_path) if os.path.exists(output_path) else None)

        return response_data

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/tts_audio", tags=["Compatibility Endpoints"])
async def compatible_generate_and_return_tts_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    """[兼容旧版] 生成TTS并直接返回音频文件 - 使用队列"""
    response_data = await compatible_generate_tts(request, background_tasks)
    task_id = response_data.id
    file_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generated audio file not found.")
    return FileResponse(file_path, media_type="audio/wav", filename=f"{task_id}.wav")

@app.post("/v1/tts_v2", response_model=TTSResponse, tags=["V2 Endpoints"])
async def v2_generate_tts(request: TTSRequestV2, background_tasks: BackgroundTasks):
    """V2版本TTS接口，支持直接指定音频路径 - 使用队列"""
    global tts, inference_queue
    task_id = str(uuid.uuid4())
    logger.info(f"Received V2 request {task_id}")

    if tts is None:
        raise HTTPException(status_code=503, detail="TTS model is not ready.")

    try:
        # 生成参数
        generation_kwargs = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "max_mel_tokens": request.max_mel_tokens,
        }

        # 定义推理任务 - 使用队列！
        def inference_task():
            start_time = time.time()
            logger.info(f"开始处理V2兼容性推理任务: '{request.text[:30]}...'")

            result = tts.infer(
                spk_audio_prompt=request.spk_audio_path,
                text=request.text,
                output_path=None,
                emo_audio_prompt=request.emo_audio_path,
                emo_alpha=request.emo_alpha,
                emo_vector=request.emo_vector,
                use_emo_text=request.use_emo_text,
                emo_text=request.emo_text,
                use_random=request.use_random,
                interval_silence=request.interval_silence,
                max_text_tokens_per_segment=request.max_text_tokens_per_segment,
                **generation_kwargs
            )

            inference_time = time.time() - start_time
            logger.info(f"V2兼容性推理任务完成，耗时: {inference_time:.2f}秒")
            return result

        # 使用队列执行推理
        logger.info(f"提交V2兼容性推理任务到队列，当前队列长度: {inference_queue.get_queue_size()}")
        audio_result = await inference_queue.submit(inference_task)

        sr, wav = convert_audio_format(audio_result)

        # 保存音频文件
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
        sf.write(output_path, wav, sr, subtype='PCM_16')

        duration = len(wav) / sr

        response_data = TTSResponse(
            id=task_id,
            audio_url=f"/v1/audio/{task_id}",
            duration=round(duration, 2),
            text=request.text,
            sampling_rate=sr
        )

        # 清理文件的后台任务
        background_tasks.add_task(lambda: os.remove(output_path) if os.path.exists(output_path) else None)

        return response_data

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/audio/{audio_id}", tags=["Compatibility Endpoints"])
async def compatible_get_audio(audio_id: str):
    """[兼容旧版] 获取生成的音频文件"""
    file_path = os.path.join(OUTPUT_DIR, f"{audio_id}.wav")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found or has been cleaned up.")
    return FileResponse(file_path, media_type="audio/wav", filename=f"{audio_id}.wav")

@app.get("/v1/references", tags=["Compatibility Endpoints"])
async def compatible_list_references():
    """[兼容旧版] 列出所有可用的参考音频ID (说话人)"""
    if tts is None or not hasattr(tts, 'speaker_dict'):
        return {"references": []}
    references = [{"id": spk_id, "name": spk_id} for spk_id in tts.speaker_dict.keys()]
    return {"references": references}

@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "healthy", "version": "index-tts2 2.0"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 保持新仓库的命令行参数风格，同时可以被环境变量覆盖
    default_model_dir = os.environ.get("TTS_MODEL_DIR", "checkpoints")
    default_port = int(os.environ.get("SERVICE_PORT", 11997))

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--model_dir", type=str, default=default_model_dir)
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed")
    args = parser.parse_args()

    # 简单的启动前检查
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        logger.error("Please specify a valid path using --model_dir or the TTS_MODEL_DIR environment variable.")
    else:
        logger.info(f"Starting IndexTTS2 API on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)