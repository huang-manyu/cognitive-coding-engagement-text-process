import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


GPUS = [0, 1, 2, 3, 4, 5]
CHUNK_SEC = 30
SAMPLE_RATE = 16000
MODEL_ID = "/data/disk1/guohaoran/model/whisper-large-v3"


def load_model_on_gpu(gpu_id):
    device = f"cuda:{gpu_id}"
    dtype = torch.float16
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor, device, dtype


def transcribe_chunk(args):
    """转录单个 chunk，返回 (chunk_idx, segments)"""
    chunk_idx, audio_chunk, model, processor, device, dtype = args
    inputs = processor(audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=dtype)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="chinese",
            task="transcribe",
            return_timestamps=True,
        )

    token_ids = predicted_ids[0].tolist()
    text = processor.tokenizer.decode(token_ids, skip_special_tokens=False, decode_with_timestamps=True)
    segments = parse_timestamp_text(text, chunk_idx * CHUNK_SEC)
    return chunk_idx, segments


def parse_timestamp_text(text: str, time_offset: float):
    """解析 Whisper 带时间戳的输出，如 <|0.00|>你好<|0.50|>"""
    import re
    segments = []
    for m in re.finditer(r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>", text):
        start = float(m.group(1)) + time_offset
        content = m.group(2).strip()
        end = float(m.group(3)) + time_offset
        if content:
            segments.append({
                "text": content,
                "start": round(start, 3),
                "end": round(end, 3),
            })
    return segments


def audio_transcribe(task_id: str):
    task_dir = Path("tasks") / task_id
    input_audio = task_dir / "audio" / "audio.mp3"
    output_dir = task_dir / "automatic_speech_recognition"
    output_file = output_dir / "output.json"

    if not input_audio.exists():
        raise FileNotFoundError(f"音频文件不存在: {input_audio}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载音频并切分
    print(f"正在加载音频: {input_audio}")
    audio, _ = librosa.load(str(input_audio), sr=SAMPLE_RATE)
    chunk_samples = CHUNK_SEC * SAMPLE_RATE

    audio_chunks = []
    for i in range(0, len(audio), chunk_samples):
        audio_chunks.append(audio[i:i + chunk_samples])
    print(f"音频切分为 {len(audio_chunks)} 个 {CHUNK_SEC}s chunk，使用 GPU {GPUS}")

    # 每张 GPU 加载一个模型
    print("正在加载模型到各 GPU...")
    gpu_resources = []
    for gpu_id in GPUS:
        model, processor, device, dtype = load_model_on_gpu(gpu_id)
        gpu_resources.append((model, processor, device, dtype))
        print(f"  GPU {gpu_id} 就绪")

    # 轮询分配 chunk 到 GPU
    tasks = []
    for i, chunk in enumerate(audio_chunks):
        res = gpu_resources[i % len(gpu_resources)]
        tasks.append((i, chunk, *res))

    # 多线程并行转录
    print("正在转录...")
    all_segments = [None] * len(audio_chunks)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = {executor.submit(transcribe_chunk, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            chunk_idx, segs = future.result()
            all_segments[chunk_idx] = segs
            done += 1
            if done % 5 == 0 or done == len(tasks):
                print(f"  进度: {done}/{len(tasks)}")

    # 合并所有 chunk 结果
    segments = []
    for segs in all_segments:
        if segs:
            segments.extend(segs)
    segments.sort(key=lambda x: x["start"])
    print(f"Whisper 共输出 {len(segments)} 个片段")

    output_data = {
        "task_id": task_id,
        "source_audio": str(input_audio),
        "model": MODEL_ID,
        "segments": segments,
        "segment_count": len(segments),
    }

    output_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"转录结果已导出: {output_file}，共 {len(segments)} 个片段")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: uv run src/c_automatic_speech_recognition.py <task_id>")
        sys.exit(1)

    audio_transcribe(sys.argv[1])
