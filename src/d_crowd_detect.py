import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch


# 嘈杂相关标签索引（从 class_labels_indices.csv 确认）
CROWD_LABELS = {68, 69, 70}  # Chatter, Crowd, Hubbub

SAMPLE_RATE = 32000
WINDOW_SEC = 2
HOP_SEC = 1
DEFAULT_THRESHOLD = 0.3
DEFAULT_CHECKPOINT = "/data/disk1/guohaoran/model/panns/Cnn14_mAP=0.431.pth"


def crowd_detect(task_id: str):
    task_dir = Path("tasks") / task_id
    input_audio = task_dir / "audio" / "audio.mp3"
    output_dir = task_dir / "crowd_detect"
    output_file = output_dir / "output.json"

    if not input_audio.exists():
        raise FileNotFoundError(f"音频文件不存在: {input_audio}")

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = os.environ.get("PANNS_CHECKPOINT", DEFAULT_CHECKPOINT)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"PANNs 模型文件不存在: {checkpoint_path}")

    threshold = float(os.environ.get("CROWD_THRESHOLD", DEFAULT_THRESHOLD))

    # 加载模型
    print(f"正在加载 PANNs 模型: {checkpoint_path}")
    from panns_inference import AudioTagging, labels

    at = AudioTagging(checkpoint_path=checkpoint_path, device="cuda")

    # 加载音频
    print(f"正在加载音频: {input_audio}")
    waveform, _ = librosa.load(str(input_audio), sr=SAMPLE_RATE, mono=True)
    total_duration = len(waveform) / SAMPLE_RATE
    print(f"音频时长: {total_duration:.1f} 秒")

    # 滑动窗口切片
    window_samples = WINDOW_SEC * SAMPLE_RATE
    hop_samples = HOP_SEC * SAMPLE_RATE

    print(f"正在分析音频 (窗口={WINDOW_SEC}s, 步长={HOP_SEC}s, 阈值={threshold})...")

    windows = []
    crowd_count = 0
    offset = 0

    while offset + window_samples <= len(waveform):
        chunk = waveform[offset : offset + window_samples]
        audio_input = chunk[np.newaxis, :]  # (1, samples)

        clipwise_output, _ = at.inference(audio_input)
        probs = clipwise_output[0]  # (527,)

        start_sec = offset / SAMPLE_RATE
        end_sec = start_sec + WINDOW_SEC

        # 所有 527 个类别的概率
        all_probs = {labels[i]: round(float(probs[i]), 4) for i in range(len(labels))}

        # 超过阈值的标签
        high_conf_tags = {labels[i]: round(float(probs[i]), 4)
                          for i in range(len(labels)) if probs[i] >= threshold}

        # 判断是否嘈杂
        crowd_probs = [probs[i] for i in CROWD_LABELS]
        is_crowd = bool(max(crowd_probs) >= threshold)
        if is_crowd:
            crowd_count += 1

        windows.append({
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "all_probabilities": all_probs,
            "high_confidence_tags": high_conf_tags,
            "is_crowd": is_crowd,
        })

        offset += hop_samples

    print(f"分析完成: {len(windows)} 个窗口，{crowd_count} 个嘈杂窗口")

    result = {
        "task_id": task_id,
        "source": str(input_audio),
        "model": "Cnn14_mAP=0.431",
        "threshold": threshold,
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "crowd_label_indices": list(CROWD_LABELS),
        "total_windows": len(windows),
        "crowd_windows": crowd_count,
        "windows": windows,
    }

    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"检测结果已导出: {output_file}")
    print(f"  总窗口数: {len(windows)}")
    print(f"  嘈杂窗口: {crowd_count}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: uv run src/d_crowd_detect.py <task_id>")
        sys.exit(1)

    crowd_detect(sys.argv[1])
