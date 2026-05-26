import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline


def audio_to_info(task_id: str):
    task_dir = Path("tasks") / task_id
    input_audio = task_dir / "audio" / "audio.mp3"
    output_dir = task_dir / "speaker_diarization"
    output_info = output_dir / "output.json"

    if not input_audio.exists():
        raise FileNotFoundError(f"音频文件不存在: {input_audio}")

    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("请设置环境变量 HF_TOKEN（HuggingFace access token）")

    print("正在加载 pyannote 管线...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: CUDA 不可用，将使用 CPU 运行")

    print(f"正在加载音频: {input_audio}")
    data, sample_rate = sf.read(str(input_audio))
    # soundfile 返回 (samples, channels)，转成 (channels, samples) 的 tensor
    waveform = torch.from_numpy(data).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T

    print("正在做说话人分离...")
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # pyannote 4.0 返回 DiarizeOutput，通过 .speaker_diarization 访问 Annotation
    annotation = output.speaker_diarization

    # 提取所有说话片段
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
        })

    # 从 diarization 结果中计算重叠区域（同一时刻有多个说话人的区间）
    print("正在计算重叠区域...")
    from pyannote.core import Timeline

    overlaps = []
    tracks = list(annota tion.itertracks(yield_label=True))
    for i, (turn_a, _, speaker_a) in enumerate(tracks):
        for turn_b, _, speaker_b in tracks[i + 1:]:
            if speaker_a == speaker_b:
                continue
            overlap_start = max(turn_a.start, turn_b.start)
            overlap_end = min(turn_a.end, turn_b.end)
            if overlap_start < overlap_end:
                overlaps.append({
                    "start": round(overlap_start, 3),
                    "end": round(overlap_end, 3),
                    "duration": round(overlap_end - overlap_start, 3),
                    "speakers": sorted({speaker_a, speaker_b}),
                })

    # 合并重叠的重叠区间
    overlaps.sort(key=lambda x: x["start"])
    merged = []
    for ov in overlaps:
        if merged and ov["start"] <= merged[-1]["end"]:
            prev = merged[-1]
            new_end = max(prev["end"], ov["end"])
            prev["end"] = new_end
            prev["duration"] = round(new_end - prev["start"], 3)
            prev["speakers"] = sorted(set(prev["speakers"]) | set(ov["speakers"]))
        else:
            merged.append(ov)
    overlaps = merged

    # 汇总说话人统计
    speaker_stats = {}
    for seg in segments:
        sp = seg["speaker"]
        if sp not in speaker_stats:
            speaker_stats[sp] = {"total_duration": 0, "segment_count": 0}
        speaker_stats[sp]["total_duration"] = round(
            speaker_stats[sp]["total_duration"] + seg["duration"], 3
        )
        speaker_stats[sp]["segment_count"] += 1

    result = {
        "task_id": task_id,
        "source": str(input_audio),
        "speakers": speaker_stats,
        "segments": segments,
        "overlaps": overlaps,
        "overlap_count": len(overlaps),
        "segment_count": len(segments),
    }

    output_info.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"分析结果已导出: {output_info}")
    print(f"  说话人数: {len(speaker_stats)}")
    print(f"  片段总数: {len(segments)}")
    print(f"  重叠区域: {len(overlaps)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: uv run .\\src\\b_speaker_diarization.py <task_id>")
        sys.exit(1)

    audio_to_info(sys.argv[1])
