import json
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path


def find_primary_speaker(asr_start, asr_end, spk_segments, spk_ends):
    """找与 ASR 片段重叠时长最长的说话人"""
    # 二分查找：找第一个 end > asr_start 的 speaker segment
    lo = bisect_right(spk_ends, asr_start)
    duration_by_speaker = {}
    for i in range(lo, len(spk_segments)):
        seg = spk_segments[i]
        if seg["start"] >= asr_end:
            break
        overlap = min(asr_end, seg["end"]) - max(asr_start, seg["start"])
        if overlap > 0:
            sp = seg["speaker"]
            duration_by_speaker[sp] = duration_by_speaker.get(sp, 0) + overlap
    if not duration_by_speaker:
        return None
    return max(duration_by_speaker, key=duration_by_speaker.get)


def find_overlapping_speakers(asr_start, asr_end, primary, overlaps, ovl_ends):
    """找与 ASR 片段时间重叠的 overlap 记录中，除主说话人外的其他说话人"""
    lo = bisect_right(ovl_ends, asr_start)
    speakers = set()
    for i in range(lo, len(overlaps)):
        ovl = overlaps[i]
        if ovl["start"] >= asr_end:
            break
        if min(asr_end, ovl["end"]) > max(asr_start, ovl["start"]):
            speakers.update(ovl["speakers"])
    speakers.discard(primary)
    return sorted(speakers)


def find_crowd_info(asr_start, asr_end, windows, win_ends):
    """检查与 ASR 片段重叠的 crowd 窗口的 is_crowd"""
    lo = bisect_right(win_ends, asr_start)
    for i in range(lo, len(windows)):
        w = windows[i]
        if w["start"] >= asr_end:
            break
        if min(asr_end, w["end"]) > max(asr_start, w["start"]):
            if w["is_crowd"]:
                return True
    return False


def merge_crowd_ranges(windows):
    """将连续的 is_crowd=True 窗口合并为连续嘈杂段"""
    crowd_windows = [w for w in windows if w["is_crowd"]]
    if not crowd_windows:
        return []

    ranges = []
    cur_start = crowd_windows[0]["start"]
    cur_end = crowd_windows[0]["end"]

    for w in crowd_windows[1:]:
        if w["start"] <= cur_end:
            cur_end = max(cur_end, w["end"])
        else:
            ranges.append((cur_start, cur_end))
            cur_start = w["start"]
            cur_end = w["end"]

    ranges.append((cur_start, cur_end))
    return ranges


def dedup_asr_segments(segments, min_duration=0.05):
    """过滤 Whisper 幻觉重复：合并连续相同文本，丢弃极短片段"""
    if not segments:
        return segments

    result = []
    prev = segments[0]

    for seg in segments[1:]:
        if seg["text"] == prev["text"]:
            # 连续重复 → 扩展前一个片段的 end，跳过当前
            prev = {**prev, "end": seg["end"]}
        else:
            if prev["end"] - prev["start"] >= min_duration:
                result.append(prev)
            prev = seg

    if prev["end"] - prev["start"] >= min_duration:
        result.append(prev)

    return result


def merge_results(task_id: str):
    task_dir = Path("tasks") / task_id

    # 读取三个输入
    with open(task_dir / "speaker_diarization" / "output.json", encoding="utf-8") as f:
        b_data = json.load(f)
    with open(task_dir / "automatic_speech_recognition" / "output.json", encoding="utf-8") as f:
        c_data = json.load(f)
    with open(task_dir / "crowd_detect" / "output.json", encoding="utf-8") as f:
        d_data = json.load(f)

    spk_segments = b_data["segments"]
    overlaps = b_data["overlaps"]
    raw_asr = c_data["segments"]
    asr_segments = dedup_asr_segments(raw_asr)
    print(f"ASR 去重: {len(raw_asr)} → {len(asr_segments)} 段 (过滤 {len(raw_asr) - len(asr_segments)} 段幻觉重复)")
    windows = d_data["windows"]

    # 预计算 end 数组用于二分查找
    spk_ends = [s["end"] for s in spk_segments]
    ovl_ends = [o["end"] for o in overlaps]
    win_ends = [w["end"] for w in windows]

    print(f"正在合并: ASR {len(asr_segments)} 段, 说话人 {len(spk_segments)} 段, "
          f"重叠 {len(overlaps)} 段, 窗口 {len(windows)} 个")

    # 处理 ASR segments
    merged = []
    for seg in asr_segments:
        start, end = seg["start"], seg["end"]
        primary = find_primary_speaker(start, end, spk_segments, spk_ends)
        overlapping = find_overlapping_speakers(start, end, primary, overlaps, ovl_ends)
        is_crowd = find_crowd_info(start, end, windows, win_ends)

        merged.append({
            "type": "speech",
            "text": seg["text"],
            "start": start,
            "end": end,
            "speaker": primary,
            "overlapping_speakers": overlapping,
            "overlap_count": len(overlapping),
            "is_crowd": is_crowd,
        })

    # 合并连续嘈杂窗口为 crowd 段
    crowd_ranges = merge_crowd_ranges(windows)
    print(f"连续嘈杂段: {len(crowd_ranges)} 个")

    for start, end in crowd_ranges:
        merged.append({
            "type": "crowd",
            "text": "",
            "start": start,
            "end": end,
            "speaker": None,
            "overlapping_speakers": [],
            "overlap_count": 0,
            "is_crowd": True,
        })

    # 按时间排序
    merged.sort(key=lambda x: x["start"])

    output_dir = task_dir / "merge"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.json"

    result = {
        "task_id": task_id,
        "segments": merged,
        "segment_count": len(merged),
    }

    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"合并结果已导出: {output_file}")
    print(f"  总片段数: {len(merged)} (speech: {len(asr_segments)}, crowd: {len(crowd_ranges)})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: uv run src/e_merge_results.py <task_id>")
        sys.exit(1)

    merge_results(sys.argv[1])
