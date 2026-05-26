import sys
from pathlib import Path
from moviepy import VideoFileClip


def video_to_audio(task_id: str):
    task_dir = Path("tasks") / task_id
    input_video = task_dir / "input" / "video.mp4"
    output_dir = task_dir / "audio"
    output_audio = output_dir / "audio.mp3"

    if not input_video.exists():
        raise FileNotFoundError(f"输入视频不存在: {input_video}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在处理: {input_video}")
    video = VideoFileClip(str(input_video))
    video.audio.write_audiofile(str(output_audio))
    video.close()
    print(f"音频已导出: {output_audio}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: uv run .\\src\\a_video_to_audio.py <task_id>")
        sys.exit(1)

    video_to_audio(sys.argv[1])
