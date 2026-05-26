# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cognitive coding engagement analysis — uses pyannote.audio for speaker diarization and moviepy for video/audio processing. Python 3.13+.

## Package Manager

This project uses **uv**. Do not use pip directly.

```bash
# Install dependencies
uv sync

# Add a dependency
uv add <package>

# Run the script
uv run .\src\a_video_to_audio.py
```

## Key Dependencies

- **pyannote.audio** — speaker diarization and audio analysis pipelines (requires HuggingFace token for model access)
- **moviepy** — video/audio file manipulation and processing
- **torch + torchaudio** — PyTorch with CUDA support (cu130 index for GPU acceleration)
- **ffmpeg** — located at `c:/ffmpeg/bin` (shared version), used for audio/video processing
- **panns-inference** — pretrained audio neural networks for sound event detection (AudioSet 527 classes)

## Tasks Directory Structure

`tasks/` 存放输入视频和处理产物。每个任务一个子目录，目录名即任务 ID。

任务 ID 格式: `YYYYMMDD.HHmmss.N` — 日期 + 时间 + 自增序号（同一秒内多个任务时递增）。例如 `20260304.140500.0`。

```
tasks/
  20260304.140500.0/
    input/
      video.mp4        # 固定文件名，原始输入视频
    audio/
      audio.mp3        # a_video_to_audio 产物
    speaker_diarization/
      output.json      # b_speaker_diarization 产物（说话人、时间段、重叠检测）
    automatic_speech_recognition/
      output.json      # c_automatic_speech_recognition 产物（每个片段的文本转录）
    crowd_detect/
      output.json      # d_crowd_detect 产物（嘈杂讨论时间范围）
    merge/
      output.json      # e_merge_results 产物（合并 b/c/d 的最终时间线）
```

## Processing Pipeline

`src/` 下的脚本按字母前缀排序，表示处理顺序。每个脚本接收任务 ID 作为命令行参数。

- `a_video_to_audio.py` — 从 input/video.mp4 提取音频到 audio/audio.mp3（moviepy）
- `b_speaker_diarization.py` — 对 audio/audio.mp3 做说话人分离，输出 speaker_diarization/output.json（pyannote.audio）
- `c_automatic_speech_recognition.py` — 对 audio/audio.mp3 做语音识别，输出 automatic_speech_recognition/output.json（Whisper）
- `d_crowd_detect.py` — 对 audio/audio.mp3 检测嘈杂讨论区间，输出 crowd_detect/output.json（PANNs CNN14）
- `e_merge_results.py` — 合并 b/c/d 的结果，按时间线输出 merge/output.json（纯 Python，无需 GPU）

## Remote Execution

通过 `rc` 命令（定义在 `~/.bashrc`）将项目同步到远程 GPU 服务器并执行命令。

- 远程服务器: `guohaoran@10.176.56.244`
- 远程目录: `/data/disk1/guohaoran/cognitive-coding-engagement-analysis`
- 同步工具: rsync（自动排除 `.git/`、`__pycache__/`、`.venv/`、`tasks/` 等）

```bash
# 同步项目并 SSH 进入远程目录
rc

# 同步项目并在远程执行命令
rc "uv run src/a_video_to_audio.py 20260304.140500.0"
```

`tasks/` 目录不会被 rsync 同步（含大文件），需要手动传输：

```bash
# 手动传输 tasks 下的文件到远程
/c/Users/AN/bin/rsync.exe -az --info=progress2 \
  -e "ssh -i /c/Users/AN/.ssh/id_ed25519 -o Compression=no -o ServerAliveInterval=60" \
  "tasks/20260304.140500.0/input/video.mp4" \
  "guohaoran@10.176.56.244:/data/disk1/guohaoran/cognitive-coding-engagement-analysis/tasks/20260304.140500.0/input/video.mp4"

# 从远程回传 tasks 下的文件到本地（注意本地目标用 Unix 风格路径）
/c/Users/AN/bin/rsync.exe -az --info=progress2 \
  -e "ssh -i /c/Users/AN/.ssh/id_ed25519 -o Compression=no -o ServerAliveInterval=60" \
  "guohaoran@10.176.56.244:/data/disk1/guohaoran/cognitive-coding-engagement-analysis/tasks/20260304.140500.0/automatic_speech_recognition/output.json" \
  "/c/Projects/cognitive-coding-engagement-analysis/tasks/20260304.140500.0/automatic_speech_recognition/output.json"
```

远程执行 b/c/d 脚本时需要设置环境变量：

```bash
# b_speaker_diarization.py
rc "cd /data/disk1/guohaoran/cognitive-coding-engagement-analysis && HF_HUB_CACHE=/data/disk1/guohaoran/model/huggingface/hub HF_TOKEN=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 uv run src/b_speaker_diarization.py 20260304.140500.0"

# d_crowd_detect.py
rc "cd /data/disk1/guohaoran/cognitive-coding-engagement-analysis && PANNS_CHECKPOINT=/data/disk1/guohaoran/model/panns/Cnn14_mAP=0.431.pth uv run src/d_crowd_detect.py 20260304.140500.0"
```

注意：远程服务器无外网访问，所有 HuggingFace 模型需本地下载后 rsync 传到服务器。

## Models

模型存放在远程服务器 `/data/disk1/guohaoran/model/` 下：

- `huggingface/hub/` — pyannote 系列模型（HF cache 格式）
  - `models--pyannote--speaker-diarization-3.1` — 说话人分离 pipeline 配置
  - `models--pyannote--segmentation-3.0` — 分割模型（5.7MB）
  - `models--pyannote--wespeaker-voxceleb-resnet34-LM` — 说话人嵌入模型（26MB）
  - `models--pyannote--speaker-diarization-community-1` — PLDA 模型（269KB）
  - `models--pyannote--overlapped-speech-detection` — 重叠检测配置（未使用，依赖的 segmentation@Interspeech2021 需单独申请权限）
- `whisper-large-v3/` — Whisper 语音识别模型（24GB），本地路径 `D:/Development/models/whisper-large-v3`
- `panns/` — PANNs 音频分类模型
  - `Cnn14_mAP=0.431.pth` — CNN14 模型（313MB），用于检测 crowd/babble 等 AudioSet 527 类声音事件

本地 HF cache: `C:\Users\AN\.cache\huggingface\hub\`
