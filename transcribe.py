#!/usr/bin/env python3
"""
YouTube 日文影片 -> 逐字稿

用法:
    python transcribe.py "https://www.youtube.com/watch?v=XXXX"
    python transcribe.py video.mp4 --format txt,srt
    python transcribe.py "https://..." --playlist --outdir transcripts
    python transcribe.py "https://..." --model kotoba          # 用別名切模型
    python transcribe.py --list-models

後端自動判斷:
    macOS + Apple Silicon + 有 whisper-cli -> whisper.cpp (anime-whisper ggml q8_0, Metal GPU)
    其他 (含本機 WSL + RTX 3060) -> faster-whisper (anime-whisper CT2 int8)
    (mlx 後端要明確指定 --backend mlx)

模型:
    地端主力 = litagin/anime-whisper (kotoba-v2.0 微調於動漫/Galgame 語音),
    對 VTuber 情緒與非語言音辨識最佳。faster 路徑用社群 CT2 int8 轉檔。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import site
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# backend -> 預設模型
MODELS = {
    # anime-whisper 沒有官方 CT2 檔,用社群轉好的 int8 版 (MIT, 768MB)
    "faster": "quantumcookie/anime-whisper-ct2-int8",
    # M3 之後建議改用支援長片的 Qwen3-ASR JA MLX 版;kotoba-mlx 為暫時預設
    "mlx": "kaiinui/kotoba-whisper-v2.0-mlx",
    # whisper.cpp: <HF repo>/<檔名> 或本機 .bin 路徑。社群 ggml 轉檔 (MIT)
    "cpp": "Aratako/anime-whisper-ggml/ggml-anime-whisper-q8_0.bin",
}

# --model 可以給別名, 省得記整串 repo id
MODEL_ALIASES = {
    "anime": "quantumcookie/anime-whisper-ct2-int8",          # faster (CT2 int8)
    "kotoba": "kotoba-tech/kotoba-whisper-v2.0-faster",       # faster
    "kotoba-mlx": "kaiinui/kotoba-whisper-v2.0-mlx",      # mlx
    "anime-cpp": "Aratako/anime-whisper-ggml/ggml-anime-whisper-q8_0.bin",  # cpp
    "large-v3": "Systran/faster-whisper-large-v3",            # faster, 通用最穩
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",  # faster fallback
}

# trim_local 產生的暫存切片, 結束時清掉
_TMP_CLIPS: list[Path] = []


# --------------------------------------------------------------------------
# 環境: CUDA 動態函式庫 + ffmpeg 路徑
# --------------------------------------------------------------------------
def ensure_cuda_libs() -> None:
    """pip 版的 cuBLAS/cuDNN 藏在 site-packages/nvidia/*/lib，ctranslate2 用
    dlopen 找它們，所以必須在 process 啟動前就放進 LD_LIBRARY_PATH。
    補好之後 re-exec 自己一次。"""
    if platform.system() != "Linux" or os.environ.get("_KW_RELAUNCHED"):
        return

    dirs = []
    for sp in set(site.getsitepackages() + [site.getusersitepackages()]):
        dirs.extend(sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib"))))
    if not dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    missing = [d for d in dirs if d not in current]
    if not missing:
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + [p for p in current if p])
    os.environ["_KW_RELAUNCHED"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def ensure_ffmpeg() -> None:
    """把 static-ffmpeg 帶的 binary 放進 PATH，省掉 apt install ffmpeg。"""
    try:
        from static_ffmpeg import run as sf_run
    except ImportError:
        return
    try:
        ffmpeg, _ffprobe = sf_run.get_or_fetch_platform_executables_else_raise()
    except Exception:
        return
    bin_dir = str(Path(ffmpeg).parent)
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


# --------------------------------------------------------------------------
# 下載
# --------------------------------------------------------------------------
def slugify(name: str, limit: int = 120) -> str:
    name = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", name).strip(" .")
    return (name or "audio")[:limit]


def parse_time(value: str | None) -> float | None:
    """'90' / '1:30' / '01:02:03' / '1:30.5' -> 秒"""
    if value is None:
        return None
    parts = str(value).strip().split(":")
    if not all(parts) or len(parts) > 3:
        raise ValueError(f"看不懂的時間: {value}")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def trim_local(path: Path, start: float | None, end: float | None) -> Path:
    """音檔切片 (整段下載回來後在本機切, 比 yt-dlp 的 ranged download 快很多)。
    暫存目錄記進 _TMP_CLIPS, 結束時統一清掉。"""
    import subprocess
    import tempfile

    if start is None and end is None:
        return path
    tmp_dir = Path(tempfile.mkdtemp(prefix="kw_clip_"))
    _TMP_CLIPS.append(tmp_dir)
    out = tmp_dir / f"clip{path.suffix or '.m4a'}"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", str(end - (start or 0))]
    cmd += ["-vn", "-c:a", "copy", str(out)]
    if subprocess.run(cmd).returncode != 0:  # copy 不行就重編碼
        cmd[cmd.index("copy")] = "aac"
        subprocess.run(cmd, check=True)
    return out


def cleanup_tmp() -> None:
    for d in _TMP_CLIPS:
        shutil.rmtree(d, ignore_errors=True)
    _TMP_CLIPS.clear()


@dataclass
class Source:
    path: Path
    title: str


# YouTube 的 ranged download (download_ranges) 實測比整段下載慢約 75 倍
# (69 分鐘整段 25MB 只要 5 秒, 同一支只取 10 分鐘卻要 300 秒),
# 所以一律整段抓回來, 要切片再交給本機 ffmpeg。
AUDIO_FORMATS = {
    # Whisper 內部就是 16kHz 單聲道, 49kbps 的音軌對「談話」辨識實測沒有影響。
    # 注意: anime-whisper 吃情緒/非語言音, 有 BGM 的內容值得拿同一支比 low vs best,
    # 確認低位元率沒吃掉細節再定案。
    "low": ("bestaudio[abr>=32]/bestaudio/best", ["+abr"]),
    "best": ("bestaudio/best", None),
}


def download_audio(
    url: str,
    audio_dir: Path,
    playlist: bool,
    quiet: bool,
    quality: str = "low",
) -> list[Source]:
    import yt_dlp

    audio_dir.mkdir(parents=True, exist_ok=True)
    fmt, fmt_sort = AUDIO_FORMATS[quality]
    opts = {
        "format": fmt,
        "outtmpl": str(audio_dir / "%(title).120B [%(id)s].%(ext)s"),
        # preferredcodec="best" 只做 remux, 不會把 opus 重編碼成 AAC (省時間也省一次品質損失)
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "best"}],
        "http_chunk_size": 10 * 1024 * 1024,
        "noplaylist": not playlist,
        "quiet": quiet,
        "no_warnings": quiet,
        "noprogress": quiet,
        "retries": 5,
        "ignoreerrors": playlist,
    }
    if fmt_sort:
        opts["format_sort"] = fmt_sort

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    entries = info.get("entries") if info and "entries" in info else [info]
    out: list[Source] = []
    for entry in entries or []:
        if not entry:
            continue
        downloads = entry.get("requested_downloads") or []
        path = downloads[0].get("filepath") if downloads else None
        if not path or not Path(path).exists():
            stem = Path(entry.get("_filename", "")).with_suffix("")
            hits = sorted(audio_dir.glob(glob.escape(stem.name) + ".*")) if stem.name else []
            path = str(hits[0]) if hits else None
        if not path:
            print(f"  ! 找不到下載檔案: {entry.get('title')}", file=sys.stderr)
            continue
        out.append(Source(Path(path), slugify(entry.get("title") or Path(path).stem)))
    return out


# --------------------------------------------------------------------------
# 轉錄
# --------------------------------------------------------------------------
def pick_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system() == "Darwin" and platform.machine() == "arm64" and shutil.which("whisper-cli"):
        return "cpp"
    return "faster"


def resolve_model(requested: str | None, backend: str) -> str:
    """--model 可以是別名、完整 repo id, 或 None (用 backend 預設)。"""
    if not requested:
        return MODELS[backend]
    return MODEL_ALIASES.get(requested, requested)


def fmt_ts(seconds: float, sep: str = ",") -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


class FasterBackend:
    def __init__(
        self,
        model_id: str,
        device: str,
        compute_type: str,
        beam_size: int,
        chunk_length: int,
        no_repeat_ngram_size: int,
    ):
        from faster_whisper import WhisperModel

        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        if compute_type == "auto":
            # CT2 int8 權重 (anime-whisper) 搭 int8_float16 最合:int8 權重 + float16 活化,
            # 比純 float16 省 VRAM/更快, 精度幾乎無損。非量化模型會在載入時自動量化。
            compute_type = "int8_float16" if device == "cuda" else "int8"
        print(f"[model] faster-whisper / {model_id} / {device} / {compute_type}")
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        self.chunk_length = chunk_length
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def transcribe(self, audio: Path, language: str, verbose: bool):
        segments, info = self.model.transcribe(
            str(audio),
            language=None if language == "auto" else language,
            task="transcribe",
            beam_size=self.beam_size,
            chunk_length=self.chunk_length,
            # distil / anime-whisper 容易在長音檔、情緒/非語言音上鬼打牆重複:
            #   - 關掉前文條件化
            #   - no_repeat_ngram_size=5 (litagin 針對 anime-whisper 的建議)
            #   - VAD 切段 (短句訓練的模型不擅長長片, 先切乾淨語音段)
            condition_on_previous_text=False,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        out = []
        for seg in segments:
            item = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            out.append(item)
            if verbose:
                pct = f"{min(seg.end / info.duration * 100, 100):5.1f}%" if info.duration else "  --  "
                print(f"  {pct} [{fmt_ts(seg.start, '.')}] {item['text']}")
        return out, {"language": info.language, "duration": info.duration}


class MlxBackend:
    def __init__(self, model_id: str):
        import mlx_whisper  # noqa: F401

        print(f"[model] mlx-whisper / {model_id}")
        # mlx_whisper 沒有 VAD, 走 OpenAI 式 30 秒滑動窗長片轉錄。
        # kotoba / anime-whisper 這類短句訓練的 distil 模型在長片的靜音/BGM 段
        # 容易飄字、跳針。搬到 M3 跑長影片前, 建議:
        #   (a) 前置 silero-vad 切段再逐段丟, 或
        #   (b) 直接換成支援長片的 Qwen3-ASR (有 JA 的 MLX 版)。
        if "anime" in model_id or "kotoba" in model_id:
            print(
                "  ! 提醒: MLX 路徑無 VAD, 此模型為短句訓練, 長片可能跳針。"
                "長影片建議改用 faster 後端, 或在 M3 上換 Qwen3-ASR。",
                file=sys.stderr,
            )
        self.model_id = model_id

    def transcribe(self, audio: Path, language: str, verbose: bool):
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(audio),
            path_or_hf_repo=self.model_id,
            language=None if language == "auto" else language,
            task="transcribe",
            condition_on_previous_text=False,
            verbose=verbose or None,
        )
        segs = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        duration = segs[-1]["end"] if segs else 0.0
        return segs, {"language": result.get("language", language), "duration": duration}


class WhisperCppBackend:
    """whisper.cpp (whisper-cli, Apple Silicon 走 Metal GPU)。

    不用 whisper.cpp 內建 VAD: 它把語音段接成連續音訊再以 30 秒窗前進,
    anime-whisper 不出時間戳, 每窗只吐前一兩句就跳下一窗, 實測漏掉約一半內容且會跳針。
    改成先用 silero VAD 切成 <= chunk_length 秒的片段, 每段各自解碼 (同 faster-whisper 的 15 秒窗),
    -ac 讓 encoder 只算 chunk_length 秒, 省一半 encode 時間。
    """

    SR = 16000

    def __init__(self, model_id: str, beam_size: int, chunk_length: int):
        self.cli = shutil.which("whisper-cli")
        if not self.cli:
            raise RuntimeError("找不到 whisper-cli, 請先 brew install whisper-cpp")
        self.model_path = self._resolve(model_id)
        self.beam_size = beam_size
        self.chunk_length = chunk_length
        print(f"[model] whisper.cpp / {model_id}")

    @staticmethod
    def _resolve(model_id: str) -> str:
        local = Path(model_id).expanduser()
        if local.exists():
            return str(local)
        repo, _, filename = model_id.rpartition("/")
        if repo.count("/") != 1 or not filename:
            raise ValueError(f"cpp 模型要給本機路徑或 <owner>/<repo>/<檔名>: {model_id}")
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo, filename)

    def _chunks(self, audio):
        """VAD 語音段合併成 <= chunk_length 秒的片段, 回 (原始起點秒, 原始終點秒, 音訊)。"""
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        speech = get_speech_timestamps(
            audio,
            VadOptions(min_silence_duration_ms=500, max_speech_duration_s=self.chunk_length),
        )
        limit = self.chunk_length * self.SR
        groups: list[list[dict]] = []
        for sp in speech:
            size = sp["end"] - sp["start"]
            if groups and sum(g["end"] - g["start"] for g in groups[-1]) + size <= limit:
                groups[-1].append(sp)
            else:
                groups.append([sp])
        for g in groups:
            yield (
                g[0]["start"] / self.SR,
                g[-1]["end"] / self.SR,
                np.concatenate([audio[sp["start"]:sp["end"]] for sp in g]),
            )

    def transcribe(self, audio: Path, language: str, verbose: bool):
        import subprocess
        import tempfile
        import wave

        import numpy as np
        from faster_whisper.audio import decode_audio

        samples = decode_audio(str(audio), sampling_rate=self.SR)
        duration = len(samples) / self.SR
        chunks = list(self._chunks(samples))
        if not chunks:
            return [], {"language": language, "duration": duration}

        with tempfile.TemporaryDirectory(prefix="kw_cpp_") as tmp:
            wavs = []
            for i, (_, _, data) in enumerate(chunks):
                path = Path(tmp) / f"{i:05d}.wav"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(self.SR)
                    w.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
                wavs.append(path)

            cmd = [
                self.cli, "-m", self.model_path,
                "-l", "auto" if language == "auto" else language,
                "-bs", str(self.beam_size), "-bo", str(self.beam_size),
                "-mc", "0",  # 不帶前文 (同 condition_on_previous_text=False)
                "-ac", str(self.chunk_length * 50),  # encoder 只算 chunk_length 秒
                "-nt", "-otxt", "-np",
            ]
            for w in wavs:
                cmd += ["-f", str(w)]
            proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if proc.returncode != 0:
                raise RuntimeError(f"whisper-cli 失敗: {proc.stderr.strip()[-500:]}")

            out = []
            for (start, end, _), w in zip(chunks, wavs):
                txt = Path(f"{w}.txt")
                text = txt.read_text(encoding="utf-8", errors="replace").strip() if txt.exists() else ""
                text = " ".join(text.split())
                if not text:
                    continue
                out.append({"start": start, "end": end, "text": text})
                if verbose:
                    print(f"  {min(end / duration * 100, 100):5.1f}% [{fmt_ts(start, '.')}] {text}")
        return out, {"language": language, "duration": duration}


def make_backend(
    name: str,
    model_id: str,
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    chunk_length: int = 15,
    no_repeat_ngram_size: int = 5,
):
    if name == "mlx":
        return MlxBackend(model_id)
    if name == "cpp":
        return WhisperCppBackend(model_id, beam_size, chunk_length)
    return FasterBackend(model_id, device, compute_type, beam_size, chunk_length, no_repeat_ngram_size)


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


# --------------------------------------------------------------------------
# 輸出
# --------------------------------------------------------------------------
def write_outputs(segs: list[dict], meta: dict, outdir: Path, stem: str, formats: list[str]) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    if "txt" in formats:
        p = outdir / f"{stem}.txt"
        p.write_text("\n".join(s["text"] for s in segs if s["text"]) + "\n", encoding="utf-8")
        written.append(p)

    if "srt" in formats:
        p = outdir / f"{stem}.srt"
        lines = []
        for i, s in enumerate(segs, 1):
            lines.append(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text']}\n")
        p.write_text("\n".join(lines), encoding="utf-8")
        written.append(p)

    if "vtt" in formats:
        p = outdir / f"{stem}.vtt"
        lines = ["WEBVTT", ""]
        for s in segs:
            lines.append(f"{fmt_ts(s['start'], '.')} --> {fmt_ts(s['end'], '.')}")
            lines.append(s["text"])
            lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")
        written.append(p)

    if "json" in formats:
        p = outdir / f"{stem}.json"
        p.write_text(
            json.dumps({"meta": meta, "segments": segs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(p)

    return written


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="用 yt-dlp 抓 YouTube 音檔, 再用 anime-whisper / kotoba-whisper 轉日文逐字稿",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="*", help="YouTube 網址, 或本機音檔/影片路徑")
    ap.add_argument("--outdir", type=Path, default=ROOT / "transcripts", help="逐字稿輸出目錄")
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "downloads", help="音檔下載目錄")
    ap.add_argument("--format", default="txt,srt", help="輸出格式, 逗號分隔: txt,srt,vtt,json")
    ap.add_argument("--language", default="ja", help="語言代碼, auto 為自動偵測")
    ap.add_argument("--backend", choices=["auto", "faster", "mlx", "cpp"], default="auto")
    ap.add_argument(
        "--model",
        default=None,
        help="覆寫模型: 別名 (anime/kotoba/kotoba-mlx/anime-cpp/large-v3/large-v3-turbo) 或完整 repo id",
    )
    ap.add_argument("--list-models", action="store_true", help="列出可用的模型別名後結束")
    ap.add_argument("--device", default="auto", help="faster 後端: auto/cuda/cpu")
    ap.add_argument("--compute-type", default="auto", help="faster 後端: auto/float16/int8_float16/int8")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=5,
        help="防重複 (anime-whisper 建議 5, 設 0 關閉)",
    )
    ap.add_argument("--chunk-length", type=int, default=15, help="kotoba/anime-whisper 建議 15 秒")
    ap.add_argument("--start", default=None, help="從第幾秒開始, 例 90 或 1:30")
    ap.add_argument("--duration", default=None, help="只處理多長, 例 600 或 10:00")
    ap.add_argument("--end", default=None, help="到第幾秒為止 (與 --duration 擇一)")
    ap.add_argument(
        "--audio-quality",
        choices=["low", "best"],
        default="low",
        help="low = 最低位元率音軌 (約 49kbps, 談話無影響, 檔案小 3 倍); BGM 內容建議 best",
    )
    ap.add_argument("--playlist", action="store_true", help="允許整個播放清單")
    ap.add_argument("--delete-audio", action="store_true", help="轉錄完刪掉下載的音檔")
    ap.add_argument("-q", "--quiet", action="store_true", help="不要逐句印出")
    args = ap.parse_args()

    if args.list_models:
        print("模型別名:")
        for alias, repo in MODEL_ALIASES.items():
            print(f"  {alias:16s} -> {repo}")
        print("\n後端預設:")
        for backend, repo in MODELS.items():
            print(f"  {backend:16s} -> {repo}")
        return 0

    if not args.inputs:
        ap.error("需要至少一個網址或檔案路徑 (或用 --list-models)")

    formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
    bad = set(formats) - {"txt", "srt", "vtt", "json"}
    if bad:
        ap.error(f"不支援的格式: {', '.join(sorted(bad))}")

    try:
        start = parse_time(args.start)
        end = parse_time(args.end)
        dur = parse_time(args.duration)
    except ValueError as exc:
        ap.error(str(exc))
    if dur is not None:
        if end is not None:
            ap.error("--duration 和 --end 只能擇一")
        end = (start or 0.0) + dur

    ensure_ffmpeg()

    backend_name = pick_backend(args.backend)
    model_id = resolve_model(args.model, backend_name)

    try:
        # 先把所有音檔準備好, 再載入模型 (模型只載一次)
        sources: list[Source] = []
        for item in args.inputs:
            local = Path(item).expanduser()
            if local.exists():
                sources.append(Source(local, slugify(local.stem)))
                continue
            print(f"[下載] {item}")
            got = download_audio(item, args.audio_dir, args.playlist, args.quiet, args.audio_quality)
            for s in got:
                print(f"  -> {s.path}")
            sources.extend(got)

        if start is not None or end is not None:
            span = f"{args.start or 0} ~ {args.end or args.duration or 'end'}"
            print(f"[切片] {span}")
            sources = [Source(trim_local(s.path, start, end), s.title) for s in sources]

        if not sources:
            print("沒有可轉錄的音檔。", file=sys.stderr)
            return 1

        backend = make_backend(
            backend_name,
            model_id,
            args.device,
            args.compute_type,
            args.beam_size,
            args.chunk_length,
            args.no_repeat_ngram_size,
        )

        failed = 0
        for src in sources:
            print(f"\n[轉錄] {src.title}")
            try:
                segs, meta = backend.transcribe(src.path, args.language, verbose=not args.quiet)
            except Exception as exc:
                print(f"  ! 失敗: {exc}", file=sys.stderr)
                failed += 1
                continue

            meta = {**meta, "title": src.title, "source": str(src.path), "model": model_id}
            for p in write_outputs(segs, meta, args.outdir, src.title, formats):
                print(f"  ✓ {p}")

            if args.delete_audio and src.path.resolve().is_relative_to(args.audio_dir.resolve()):
                src.path.unlink(missing_ok=True)

        return 1 if failed else 0
    finally:
        cleanup_tmp()


if __name__ == "__main__":
    ensure_cuda_libs()
    sys.exit(main())
