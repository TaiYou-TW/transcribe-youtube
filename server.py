#!/usr/bin/env python3
"""
M3 轉錄服務 (見 plan.md「M3 端:Whisper 轉錄服務」)

    POST /jobs        {"url": "...", "lang": "ja"} + Authorization: Bearer <token>
        -> 202 {"job_id": "...", "status": "queued", "position": 1}
    GET  /jobs/{id}   + Authorization: Bearer <token>
        -> {"status": "queued|running|done|error", "position": N,
            "result": {"transcript": "...", "source": "subs|whisper", ...}, "error": "..."}
    GET  /health      -> 200 {"status": "ok", "busy": bool, "queued": int}

同一組 url+lang 還在排隊/執行中時重複提交, 回同一個 job (bot 重送不會多跑)。
完成的 job 保留 JOB_TTL 秒供輪詢, 之後清掉 (查不到回 404)。

環境變數:
    TRANSCRIBE_TOKEN   必填, bearer token
    WHISPER_BACKEND    auto/cpp/mlx/faster (預設 auto, 同 transcribe.py; M3 上為 cpp)
    WHISPER_MODEL      覆寫模型 (別名或 repo id)
    QUEUE_MAX          排隊上限 (不含執行中), 超過回 429 (預設 5)
    JOB_TTL            完成的 job 保留秒數 (預設 3600)
    HOST / PORT        預設 0.0.0.0 / 8000
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
import re
import secrets
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

import transcribe as tx

TOKEN = os.environ.get("TRANSCRIBE_TOKEN", "")
BACKEND = os.environ.get("WHISPER_BACKEND", "auto")
MODEL = os.environ.get("WHISPER_MODEL") or None
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "5"))
JOB_TTL = int(os.environ.get("JOB_TTL", "3600"))

# 16GB 一次只安穩跑一個 whisper: 單一 worker 依序消化 queue
_queue: asyncio.Queue[str] = asyncio.Queue()
_jobs: dict[str, dict] = {}
_state: dict = {}


# --------------------------------------------------------------------------
# 字幕 / 音檔
# --------------------------------------------------------------------------
_VTT_TS = re.compile(r"^\d{2}:\d{2}(:\d{2})?\.\d{3} --> ")
_TAG = re.compile(r"<[^>]+>")


def vtt_to_text(vtt: str) -> str:
    lines: list[str] = []
    for raw in vtt.splitlines():
        line = raw.strip()
        if not line or line == "WEBVTT" or _VTT_TS.match(line) or line.isdigit():
            continue
        if line.startswith(("Kind:", "Language:", "NOTE", "STYLE")):
            continue
        line = _TAG.sub("", line).strip()
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return "\n".join(lines)


def fetch_subs(url: str, lang: str, workdir: Path) -> tuple[str | None, dict]:
    """只抓人工上傳的字幕 (不含自動字幕);有就回純文字。"""
    import yt_dlp

    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": str(workdir / "subs.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True) or {}
    for p in workdir.glob("subs.*.vtt"):
        text = vtt_to_text(p.read_text(encoding="utf-8", errors="ignore"))
        if text:
            return text, info
    return None, info


def to_wav16k(src: Path, dst: Path) -> Path:
    """16kHz 單聲道 16-bit PCM WAV (無損, 地端用)。"""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )
    return dst


def run_job(url: str, lang: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="kw_srv_") as tmp:
        workdir = Path(tmp)
        subs, info = fetch_subs(url, lang, workdir)
        base = {"title": info.get("title"), "video_id": info.get("id")}
        if subs:
            return {**base, "transcript": subs, "source": "subs"}

        got = tx.download_audio(url, workdir / "audio", playlist=False, quiet=True)
        if not got:
            raise RuntimeError("音檔下載失敗")
        wav = to_wav16k(got[0].path, workdir / "audio.wav")
        segs, meta = _state["backend"].transcribe(wav, lang, verbose=False)
        text = "\n".join(s["text"] for s in segs if s["text"])
        return {
            **base,
            "transcript": text,
            "source": "whisper",
            "model": _state["model_id"],
            "duration": meta.get("duration"),
        }


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def _queued_jobs() -> list[str]:
    return sorted(
        (j for j in _jobs.values() if j["status"] == "queued"),
        key=lambda j: j["created"],
    )


def _view(job: dict) -> dict:
    out = {k: job[k] for k in ("job_id", "status", "url", "lang", "created", "started", "finished")}
    if job["status"] == "queued":
        out["position"] = next(i for i, j in enumerate(_queued_jobs(), 1) if j is job)
    if job["status"] == "done":
        out["result"] = job["result"]
    if job["status"] == "error":
        out["error"] = job["error"]
    return out


def _purge() -> None:
    now = time.time()
    for jid in [k for k, j in _jobs.items() if j["finished"] and now - j["finished"] > JOB_TTL]:
        del _jobs[jid]


async def worker() -> None:
    while True:
        jid = await _queue.get()
        job = _jobs.get(jid)
        if job is None:
            continue
        job.update(status="running", started=time.time())
        try:
            job["result"] = await asyncio.to_thread(run_job, job["url"], job["lang"])
            job["status"] = "done"
        except Exception as exc:
            job.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            job["finished"] = time.time()
            _queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    tx.ensure_ffmpeg()
    backend_name = tx.pick_backend(BACKEND)
    model_id = tx.resolve_model(MODEL, backend_name)
    # 模型常駐: 啟動時載入一次 (cpp 每個 job 起一次 whisper-cli, 載入約 0.3 秒, 這裡只先下載好模型)
    backend = tx.make_backend(backend_name, model_id)
    if backend_name == "mlx":
        # mlx_whisper 是 lazy load, 先用 1 秒靜音暖機把權重拉進記憶體
        with tempfile.TemporaryDirectory() as tmp:
            silence = Path(tmp) / "warm.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                 "-i", "anullsrc=r=16000:cl=mono", "-t", "1", str(silence)],
                check=True,
            )
            backend.transcribe(silence, "ja", verbose=False)
    _state.update(backend=backend, backend_name=backend_name, model_id=model_id)
    task = asyncio.create_task(worker())
    print(f"[ready] {backend_name} / {model_id}", flush=True)
    yield
    task.cancel()


app = FastAPI(title="M3 transcribe", lifespan=lifespan)


def check_token(authorization: str = Header(default="")) -> None:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, TOKEN):
        raise HTTPException(status_code=401, detail="invalid token")


class TranscribeReq(BaseModel):
    url: str
    lang: str = "ja"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "busy": any(j["status"] == "running" for j in _jobs.values()),
        "queued": len(_queued_jobs()),
        "backend": _state.get("backend_name"),
        "model": _state.get("model_id"),
    }


@app.post("/jobs", status_code=202, dependencies=[Depends(check_token)])
async def submit_job(req: TranscribeReq):
    _purge()
    for job in _jobs.values():
        if job["url"] == req.url and job["lang"] == req.lang and job["status"] in ("queued", "running"):
            return _view(job)
    queued = len(_queued_jobs())
    if queued >= QUEUE_MAX:
        raise HTTPException(
            status_code=429,
            detail={"error": "queue full", "queued": queued},
            headers={"Retry-After": "60"},
        )
    jid = uuid.uuid4().hex
    _jobs[jid] = {
        "job_id": jid, "status": "queued", "url": req.url, "lang": req.lang,
        "created": time.time(), "started": None, "finished": None,
        "result": None, "error": None,
    }
    _queue.put_nowait(jid)
    return _view(_jobs[jid])


@app.get("/jobs/{job_id}", dependencies=[Depends(check_token)])
async def get_job(job_id: str):
    _purge()
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _view(job)


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("TRANSCRIBE_TOKEN 未設定")
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
