# transcribe-youtube

把 YouTube 日文影片（VTuber 直播為主）轉成逐字稿。包含兩部分：

- **`transcribe.py`**：命令列工具，下載音檔 → 轉錄 → 輸出 txt / srt / vtt / json。
- **`server.py`**：常駐的轉錄服務（FastAPI），用「提交 job + 輪詢」的 HTTP API，給 Discord bot 等外部程式呼叫。

轉錄模型是 [litagin/anime-whisper](https://huggingface.co/litagin/anime-whisper)（以 kotoba-whisper-v2.0 微調於動漫 / Galgame 語音），對 VTuber 的情緒與非語言音辨識最好。

## 後端

| 後端 | 何時用 | 模型 |
| --- | --- | --- |
| `cpp`（whisper.cpp，Metal GPU） | Apple Silicon 上有 `whisper-cli` 時的預設 | `Aratako/anime-whisper-ggml` q8_0 |
| `faster`（faster-whisper / CTranslate2） | 其他平台的預設（CUDA 或 CPU） | `quantumcookie/anime-whisper-ct2-int8` |
| `mlx`（mlx-whisper） | 需明確指定；沒有 VAD，長片容易跳針 | `kaiinui/kotoba-whisper-v2.0-mlx` |

`cpp` 後端先用 silero VAD 把語音切成 ≤15 秒的片段，再把每段分別交給 whisper.cpp 解碼。whisper.cpp 內建的 VAD 會把語音段接起來、以 30 秒窗口前進，而 anime-whisper 不輸出時間戳，每個窗口只會吐出前一兩句，實測漏掉約一半內容，還會跳針。

M3（16GB）上轉錄一支 1 小時直播（約 39 分鐘語音）的實測：

| 後端 | 轉錄時間 | 峰值記憶體 |
| --- | --- | --- |
| faster-whisper（CPU int8） | 977 秒 | 8.5 GB |
| whisper.cpp（先切 ≤15 秒片段） | 229 秒 | 1.1 GB |

兩者的逐字稿品質相當。

## 安裝（macOS / Apple Silicon）

```sh
brew install uv ffmpeg whisper-cpp
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements.txt
```

模型在第一次使用時自動從 Hugging Face 下載。

## 命令列

```sh
.venv/bin/python transcribe.py "https://www.youtube.com/watch?v=XXXX"
.venv/bin/python transcribe.py video.mp4 --format txt,srt
.venv/bin/python transcribe.py "https://..." --start 1:30 --duration 10:00
.venv/bin/python transcribe.py "https://..." --backend faster --model anime
.venv/bin/python transcribe.py --list-models
```

輸出放在 `transcripts/`，下載的音檔放在 `downloads/`。其他參數見 `--help`。

## 轉錄服務

### 設定與啟動

```sh
cp .env.example .env    # 填入 TRANSCRIBE_TOKEN (openssl rand -hex 24)
./serve.sh              # 前景執行 (caffeinate -s 防睡眠)
```

用 launchd 常駐（開機自啟、掛掉自動重啟）：

```sh
sed "s#__DIR__#$PWD#g" deploy/com.youtube-transcript.m3.plist \
  > ~/Library/LaunchAgents/com.youtube-transcript.m3.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.youtube-transcript.m3.plist

launchctl kickstart -k gui/$(id -u)/com.youtube-transcript.m3   # 重啟
launchctl bootout gui/$(id -u)/com.youtube-transcript.m3        # 停止
tail -f logs/server.log
```

要蓋著筆電跑，`caffeinate -s` 不夠（蓋上還是會睡），另外需要 `sudo pmset -a disablesleep 1`。這個設定插不插電都生效，拔電後也不會睡，請確認電腦會一直插著電。

### 環境變數

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `TRANSCRIBE_TOKEN` | （必填） | Bearer token |
| `WHISPER_BACKEND` | `auto` | `auto` / `cpp` / `mlx` / `faster` |
| `WHISPER_MODEL` | 依後端 | 別名、HF repo id；`cpp` 可給 `<owner>/<repo>/<檔名>` 或本機 `.bin` 路徑 |
| `QUEUE_MAX` | `5` | 排隊上限（不含執行中），超過回 429 |
| `JOB_TTL` | `3600` | 完成的 job 保留秒數，過期後查詢回 404 |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | 監聽位址 |

### API

除 `/health` 外都要帶 `Authorization: Bearer <token>`。

| Method / Path | 說明 | 回應 |
| --- | --- | --- |
| `POST /jobs` `{"url": "...", "lang": "ja"}` | 提交轉錄 job | `202 {"job_id", "status": "queued", "position"}` |
| `GET /jobs/{id}` | 查詢 job | `{"status": "queued\|running\|done\|error", "position", "result", "error"}` |
| `GET /health` | 存活探測 | `{"status": "ok", "busy", "queued", "backend", "model"}` |

- 單一 worker 依提交順序一次處理一支影片。
- 同一組 `url` + `lang` 還在排隊或執行中時重複提交，會回同一個 job。
- 影片有人工上傳的字幕就直接回傳（`source: "subs"`），否則下載音檔轉錄（`source: "whisper"`）。
- `result`：`{"transcript", "source", "title", "video_id", "model", "duration"}`。

```sh
source .env
curl -s -X POST localhost:8000/jobs -H "Authorization: Bearer $TRANSCRIBE_TOKEN" \
  -H 'Content-Type: application/json' -d '{"url": "https://www.youtube.com/watch?v=XXXX"}'
curl -s localhost:8000/jobs/<job_id> -H "Authorization: Bearer $TRANSCRIBE_TOKEN"
```

## License

MIT
