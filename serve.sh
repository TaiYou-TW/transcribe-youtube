#!/bin/zsh
# M3 轉錄服務啟動: 載入 .env, 用 caffeinate -s 防睡眠
cd "$(dirname "$0")"
set -a; source ./.env; set +a
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
exec /usr/bin/caffeinate -s ./.venv/bin/python server.py
