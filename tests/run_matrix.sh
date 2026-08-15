#!/bin/bash
# 对当前 8000 端口的服务跑完整测试矩阵（功能 + prefill/TTFT/decode + 捞针 + 并发）。
# 用法: ./tests/run_matrix.sh <label>   在仓库根目录执行
set -e
cd "$(dirname "$0")/.."
LABEL=$1
MODEL=$(curl -s http://127.0.0.1:8000/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
echo "model: $MODEL  label: $LABEL"

python3 tests/functional_test.py 8000 "$LABEL"
python3 tests/prefill_test.py http://127.0.0.1:8000 "$MODEL" "$LABEL" 1024 32768 100000 200000
python3 tests/needle_test.py http://127.0.0.1:8000 "$MODEL" "$LABEL" 100000 0.7
for N in 1 2 4 8; do
  python3 tests/conc_test.py http://127.0.0.1:8000 "$MODEL" "$N" 500 "$LABEL"
done
echo "matrix done: $LABEL"
