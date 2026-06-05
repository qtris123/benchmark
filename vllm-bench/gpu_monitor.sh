#!/usr/bin/env bash
# Continuously polls GPU usage and writes status to a txt file.

OUTPUT_FILE="${1:-/localhome/local-triv/benchmarks/gpu_status.txt}"
INTERVAL="${2:-2}"  # seconds between updates

while true; do
    {
        echo "=== GPU Status @ $(date '+%Y-%m-%d %H:%M:%S') ==="
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit \
            --format=csv,noheader,nounits \
        | awk -F', ' '{
            printf "GPU %s (%s)\n", $1, $2
            printf "  Util:  %s%% GPU  |  %s%% MEM\n", $3, $4
            printf "  VRAM:  %s MiB / %s MiB\n", $5, $6
            printf "  Temp:  %s°C\n", $7
            printf "  Power: %s W / %s W\n", $8, $9
            print ""
        }'
        echo "--- Processes ---"
        nvidia-smi --query-compute-apps=pid,used_memory,name \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' '{
            printf "  PID %-8s  VRAM %-8s MiB  %s\n", $1, $2, $3
        }'
        echo ""
        echo "(refresh every ${INTERVAL}s  |  Ctrl-C to stop)"
    } >> "$OUTPUT_FILE"
    sleep "$INTERVAL"
done
