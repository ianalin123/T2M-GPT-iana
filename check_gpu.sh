#!/bin/bash
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu --format=csv
echo ""
echo "=== GPU Processes ==="
nvidia-smi pmon -c 1
