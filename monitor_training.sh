#!/bin/bash
RUN_DIR="checkpoints/f192-b15_2026-02-15_16-59-21"
METRICS="$RUN_DIR/training_metrics.json"

while true; do
  if [ -f "$METRICS" ]; then
    echo "=== $(date +%H:%M:%S) Training Progress ==="
    # Extract last iteration metrics
    python3 -c "
import json, sys
try:
    with open('$METRICS') as f:
        data = json.load(f)
    if data['iterations']:
        last = data['iterations'][-1]
        print(f\"Iteration {last['iteration']}/{data.get('total_iterations', '?')}\")
        print(f\"  Loss: total={last.get('loss_total', 0):.4f}, policy={last.get('loss_policy', 0):.4f}, value={last.get('loss_value', 0):.4f}\")
        print(f\"  Games: W={last.get('wins', 0)} D={last.get('draws', 0)} L={last.get('losses', 0)} (draw rate={last.get('draw_rate', 0)*100:.1f}%)\")
        print(f\"  Avg game length: {last.get('avg_game_length', 0):.1f} moves\")
        if 'grad_norm_avg' in last:
            print(f\"  Gradients: avg={last['grad_norm_avg']:.3f}, max={last['grad_norm_max']:.3f}\")
    else:
        print(\"No iterations completed yet\")
except Exception as e:
    print(f\"Waiting for metrics... ({e})\")
" 2>/dev/null || echo "Metrics file not ready"
  else
    echo "=== $(date +%H:%M:%S) Waiting for training to start..."
  fi
  echo ""
  sleep 30
done
