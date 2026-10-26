#!/usr/bin/env bash
# Sequential queue of (plan slug model) triples; pushes each when Kaggle frees a batch GPU slot.
# usage: bash queue.sh "full iot-full olmo2" "prefix iot-px qwen25" ...
cd "$(dirname "$0")"
for job in "$@"; do
  set -- $job
  plan="$1"; slug="$2-$3"; m="$3"
  while true; do
    out=$(python push.py run "$slug" "$m" "$plan" 2>&1)
    if echo "$out" | grep -q "successfully pushed"; then
      echo "$(date +%H:%M) pushed $slug ($plan)"
      break
    fi
    if ! echo "$out" | grep -q "Maximum batch GPU session"; then
      echo "$(date +%H:%M) unexpected push output for $slug:"; echo "$out" | tail -3
    fi
    sleep 240
  done
  sleep 150
  st=$(kaggle kernels status "abdelbassetchenna/$slug" 2>/dev/null | tail -1)
  echo "$(date +%H:%M) $st"
done
echo "$(date +%H:%M) queue done"
