#!/bin/bash
# runs on the mac. waits for the cnn seeds 11-30 sweep to exit, then: pushes the seed 1 fp16
# checkpoints to a separate root on the pod (so the sync loop never touches them), runs the
# clean timing on the now idle gpu, pulls timing.json back, waits until every finished run is
# synced to the mac, then stops (not terminates) the pod.
HOST=clifford-pod
POD=2rarwzc6vgkvvx
LOCAL=$HOME/Desktop/clifford-vae/results_rev
q() { ssh -o ConnectTimeout=60 "$HOST" "$1" 2>/dev/null; }
while [ "$(q 'pgrep -f "[s]weep.py --groups cifar10,fashionmnist --seeds 11-30" >/dev/null && echo up || echo down')" != "down" ]; do sleep 900; done
echo "$(date) sweep finished"

cd "$LOCAL" && rsync -aR --include '*/' --include 'seed1/best_fp16.pt' --include 'seed1/config.json' --exclude '*' \
  cifar10/ fashionmnist/ "$HOST":/workspace/timing_root/
q 'cd /workspace/clifford-vae && REVISION_RESULTS=/workspace/timing_root python revision/timing.py --datasets cifar10,fashionmnist > /workspace/logs/timing.log 2>&1'
mkdir -p "$LOCAL/_aggregate"
scp -q "$HOST":/workspace/timing_root/_aggregate/timing.json "$LOCAL/_aggregate/timing.json" && echo "$(date) timing pulled"

while true; do
  left=$(q 'cd /workspace/results_rev && n=0; for m in */*/seed*/metrics.json; do d=$(dirname $m); grep -q "\"status\": \"done\"" $m && [ -f $d/best.pt ] && n=$((n+1)); done; echo $n')
  [ "$left" = "0" ] && break
  echo "$(date) waiting on sync: ${left:-?} runs"
  sleep 900
done
pkill -f sync_results.sh
curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $(cat ~/.runpod/api_key)" \
  https://api.runpod.io/graphql -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD\\\"}) { id desiredStatus } }\"}"
echo; echo "$(date) pod stopped"
