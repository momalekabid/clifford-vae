#!/bin/bash
# pull finished runs from the pod to the mac every 2 hours.
# on the pod, each finished run's best.pt gets an fp16 copy (best_fp16.pt); only that copy comes
# down (the fp32 set for 20 seeds is ~52 GB, too big for this disk). wandb dirs, locks and
# codes_test1000.pt (regenerable from the checkpoint) stay on the pod.
#
#   nohup bash revision/sync_results.sh > ~/clifford_sync.log 2>&1 &
#   stop: pkill -f sync_results.sh
HOST=${HOST:-clifford-pod}
DEST=${DEST:-$HOME/Desktop/clifford-vae/results_rev}
mkdir -p "$DEST"
while true; do
  ssh -o ConnectTimeout=30 "$HOST" 'cd /workspace/clifford-vae && python - <<"EOF"
import glob, json, os, torch
n = 0
for m in glob.glob("/workspace/results_rev/*/*/seed*/metrics.json"):
    rd = os.path.dirname(m)
    src, dst = os.path.join(rd, "best.pt"), os.path.join(rd, "best_fp16.pt")
    try:
        if json.load(open(m)).get("status") != "done" or not os.path.exists(src) or os.path.exists(dst):
            continue
    except Exception:
        continue
    ck = torch.load(src, map_location="cpu")
    ck["state"] = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["state"].items()}
    ck["dtype"] = "float16 copy of best.pt"
    torch.save(ck, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    n += 1
print("fp16 copies made:", n)
EOF'
  rsync -az --exclude best.pt --exclude '*.tmp' --exclude .lock --exclude wandb/ --exclude codes_test1000.pt \
    "$HOST":/workspace/results_rev/ "$DEST"/
  echo "$(date) synced: $(ls "$DEST"/*/*/seed*/best_fp16.pt 2>/dev/null | wc -l) checkpoints, $(du -sh "$DEST" | cut -f1)"
  sleep 7200
done
