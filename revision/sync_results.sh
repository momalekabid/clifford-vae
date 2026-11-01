#!/bin/bash
# pull finished runs from the pod to the mac every 2 hours.
# on the pod, each finished run's best.pt gets an fp16 copy (best_fp16.pt); only that copy comes
# down (the fp32 set for 30 seeds is ~75 GB, too big for either disk). wandb dirs, locks and
# codes_test1000.pt (regenerable from the checkpoint) stay on the pod. once the fp16 copy is
# here, the pod fp16 copy and the fp32 best.pt are deleted (marker best_fp16.pt.synced).
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
        if json.load(open(m)).get("status") != "done" or not os.path.exists(src) or os.path.exists(dst) \
                or os.path.exists(dst + ".synced"):
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
  rsync -az --exclude best.pt --exclude '*.tmp' --exclude '*.synced' --exclude .lock --exclude wandb/ --exclude codes_test1000.pt \
    "$HOST":/workspace/results_rev/ "$DEST"/
  # free pod disk: once the fp16 copy arrived intact, drop it and the fp32 best.pt (the mac copy
  # is the one we keep), leave a marker so they aren't remade
  ( cd "$DEST" && find . -name best_fp16.pt -exec stat -f "%z %N" {} \; ) | ssh "$HOST" 'cd /workspace/results_rev && while read size f; do
      [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$size" ] && rm -f "$f" "$(dirname "$f")/best.pt" && touch "$f.synced"; done'
  echo "$(date) synced: $(ls "$DEST"/*/*/seed*/best_fp16.pt 2>/dev/null | wc -l) checkpoints, $(du -sh "$DEST" | cut -f1)"
  sleep 7200
done
