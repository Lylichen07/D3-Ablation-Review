#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
mkdir -p logs
SUBSETS="Crafter Gen2 HotShot Lavie ModelScope MoonValley MorphStudio Show_1 Sora WildScrape"
REAL=GenVideo/csv/real_MSRVTT.csv

run() {
  log="logs/$1.txt"; shift
  # "Skipped (< 8 frames):" only appears in logs of finished runs that also wrote per-video scores
  if grep -q "Skipped (< 8 frames):" "$log" 2>/dev/null; then return; fi
  echo "$(date +%H:%M) run $log"
  python eval.py --gpu-id 0 --loss l2 "$@" > "$log" 2>&1
  grep -q "AP Score:" "$log" && echo "      ok" || echo "      FAILED: $(tail -1 "$log")"
}

for enc in XCLIP-16 ResNet-18; do
  for sub in $SUBSETS; do
    run "${enc}_${sub}" --encoder $enc --real-csv $REAL --fake-csv GenVideo/csv/$sub.csv
  done
done
for enc in ResNet-18-rand XCLIP-16-rand; do
  for seed in 0 1 2; do
    for sub in $SUBSETS; do
      run "${enc}_s${seed}_${sub}" --encoder $enc --seed $seed --real-csv $REAL --fake-csv GenVideo/csv/$sub.csv
    done
  done
done

# RandProj: adaptive_avg_pool2d has no valid kernel under 11.5.1 -> use the gfx1100 override for this condition only
for seed in 0 1 2; do
  for sub in $SUBSETS; do
    HSA_OVERRIDE_GFX_VERSION=11.0.0 run "RandProj_s${seed}_${sub}" --encoder RandProj --seed $seed \
      --real-csv $REAL --fake-csv GenVideo/csv/$sub.csv
  done
done
echo "Finished. $(grep -l 'AP Score:' logs/*_*.txt | wc -l) logs with results."
