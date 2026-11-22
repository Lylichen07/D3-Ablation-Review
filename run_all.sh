#!/bin/bash
mkdir -p logs
SUBSETS="Crafter Gen2 HotShot Lavie ModelScope MoonValley MorphStudio Show_1 Sora WildScrape"
REAL=GenVideo/csv/real_MSRVTT.csv

run() {  # $1 = log name, rest = extra args
  log="logs/$1.txt"; shift
  if [ -s "$log" ] && grep -qi "ap" "$log"; then
    echo "skip $log (already done)"; return
  fi
  echo "run  $log"
  python eval.py --gpu-id 0 --loss l2 "$@" > "$log" 2>&1
}

# Pre-trained encoders: 1 run each
for enc in XCLIP-16 ResNet-18; do
  for sub in $SUBSETS; do
    run "${enc}_${sub}" --encoder $enc --real-csv $REAL --fake-csv GenVideo/csv/$sub.csv
  done
done

# Random conditions: 3 seeds each
for enc in ResNet-18-rand XCLIP-16-rand RandProj; do
    for sub in $SUBSETS; do
      run "${enc}_${sub}" --encoder $enc --real-csv $REAL --fake-csv GenVideo/csv/$sub.csv
    done
done

echo "All done."