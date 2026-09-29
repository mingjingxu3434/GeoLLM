#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/make_synthetic_benchmark.py --out data/processed_synth --per-city 20 --seed 7
python scripts/run_main_experiment.py --data data/processed_synth --epochs 3 --out results/main_smoke
python scripts/run_ablation.py --data data/processed_synth --city Barcelona --epochs 1 --out results/ablation_smoke
python scripts/run_robustness.py --data data/processed_synth --city Barcelona --epochs 1 --out results/robustness_smoke
python scripts/run_cross_city.py --data data/processed_synth --epochs 1 --out results/cross_city_smoke
python scripts/run_graphgps_baseline.py --data data/processed_synth --city Barcelona --epochs 1 --out results/baselines_smoke
