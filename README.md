# CARE GeoLLM — Python reproduction project

A modular PyTorch implementation of the CARE GeoLLM method described in the supplied manuscript.
The repository separates **method reproduction**, **real geospatial preprocessing**, and **experiment reproduction** so that the model can be tested immediately while real city data remain versioned and replaceable.

## 1. What is implemented

### Core method
- typed geospatial graph input and relative Fourier coordinates;
- compact 6-layer planning-language Transformer;
- at most 16 structured intervention tokens;
- entity alignment masks;
- direct intervention feature updates;
- rank-8 low-rank hypernetwork modulation;
- 8-block, 8-head GeoTransformer factual/counterfactual passes;
- shared four-target decoder;
- heteroscedastic uncertainty head;
- untreated-entity consistency loss;
- 90% split-conformal calibration.

### Data pipeline
- EUBUCCO-compatible vector loader;
- WSF3D raster sampling hooks;
- WorldPop raster sampling hooks;
- optional OSM-derived walk/service support layer helper;
- canonical 24-feature graph schema;
- spatially grouped 70/10/10/10 split;
- deterministic tokenizer fallback;
- `.pt` community graph format.

### Experiments
- main CARE GeoLLM experiment;
- GraphGPS-style trainable baseline;
- official/pre-extracted embedding interface for SatMAE, Scale-MAE, UrbanCLIP, RemoteCLIP, GeoChat, UrbanLLM, and AnySat;
- ablations: no language operator, no consistency, static attention, no uncertainty;
- spatial-block missingness robustness at 0/5/10/20/30%;
- leave-one-city-out Barcelona/Rotterdam/Vienna transfer;
- MAE, RMSE, R² and Spearman metrics;
- CSV/JSON result export and conformal interval coverage.

## 2. Important reproduction boundary

The manuscript specifies the architecture, objective, training hyperparameters, data sources, evaluation protocol, and reported results. It does **not** release a canonical processed Barcelona/Rotterdam/Vienna graph package, intervention/outcome annotations, exact city screening polygons, official plan tokenizer checkpoint, or exact street/service extraction files.

Therefore:
1. the model architecture and experimental interfaces are implemented directly;
2. public data adapters are provided;
3. a deterministic synthetic benchmark is included for executable testing;
4. reported paper numbers are **not hard-coded as training output**;
5. real empirical reproduction requires the actual intervention labels/splits or an independently constructed versioned benchmark.

## 3. Installation

Minimal model/experiment environment:

```bash
pip install -r requirements.txt
```

Real GIS preprocessing:

```bash
pip install -r requirements-geo.txt
```

Optional editable install:

```bash
pip install -e .
```

The scripts also add the repository root to `sys.path`, so editable installation is not required.

## 4. Immediate executable smoke test

```bash
python scripts/make_synthetic_benchmark.py \
  --out data/processed_synth --per-city 20

python scripts/run_main_experiment.py \
  --data data/processed_synth \
  --epochs 3 \
  --out results/main_smoke
```

Or run the full smoke-test suite:

```bash
bash scripts/run_all_synthetic.sh
```

## 5. Public source portals from the manuscript

```bash
python scripts/download_sources.py --show
```

The script prints:
- EUBUCCO v0.1 portal;
- World Settlement Footprint 3D portal;
- WorldPop Global1 age/sex structures portal.

These portals contain multiple products/files, so the repository does not guess which city archive or raster tile the user intends. Once a direct file URL is known:

```bash
python scripts/download_sources.py \
  --url "DIRECT_FILE_URL" \
  --out data/raw/source_file.ext
```

## 6. Build a real community graph

Example:

```bash
python scripts/build_city_graph.py \
  --buildings data/raw/barcelona/eubucco.gpkg \
  --city Barcelona \
  --community-id BCN_001 \
  --plan "install lifts in stair only blocks and improve insulation in older buildings" \
  --wsf-fraction data/raw/barcelona/wsf3d_fraction.tif \
  --wsf-height data/raw/barcelona/wsf3d_height.tif \
  --wsf-volume data/raw/barcelona/wsf3d_volume.tif \
  --worldpop-total data/raw/barcelona/worldpop_total.tif \
  --worldpop-65plus data/raw/barcelona/worldpop_65plus.tif \
  --target 0.18 0.14 0.11 0.08 \
  --spatial-group BCN_BLOCK_A \
  --out-dir data/processed
```

The `--target` values must come from real measured/simulated intervention outcomes in an empirical reproduction. Do not use the example values as evidence.

## 7. Main experiment

Per city:

```bash
python scripts/run_main_experiment.py \
  --data data/processed \
  --city Barcelona \
  --epochs 120 \
  --seed 7 \
  --out results/main_barcelona
```

The manuscript specifies seeds `7, 17, 27, 37, 47`; run the command once per seed and aggregate the output CSV for a five-run report.

## 8. Ablation study

```bash
python scripts/run_ablation.py \
  --data data/processed \
  --city Barcelona \
  --epochs 120 \
  --out results/ablation
```

Variants:
- `full`;
- `no_language_operator`;
- `no_consistency`;
- `static_attention`;
- `no_uncertainty`.

## 9. Robustness study

```bash
python scripts/run_robustness.py \
  --data data/processed \
  --city Barcelona \
  --epochs 120 \
  --out results/robustness
```

The corruption utility masks spatially contiguous feature blocks while preserving geometry and intervention identity.

## 10. Cross-city transfer

```bash
python scripts/run_cross_city.py \
  --data data/processed \
  --epochs 120 \
  --out results/cross_city
```

Each of Barcelona, Rotterdam and Vienna is held out in turn.

## 11. GraphGPS baseline

```bash
python scripts/run_graphgps_baseline.py \
  --data data/processed \
  --city Barcelona \
  --epochs 120 \
  --out results/baselines
```

`GraphGPSLite` is a compact same-feature graph baseline implemented in this repository. For exact official GraphGPS reproduction, substitute the official implementation/checkpoint while preserving the same graph split.

## 12. Foundation-model baselines

Large external models are intentionally **not** re-created with fake miniature substitutes. Extract official embeddings using each project's released code/checkpoint, then build an NPZ with:

- `embedding [M,D]`;
- `target [M,4]`;
- `split [M]`.

A helper converts a CSV manifest to NPZ:

```bash
python scripts/prepare_embedding_npz.py \
  --csv baseline_manifest.csv \
  --out anysat_embeddings.npz
```

Then train the common quantitative decoder:

```bash
python scripts/run_embedding_baseline.py \
  --method AnySat \
  --npz anysat_embeddings.npz \
  --out results/baselines
```

Supported names: SatMAE, ScaleMAE, UrbanCLIP, RemoteCLIP, GeoChat, UrbanLLM, AnySat.

## 13. Project structure

```text
care_geollm/
  config.py
  data.py
  tokenizer.py
  language.py
  model.py
  losses.py
  conformal.py
  metrics.py
  splits.py
  corruption.py
  baselines.py
  synthetic_data.py
  preprocess/
    common.py
    eubucco.py
    raster.py
    osm_support.py
    build_graph.py
  experiments/
    engine.py
    io.py
scripts/
  download_sources.py
  build_city_graph.py
  make_synthetic_benchmark.py
  run_main_experiment.py
  run_ablation.py
  run_robustness.py
  run_cross_city.py
  run_graphgps_baseline.py
  prepare_embedding_npz.py
  run_embedding_baseline.py
  run_all_synthetic.sh
configs/
  cities.example.json
```

## 14. Manuscript-to-code map

Algorithm step | Code
---|---
`H0 <- SpatialEncode(G0)` | `SpatialEncoder`
`Q <- LanguageAdapter(P)` | `PlanningLanguageAdapter`
`M <- EntityAlign(Q,H0)` | `EntityAlign`
`DeltaX <- DirectUpdates(Q,M)` | `DirectUpdate`
`DeltaTheta <- Hypernetwork(Q,M)` | `HyperNetwork`
factual GeoTransformer | `GeoTransformer(..., modulation=None)`
counterfactual GeoTransformer | localized low-rank modulation in `CAREGeoLLM`
`DeltaY <- YP-Y0` | `CAREGeoLLM.forward`
`I90 <- ConformalCalibrate(...)` | `ConformalCalibrator`

## 15. Notes for publication-quality reproduction

Before claiming numerical reproduction of the manuscript tables, freeze and publish:
- exact source file versions and timestamps;
- exact city/community polygons;
- CRS transformations;
- feature dictionary hash;
- treatment/intervention dictionary;
- intervention labels and outcome derivation;
- adjacency/path construction;
- five random seeds;
- official external baseline checkpoints and embedding extraction settings.

This repository is structured so those components can be added without changing the central CARE GeoLLM model.
