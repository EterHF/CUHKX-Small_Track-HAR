# CUHK-X Small Model Track — Compact Visual–Temporal Fusion

[English](README.md) | [简体中文](README.zh-CN.md)

Source code and technical report for team **YE Mao1023**, recipient of the Grand Finals Second Prize in the CUHK-X Challenge 2026 Small Model Track. The submitted system combines Depth_Color and infrared video features with skeleton-based temporal recognition. Submission **56036959** achieved **97.512% public** and **96.568% private** leaderboard accuracy. Its frozen inference bundle, including the detector, occupies **50,733,469 bytes**.

## Technical report

[Technical report (PDF)](docs/public/TECHNICAL_REPORT.pdf)

The report describes the visual branch, skeleton frame classifier, residual temporal decoder and confidence-weighted fusion. Sources are listed in **Section 4**, training and validation settings in **Section 5**, and release contents and inference requirements in **Section 8**.

This repository contains training and inference source, configurations and the report. Model checkpoints and competition data are not bundled with this release. Running inference requires the checkpoint identified in `final_config.json`. Initialization requirements and historical reproduction results are documented in the report.

## Code layout

| Path | Contents |
| --- | --- |
| `inference.sh`, `final_config.json` | Submitted-model inference entry point and artifact identity |
| `visual_temporal/data/` | Input indexing, preprocessing and caches |
| `visual_temporal/models/` | Visual and skeleton architectures; shared fusion components |
| `visual_temporal/training/` | Visual adaptation and temporal decoder training |
| `visual_temporal/release/` | Branch inference, fusion, packing and raw-input replay |
| `docs/public/TECHNICAL_REPORT.pdf` | Method, recorded results, sources and reproduction settings |

## Setup and inference

The recorded Linux GPU environment uses Python 3.11 and CUDA 12.9 wheels:

```bash
python -m pip install -r requirements-inference.txt
python -m pip install -e . --no-deps
```

Obtain competition data through the [Small Model Track dataset](https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track). For the download helper, install the CLI with `python -m pip install -U huggingface_hub`. After accepting the dataset's access conditions and authenticating with `hf auth login`, run `bash download_all_data.sh` to download the archives into `data/Small-Model-Track/`.

Place the corresponding checkpoint at the path in `final_config.json` and prepare the following input layout:

```text
Testing/
  test_file/test.csv
  test_file/sample_submission.csv
  data/small_model_track_test/<sample_id>/...
```

The official `Testing/data/small_model_track_test.zip` layout is also accepted.

```bash
bash inference.sh /absolute/path/to/Testing /absolute/path/to/prediction.csv
```

Output must be outside the input data directory. `CUHKX_PYTHON` selects Python; `CUHKX_DEVICE` selects the CUDA device (default `cuda:0`). The entry point checks checkpoint size before processing the input.

## Training implementation

Training requires authorized training data and the original initializations. Keep `HAR.z01`–`HAR.z08` beside `HAR.zip`; inside `data/Small-Model-Track/Training/data/`, run `zip -s 0 HAR.zip --out HAR_full.zip` followed by `unzip HAR_full.zip`.

| Stage | Implementation under `visual_temporal/` |
| --- | --- |
| Indexing, cropping, normalization and caches | `data/` |
| Visual classifier-head adaptation | `training/public_finetune.py` |
| Frozen skeleton frame classifier | `models/public_dstformer.py`, `models/dstformer.py` |
| Residual TCN and subject-fold training | `training/temporal.py`, `training/temporal_suite.py` |
| Confidence fusion and checkpoint packaging | `release/blend.py`, `release/prune_fusion.py`, `release/bundle.py` |

Inspect supported arguments with:

```bash
python -m visual_temporal.training.public_finetune --help
python -m visual_temporal.training.temporal_suite --help
```

Report Sections 3–5 describe the submitted settings and recorded training results. The training tools expose additional options; their defaults are not the submitted recipe. Prepare the required data and initializations for each stage; evaluate newly trained models under their own experimental settings.

## Licenses

Project-authored code uses [Apache 2.0](LICENSE), and original report text and figures use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party components retain their applicable terms; see [NOTICE](NOTICE) for copyright and license notices.
