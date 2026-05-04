# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is EfficientOCF with query-based auxiliary occupancy forecasting (QPAGC), built on top of mmdetection3d. It performs vision-based 3D occupancy forecasting from multi-view camera images on the nuScenes dataset. The model predicts future occupancy states, instance flow, and height maps using a spatiotemporal decoupling approach, extended with a query-based instance prediction module.

## Commands

### Training

```bash
# Distributed training (main entry point)
bash run.sh <config_path> <num_gpus> [extra_args...]

# Example: train query main model on 2 GPUs
bash run.sh projects/configs/baselines/EfficientOCF_V1.1_query_main.py 2

# LSS view-transform pretraining (pretrain_view_transform_only=True)
bash run_lss_pretrain.sh [config] [gpus] [extra_args...]

# Override config options at runtime
bash run.sh projects/configs/baselines/EfficientOCF_V1.1_query_main.py 2 --cfg-options model.freeze_lss_pretrained=True
```

### Evaluation

```bash
# Distributed evaluation
bash run_eval.sh <config_path> <checkpoint_path> <num_gpus>

# Example
CUDA_VISIBLE_DEVICES=0 bash run_eval.sh \
  projects/configs/baselines/EfficientOCF_V1.1_query_main.py \
  work_dirs/epoch_15.pth 1
```

### Data Preprocessing

```bash
# Precompute distance transform fields (required before training with DT loss)
cd playground
python precompute_dt_full.py --base_dir <project_root> \
  --ann_files <train_pkl> <val_pkl> \
  --time_receptive_field 3 --n_future_frames 4 \
  --save_as_gmo_style --save_mode uint8 \
  --out_dir <output_dir>

# Generate depth ground truth
python tools/gen_data/gen_depth_gt.py
```

## Architecture

### Model Pipeline (EfficientOCF detector)

The main model class is `EfficientOCF` in `projects/occ_plugin/occupancy/detectors/efficientocf.py`. It inherits from `BEVDepth` and mixes in several mixin classes for modularity:

1. **Image backbone + neck** (ResNet18 + SECONDFPN): Extract multi-view image features
2. **View transformer** (`ViewTransformerLiftSplatShootVoxel`): Lift-Splat-Shoot to project 2D image features into 3D BEV voxel features with depth estimation
3. **Temporal aggregation** (`TransformerModule`): Cross-attention module that aggregates multi-camera, multi-timestep BEV features into query-based instance representations
4. **Three parallel encoder-predictor-neck branches** (each: CustomResNet2D + Predictor + FPN):
   - **Occupancy branch** -> `OccHead`: predicts voxel occupancy
   - **Height branch** -> `HeightHead`: predicts height maps
   - **Flow branch** -> `FlowHead`: predicts 2D instance flow
5. **Query head** (`QueryHead`): Learnable queries predict instance centers, Gaussian spreads, classification, and objectness scores. Uses Hungarian matching for GT assignment.

### Key Mixin Classes (in `detectors/`)

- `EfficientOCFLossMixin` (`utils_loss.py`): All loss computations (GMO BCE, DT loss, query losses, gt2p labeled loss)
- `EfficientOCFMatcherMixin` (`utils_matcher.py`): Hungarian matching between predicted queries and GT instances
- `EfficientOCFBEVPoolMixin` (`utils_bev_pool.py`): Gaussian-based BEV pooling of query features
- `EfficientOCFGTPrepMixin` (`utils_gt_prep.py`): Ground truth preparation for query supervision
- `EfficientOCFVisualizationMixin` (`utils_visualization.py`): Debug visualization utilities
- `EfficientOCFDebugMixin` (`utils_dbg.py`): Debug logging and gradient monitoring
- `EfficientOCFInstanceImgDebugMixin` (`utils_instance_img_debug.py`): Instance-level image debug vis

### Plugin System

The project uses mmdetection3d's plugin mechanism. Setting `plugin=True` and `plugin_dir="projects/occ_plugin/"` in config files auto-registers all custom modules (detectors, heads, backbones, datasets, pipelines).

### Config Structure

- `projects/configs/baselines/`: Main experiment configs (e.g., `EfficientOCF_V1.1_query_main.py`)
- `projects/configs/_base_/`: Base runtime and schedule configs
- `projects/configs/datasets/`: Dataset-specific config (`custom_nus-3d.py`)
- Configs are mmcv-style Python files that support inheritance via `_base_`

### Data Pipeline

Custom pipeline transforms in `projects/occ_plugin/datasets/pipelines/`:
- `loading_instance.py` (`LoadInstanceWithFlow`): Loads instance segmentation and flow data
- `loading_bevdet.py` (`LoadMultiViewImageFromFiles_BEVDet`): Multi-view image loading with augmentation
- `loading_occupancy.py` (`LoadOccupancy`): Loads occupancy GT, height maps, and precomputed DT fields
- `transform_3d.py`: Image augmentation transforms

### Dataset Symlinks

`data/` contains symlinks to external dataset paths:
- `nuscenes` -> nuScenes raw data
- `nuScenes-Occupancy` -> Occupancy GT labels
- `efficientocf` -> Preprocessed OCF instance/flow data
- `efficientocf_bboxcls` -> Bbox-based class segmentation data
- `occ_dt` -> Precomputed distance transform fields

### Custom Ops

`projects/occ_plugin/ops/occ_pooling/`: Custom CUDA occupancy pooling operation used by the view transformer.

### Two-Stage Training

The typical workflow is:
1. **Stage 1**: Pretrain the LSS view transformer using `run_lss_pretrain.sh` (sets `model.pretrain_view_transform_only=True`, only trains image backbone + neck + view transformer with depth supervision)
2. **Stage 2**: Full model training using `run.sh` with `lss_pretrained_ckpt_path` pointing to the Stage 1 checkpoint

### Semantic Class Mapping

Foreground classes use sparse nuScenes occupancy IDs: `[2, 3, 4, 5, 6, 7, 9, 10]` mapped to `['bicycle', 'bus', 'car', 'construction', 'motorcycle', 'pedestrian', 'trailer', 'truck']`. The query classifier uses a compact contiguous label space: 0=background, then the 8 foreground classes.

### Gradient Accumulation

Training uses `GradientCumulativeOptimizerHook` with `cumulative_iters=8` to simulate larger batch sizes on limited GPU memory.


### Instruction

Token and Context Efficiency
- Minimize token usage at all times. Avoid verbose explanations unless the topic is genuinely complex or the user explicitly asks for detail.
- Do not repeat information already established in the conversation. Assume context carries forward.
- Keep responses concise and direct. Prefer substance over elaboration.

File and Codebase Exploration
- Avoid browsing files or directories speculatively. Only perform broad exploration when it is strictly necessary and clearly justified.
- When reading files, target only the specific file or section relevant to the task.

Code Modification Policy
- Never edit, delete, or overwrite any file or code without an explicit request from the user.
- When proposing changes, present the diff or the intended modification first and wait for approval if the scope is non-trivial.

Code Quality and Readability
- Write code that is simple, clear, and easy to read. Prioritize readability over cleverness.
- Keep files short. Avoid unnecessary abstractions, boilerplate, and redundant structure.
- Do not add comments that merely restate what the code already clearly expresses.
Avoid premature generalization or over-engineering beyond what the task requires.

Python-Specific Conventions
- Do not litter code with assert statements or debug prints unless they are genuinely essential to correctness or safety.
- Avoid defensive programming patterns that bloat the code without adding real value.
Do not add type hints, docstrings, or logging scaffolding unless asked.

Ambiguity and Decision Gates
- If a task requires a design decision that would significantly affect implementation direction, stop and ask the user before writing any code, regardless of whether a "Plan mode" is active.
- If the intended behavior, architecture, or scope is unclear, surface the ambiguity explicitly with a focused question rather than making assumptions and proceeding.
Prefer one precise question over multiple vague ones.

Minimal diff principle: When modifying existing code, change only what is necessary. Do not reformat, rename, or restructure surrounding code unless that is the explicit goal.

No unsolicited refactoring: Do not refactor or reorganize code beyond the stated scope of a task, even if improvements seem obvious.

Assumption transparency: If a task requires making a non-trivial assumption to proceed, state the assumption explicitly at the top of the response rather than embedding it silently in the implementation.
