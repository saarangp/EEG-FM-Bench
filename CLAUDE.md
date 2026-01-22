# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG-FM-Bench is a comprehensive benchmark framework for evaluating EEG foundation models. It provides standardized preprocessing pipelines, training infrastructure, and visualization tools for comparing multiple foundation models across 14 EEG datasets spanning 10 paradigms.

## Instructions for claude
You won't have access to data etc, so I will run all the tests on my end. We can make changes, I'll push it and pull it from my remote server to test, and come back and discuss with you. 

If you want to check compile stuff you can activate the eeg2025 conda env. 

### Setup
```bash
pip install -r requirements.txt
```

### Preprocessing
```bash
# Preprocess datasets (after downloading from original sources)
python preproc.py conf_file=preproc/preproc_remote.yaml

# Config can be absolute path, relative path, or relative to CONF_ROOT (./assets/conf)
```

### Model Training
```bash
# Train/evaluate a model
python baseline_main.py conf_file=example/eegpt/eegpt.yaml model_type=eegpt

# List available models
python baseline_main.py list-models
```

### Visualization
```bash
# t-SNE embeddings
python visualize.py t_sne <model_config.yaml> <vis_config.yaml>

# Integrated gradients
python visualize.py integrated_gradients <model_config.yaml> <vis_config.yaml>
```

### HPC (SLURM)
```bash
sbatch slurm/preproc_submit.slurm conf_file=your_config.yaml
sbatch slurm/baseline_submit.slurm conf_file=your_config.yaml model_type=eegpt
```

## Architecture

### Configuration System
- Uses OmegaConf + Pydantic for hierarchical config management
- Config priority: code defaults < YAML file < CLI args
- Base classes in `common/config.py`: `AbstractConfig`, `BaseDataArgs`, `BaseModelArgs`, `BaseTrainingArgs`, `BaseLoggingArgs`
- Model-specific configs extend these in `baseline/<model>/<model>_config.py`

### Model Registry Pattern
Models are registered in `baseline/__init__.py` via `ModelRegistry.register_model()`:
```python
ModelRegistry.register_model(
    model_type='model_name',
    config_class=ModelConfig,
    adapter_class=ModelDataLoaderFactory,  # or None if no data conversion needed
    trainer_class=ModelTrainer
)
```

### Key Abstractions (`baseline/abstract/`)
- `AbstractTrainer`: Base trainer with distributed training, logging (WandB/Comet), checkpoint management, and metric calculation
- `AbstractDatasetAdapter`: Runtime data format conversion for models with specific input requirements
- `AbstractDataLoaderFactory`: Creates adapted dataloaders

### Data Pipeline (`data/`)
- **Processor** (`data/processor/`): EEG preprocessing using MNE (filtering, resampling, segmentation)
  - `EEGDatasetBuilder`: Base class for dataset preprocessing
  - `DATASET_SELECTOR` in `wrapper.py`: Maps dataset names to builder classes
- **Dataset** (`data/dataset/`): Dataset-specific configurations inheriting `EEGConfig`
  - Each file contains citation info, expected directory structure, montage definitions, and preprocessing parameters

### Path Configuration
Edit `common/path.py` before running:
```python
RUN_ROOT = './assets/run'           # Training outputs
LOG_ROOT = './assets/run/log'       # Logs
CONF_ROOT = './assets/conf'         # Config files
DATABASE_RAW_ROOT = './dataset'     # Raw downloaded data
DATABASE_PROC_ROOT = './arrow'      # Processed Arrow datasets
DATABASE_CACHE_ROOT = './cache'     # Preprocessing cache
```

## Supported Models

| Model Type | Config Class | Has Adapter | Description |
|------------|--------------|-------------|-------------|
| `eegpt` | `EegptConfig` | Yes | Dual self-supervised transformer |
| `labram` | `LabramConfig` | Yes | Vector quantized brain model |
| `cbramod` | `CBraModConfig` | Yes | Criss-cross attention |
| `bendr` | `BendrConfig` | No | Contrastive transformer |
| `biot` | `BiotConfig` | No | Cross-data biosignal learning |
| `eegnet` | `EegNetConfig` | No | Compact CNN baseline |
| `conformer` | `ConformerConfig` | No | Hybrid CNN-Transformer |

## Adding New Components

### New Model
1. Create `baseline/<model>/` with:
   - `<model>_config.py` extending `AbstractConfig`
   - `<model>_trainer.py` extending `AbstractTrainer`
   - `model.py` with the architecture
   - Optional: `<model>_adapter.py` for data format conversion
2. Register in `baseline/__init__.py`
3. Create example config in `assets/conf/example/<model>/`

### New Dataset
1. Create `data/dataset/<dataset>.py` implementing:
   - Config class extending `EEGConfig` with montage, preprocessing params
   - Builder class extending `EEGDatasetBuilder`
2. Add to `DATASET_SELECTOR` in `data/processor/wrapper.py`

## Key Dependencies

- `mne`: EEG signal processing
- `torch`: Deep learning framework (CUDA 12.4+)
- `datasets`: HuggingFace datasets for Arrow storage
- `omegaconf` + `pydantic`: Configuration management
- `wandb`: Experiment tracking
- `captum`: Integrated gradients visualization
