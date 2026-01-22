# EEG-FM-Bench Changelog

Working document tracking changes made during development sessions.

---

## 2026-01-22: Data Efficiency Evaluation Feature

### Summary
Added percentage-based data efficiency evaluation with multiple episodes for evaluating models on varying amounts of training data (10%, 25%, 50%, 75%, 100%) with stratified sampling.

### Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `common/config.py` | Modified | Added `DataEfficiencyArgs` class and `data_efficiency` field to `AbstractConfig` |
| `data/processor/efficiency.py` | **New** | Stratified subsampling utility using sklearn's `train_test_split` |
| `data/processor/__init__.py` | Modified | Added import for `stratified_subsample` |
| `baseline/abstract/adapter.py` | Modified | Added `subsample_fraction` and `subsample_seed` params to `loading_dataset()` and `create_dataloader()` |
| `baseline/abstract/trainer.py` | Modified | Added numpy import, updated `create_single_dataloader()`, added data efficiency check in `run_separate_training()`, added `_run_data_efficiency_evaluation()`, `_eval_epoch_with_return()`, `_save_efficiency_results()` methods |

### New Config Options
```yaml
data_efficiency:
  enabled: true                              # Enable data efficiency mode
  fractions: [0.1, 0.25, 0.5, 0.75, 1.0]    # Training data fractions
  num_episodes: 5                            # Number of episodes per fraction
  seed_base: 42                              # Base seed for reproducibility
```

### Output
- Results saved to `{output_dir}/data_efficiency_results.csv`
- Console output shows mean±std balanced accuracy per fraction

### Key Implementation Details
- Stratified sampling maintains class distribution
- Each episode loads fresh pretrained weights
- Val/test sets remain full (only training subsampled)
- Works with single-task (separate) training only

---
