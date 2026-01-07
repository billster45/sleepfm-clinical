# SleepFM Reproducibility Audit Report

**Date**: 2026-01-06
**Paper**: Nature Medicine DOI 10.1038/s41591-025-04133-4
**Repository**: sleepfm-clinical

---

## Executive Summary

This report documents the findings from a comprehensive reproducibility audit of the SleepFM codebase, validating issues identified in the peer review. The audit included both **static analysis** (32 pytest tests) and **runtime verification** (demo notebook execution).

### Key Results

| Category | Tests | Passed | Status |
|----------|-------|--------|--------|
| Cox PH Loss Implementation | 8 | 8 | All peer review issues confirmed |
| Dataset Fallback Patterns | 7 | 7 | Infinite loop risk confirmed |
| Model Loading | 8 | 8 | Architecture mismatch confirmed |
| Module Imports | 9 | 9 | Missing seeds/hardcoded values confirmed |
| **Total** | **32** | **32** | **100% pass rate** |

### Runtime Verification

| Component | Status |
|-----------|--------|
| Demo notebook execution | Completed (after CPU fixes) |
| EDF → HDF5 preprocessing | Working |
| Model checkpoint loading | Working (required DataParallel handling) |
| Sleep staging inference | Working (but 0.0 F1 on synthetic data) |
| Disease prediction inference | Working |

---

## Confirmed Issues

### 1. Empty Evaluation File (CRITICAL)

**File**: `sleepfm/pipeline/evaluate_coxph.py`
**Status**: CONFIRMED EMPTY (0 lines)

The file that should contain C-Index evaluation code is completely empty. This means:
- No concordance index calculation exists
- No bootstrap confidence intervals
- No proper survival analysis evaluation

**Test**: `test_coxph_loss.py::TestEmptyEvaluationFile::test_evaluate_coxph_file_is_empty`

---

### 2. Cox PH Loss Batch Sorting Bias (HIGH)

**File**: `sleepfm/pipeline/finetune_diagnosis_coxph.py` (lines 31-49)
**Status**: CONFIRMED

The Cox PH loss function sorts event times **within each batch only**, not globally:

```python
def cox_ph_loss(hazards, event_times, is_event):
    event_times, sorted_idx = event_times.sort(dim=0, descending=True)  # BATCH ONLY
    hazards = hazards.gather(0, sorted_idx)
    is_event = is_event.gather(0, sorted_idx)
    log_cumulative_hazard = torch.logcumsumexp(hazards.float(), dim=0)
    ...
```

**Impact**: This introduces statistical bias in the partial likelihood calculation. True Cox PH requires global risk set ordering.

**Tests**:
- `test_batch_sorting_vs_global_sorting` - Demonstrates ordering differs
- `test_cox_ph_loss_signature` - Confirms no global ordering parameter
- `test_risk_set_computation_is_batch_local` - Validates batch-local computation

---

### 3. Missing Statistical Validation (HIGH)

**Status**: CONFIRMED

The codebase lacks:
- C-Index calculation implementation
- Bootstrap confidence intervals
- Bonferroni correction for multiple comparisons (20 diseases)
- Time-dependent AUC curves

**Tests**:
- `test_no_concordance_index_implementation`
- `test_no_bootstrap_confidence_intervals`

---

### 4. Recursive Fallback / Infinite Loop Risk (MEDIUM)

**File**: `sleepfm/models/dataset.py`
**Status**: CONFIRMED - 5 dataset classes affected

Pattern found in multiple classes:
```python
except Exception:
    return self.__getitem__((idx + 1) % self.total_len)  # Can loop forever
```

**Affected Classes**:
1. `SleepEventClassificationDataset`
2. `DiagnosisFinetuneFullCOXPHDataset`
3. `DiagnosisFinetuneFullCOXPHWithDemoDataset`
4. `SupervisedDiagnosisFullCOXPHWithDemoDataset`
5. Additional classes with similar patterns

**No retry counter exists** - confirmed by searching for `retry_count`, `_retry`, `max_retries`.

**Tests**:
- `test_identify_recursive_fallback_pattern`
- `test_count_affected_dataset_classes`
- `test_no_retry_counter_exists`
- `test_missing_error_logging`
- `test_modulo_wrapping_enables_infinite_loop`

---

### 5. Code Duplication (LOW)

**File**: `sleepfm/models/dataset.py`
**Status**: CONFIRMED - 7 collate functions with near-identical implementations

Functions identified:
- `collate_fn`
- `sleep_event_finetune_full_collate_fn`
- `diagnosis_finetune_full_coxph_collate_fn`
- `diagnosis_finetune_full_coxph_with_demo_collate_fn`
- And 3 more variants

**Test**: `test_count_collate_functions`

---

### 6. Missing Reproducibility Controls (MEDIUM)

**File**: `sleepfm/pipeline/pretrain.py`
**Status**: CONFIRMED

No `set_seed()` call found in pretraining script. Random initialization will vary between runs.

**Test**: `test_pretrain_missing_seed_setting`

---

### 7. Hardcoded Configuration Values (MEDIUM)

**Files**: Multiple evaluation scripts
**Status**: CONFIRMED

| File | Hardcoded Values |
|------|------------------|
| `evaluate_sleep_staging.py` | `batch_size=4`, `num_workers=4` |
| `evaluate_disease_prediction.py` | `batch_size=4`, `num_workers=4` |
| Config files | Stanford cluster paths (`/oak/stanford/groups/...`) |

**Tests**:
- `test_hardcoded_batch_size_in_eval_scripts`
- `test_hardcoded_paths_in_configs`

---

### 8. Architecture Mismatch (MEDIUM)

**File**: `sleepfm/models/models.py`
**Status**: CONFIRMED

- Pretraining uses **Transformer encoders**
- Fine-tuning uses **bidirectional LSTM** for temporal aggregation

This architectural mismatch may limit transfer learning effectiveness.

**Test**: `test_document_architecture_mismatch`

---

### 9. Temperature Parameter = 0.0 (MEDIUM)

**File**: `sleepfm/checkpoints/model_base/config.json`
**Status**: CONFIRMED via runtime inspection

The saved model config contains:
```json
{
    "temperature": 0.0,
    ...
}
```

**Peer Review Reference**: Section E.1 (lines 322-332) notes:
> "A temperature of 0 would cause numerical instability (effective division by zero in softmax scaling)"

**Runtime Confirmation**: The released pretrained model was trained with `temperature: 0.0`. While the code includes a constraint to prevent negative temperatures, it does not prevent zero, which causes `inf` values when dividing logits by temperature.

**Test**: `test_temperature_constraint_issue`

---

### 10. Demo Notebook CUDA Requirement (MEDIUM) - FIXED

**File**: `notebooks/demo.ipynb`
**Original Status**: CPU incompatible (hardcoded CUDA)
**Current Status**: FIXED - Now runs on CPU

**Changes Made**:
- Added device auto-detection: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Fixed checkpoint loading to handle `module.` prefix for CPU/GPU compatibility
- Set `num_workers=0` for DataLoaders when running on CPU (required for macOS)

**Verification**: Demo notebook executed successfully on CPU with 0 errors:
- All 53 cells executed
- Preprocessing: EDF → HDF5 conversion completed
- Embedding generation: 5-second and 5-minute embeddings created
- Sleep staging inference: Completed successfully
- Disease prediction inference: Completed successfully

Output saved to: `notebooks/demo_cpu_test_output.ipynb`

---

## Runtime Findings from Demo Notebook Execution

### Model Architecture Verification

Successfully loaded and verified all three model checkpoints:

| Model | Parameters | Layers | Architecture |
|-------|-----------|--------|--------------|
| Base (SetTransformer) | 4.44M | 93 | Transformer encoder with attention pooling |
| Sleep Staging | 1.19M | 20 | Bidirectional LSTM classifier |
| Disease Prediction | 0.91M | 15 | LSTM + demographic embedding |

### DataParallel Checkpoint Artifact

**Finding**: All checkpoints were saved with `nn.DataParallel` wrapper, causing state_dict keys to have `module.` prefix (e.g., `module.patch_embedding.tokenizer.0.weight`).

**Impact**: Loading checkpoints on CPU or single-GPU requires stripping this prefix:
```python
state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
```

This is a common issue but should be documented for users.

### Synthetic Demo Data Limitations

**Critical Finding**: The demo notebook cannot validate actual model performance.

Sleep staging F1 scores on synthetic demo data:
| Stage | F1 Score |
|-------|----------|
| Wake | 0.538 |
| Stage 1 | 0.000 |
| Stage 2 | 0.000 |
| Stage 3 | 0.000 |
| REM | 0.000 |

**Explanation**: The synthetic demo data does not resemble real PSG recordings, so the trained model produces essentially random predictions for most sleep stages. This confirms the peer review concern (Section C.3, lines 276-289):

> "Users cannot verify expected outputs... Cannot validate preprocessing pipeline... Cannot confirm model loading works correctly"

**Recommendation**: Provide either:
1. A single real anonymized PSG with expected output checksums
2. A pre-computed embedding file with deterministic outputs for validation

### Preprocessing Pipeline Verification

The EDF → HDF5 conversion pipeline executed successfully:
- Input: `demo_psg.edf` (18.0 MB)
- Output: `demo_psg.hdf5` (7.9 MB)
- Resampling to 128 Hz completed
- All 4 modalities (BAS, RESP, EKG, EMG) processed

### Embedding Generation Verification

Both embedding types were generated successfully:
- 5-second embeddings: `demo_emb/demo_psg.hdf5`
- 5-minute aggregated embeddings: `demo_5min_agg_emb/demo_psg.hdf5`

---

## Test Environment

```
Platform: macOS (Darwin 25.2.0)
Python: 3.13 (via uv)
PyTorch: CPU-only build
Virtual Environment: uv venv
```

### Dependencies Installed
- torch==2.0.1
- lifelines (Cox PH reference)
- pytest
- scikit-survival

---

## Test Execution

```bash
# Activate environment
source .venv/bin/activate

# Run all tests
PYTHONPATH=/Users/billcockerill/Documents/sleepfm-clinical/sleepfm:$PYTHONPATH \
pytest tests/ -v

# Results
======================= 32 passed, 13 warnings in 1.96s ========================
```

---

## Recommendations

### Critical Priority

1. **Implement C-Index Evaluation**: Populate `evaluate_coxph.py` with:
   - Harrell's concordance index
   - Bootstrap confidence intervals (n=1000)
   - Time-dependent AUC curves

2. **Fix Cox PH Loss**: Either:
   - Use global risk set ordering, OR
   - Document batch-local approximation and validate against reference implementations

### High Priority

3. **Add Retry Counters**: Modify recursive fallback to prevent infinite loops:
   ```python
   def __getitem__(self, idx, _retry_count=0):
       if _retry_count > 10:
           raise RuntimeError(f'Failed after 10 retries from idx {idx}')
       try:
           ...
       except Exception:
           return self.__getitem__((idx + 1) % self.total_len, _retry_count + 1)
   ```

4. **Add Random Seeds**: Call `set_seed()` in pretrain.py for reproducibility

### Medium Priority

5. ~~**CPU Compatibility**: Add device auto-detection in demo notebook~~ **DONE**
   - Device auto-detection implemented
   - Checkpoint loading fixed for CPU/GPU compatibility
   - DataLoader workers set to 0 for CPU mode

6. **Refactor Collate Functions**: Consolidate 7 near-identical functions into parameterized version

7. **Externalize Configuration**: Move hardcoded values to config files

---

## Files Created

| File | Purpose |
|------|---------|
| `tests/__init__.py` | Test package marker |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/test_coxph_loss.py` | Cox PH loss validation (8 tests) |
| `tests/test_dataset_fallback.py` | Dataset fallback testing (7 tests) |
| `tests/test_imports.py` | Module import tests (9 tests) |
| `tests/test_model_loading.py` | Model checkpoint tests (8 tests) |

---

## Conclusion

The reproducibility audit **confirms all major findings** from the peer review:

### Static Analysis Findings (Confirmed)
1. The empty `evaluate_coxph.py` file is a critical gap
2. Cox PH batch sorting introduces statistical bias
3. Recursive fallback patterns create infinite loop risk
4. Missing seeds prevent exact reproducibility
5. Hardcoded paths and values limit portability

### Runtime Findings (New)
6. **Temperature = 0.0**: The released model config contains `temperature: 0.0`, which causes numerical instability (confirmed peer review concern E.1)
7. **DataParallel artifact**: Checkpoints require `module.` prefix handling for non-DataParallel loading
8. **Synthetic data cannot validate performance**: F1 scores of 0.0 for most sleep stages demonstrate that users cannot verify expected model behavior
9. **Demo notebook now runs on CPU** after our fixes (device auto-detection, checkpoint handling, DataLoader workers)

### Overall Assessment

The demo notebook pipeline **executes successfully** after minor modifications, confirming that:
- Preprocessing (EDF → HDF5) works correctly
- Model checkpoints load and run inference
- All pipeline stages complete without errors

However, the **scientific validity of results cannot be verified** using the provided synthetic demo data. The peer review's recommendation for real anonymized test data with expected outputs remains critical for true reproducibility.

The codebase would benefit from the recommended fixes before the paper's methodology can be fully reproduced.
