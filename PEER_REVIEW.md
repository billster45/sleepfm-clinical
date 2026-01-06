# Comprehensive Peer Review

## Paper: "A Multimodal Sleep Foundation Model for Disease Prediction"
**Journal:** Nature Medicine  
**DOI:** 10.1038/s41591-025-04133-4  
**Review Date:** January 6, 2026

---

## Sources Reviewed

| Source | Link |
|--------|------|
| **Published Paper** | https://www.nature.com/articles/s41591-025-04133-4 |
| **Code Repository** | https://github.com/zou-group/sleepfm-clinical/tree/sleepfm_release |
| **Author Announcement** | https://x.com/james_y_zou/status/2008576712695836685 |

### Review Methodology

This peer review was conducted through:
1. **Paper Analysis**: Full review of the published Nature Medicine article
2. **Static Code Review**: Comprehensive examination of the codebase including model architecture, training pipelines, evaluation scripts, preprocessing, and configuration files
3. **Documentation Review**: Assessment of README, demo notebook, and inline documentation

> ⚠️ **Important Note**: The code was reviewed statically (read and analyzed) but was **not executed**. Findings related to runtime behavior are based on code inspection and may require empirical validation.

---

## Executive Summary

This paper presents **SleepFM**, a multimodal foundation model for sleep analysis trained on ~585,000 hours of polysomnography (PSG) data from ~65,000 participants. The model employs a novel contrastive learning approach that accommodates multiple PSG montages (BAS, RESP, EKG, EMG modalities) and demonstrates strong performance on disease prediction (130 conditions with C-Index ≥0.75), sleep staging (F1: 0.70-0.78), and transfer learning to external datasets (SHHS).

**Overall Assessment: Accept with Minor Revisions**

---

## Key Strengths

### 1. Significant Clinical Impact and Scale
- The training scale (~585K hours from ~65K participants) represents one of the largest sleep foundation models to date
- Impressive predictive performance for clinically important outcomes:
  - All-cause mortality (C-Index: 0.84)
  - Dementia (C-Index: 0.85)
  - Myocardial infarction (C-Index: 0.81)
  - Heart failure (C-Index: 0.80)
  - Chronic kidney disease (C-Index: 0.79)
  - Stroke (C-Index: 0.78)
  - Atrial fibrillation (C-Index: 0.78)
- Coverage of 130 conditions with significant predictive power demonstrates broad clinical utility

### 2. Novel Technical Contributions

#### Montage-Agnostic Design
The architecture elegantly handles variable channel configurations across datasets through the `SetTransformer` with attention-based spatial pooling:

```python
# From sleepfm/models/models.py
class SetTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_heads, num_layers, ...):
        self.patch_embedding = Tokenizer(input_size=patch_size, output_size=embed_dim)
        self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head)
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.temporal_pooling = AttentionPooling(embed_dim, num_heads=pooling_head)
```

#### Multi-Modal Contrastive Learning
Both "pairwise" and "leave-one-out" contrastive modes allow flexible alignment between modalities, enabling the model to learn cross-modal relationships without requiring all modalities to be present.

#### Hierarchical Embeddings
The `Tokenizer` class creates both 5-second granular and 5-minute aggregated embeddings, enabling multi-scale temporal analysis appropriate for different downstream tasks.

### 3. Strong External Validation
- Transfer learning to SHHS (excluded from pretraining) demonstrates genuine generalization capability
- Competitive performance with specialized models (U-Sleep, YASA) validates the foundation model paradigm for sleep analysis

### 4. Reproducibility Efforts
- Public release of Stanford Sleep Dataset on BDSP (https://bdsp.io/content/08vg8vqv2wdtwonc1ddy/1.0)
- Model checkpoints provided for:
  - Base pretrained model (`sleepfm/checkpoints/model_base`)
  - Disease prediction model (`sleepfm/checkpoints/model_diagnosis`)
  - Sleep staging model (`sleepfm/checkpoints/model_sleep_staging`)
- End-to-end demo notebook with synthetic data (`notebooks/demo.ipynb`)

### 5. Well-Designed Architecture
The processing pipeline follows a logical flow:
```
Raw PSG → Patch Embedding → Spatial Pooling (across channels) → 
Positional Encoding → Transformer Encoder → Temporal Pooling → Embedding
```
This handles the fundamental challenge of variable-length, variable-channel PSG recordings.

---

## Areas for Improvement

### A. Methodological Concerns

#### 1. Data Leakage Risk in Time-to-Event Analysis (High Priority)

The Cox proportional hazards implementation sorts events by time within batches:

```python
# From sleepfm/pipeline/finetune_diagnosis_coxph.py
def cox_ph_loss(hazards, event_times, is_event):
    event_times, sorted_idx = event_times.sort(dim=0, descending=True)
    hazards = hazards.gather(0, sorted_idx)
    is_event = is_event.gather(0, sorted_idx)
    log_cumulative_hazard = torch.logcumsumexp(hazards.float(), dim=0)
    losses = (hazards - log_cumulative_hazard) * is_event
    ...
```

**Concern:** Cox PH loss requires comparing subjects at risk at each event time. Sorting within batches (rather than globally) introduces statistical bias when subjects from different risk sets appear in different batches. This is a known issue in deep survival analysis literature.

**Recommendation:** Consider Breslow or Efron approximations with proper at-risk set handling, or validate against existing implementations (e.g., `pycox`, `lifelines`).

#### 2. Missing Statistical Rigor in Evaluation (High Priority)

The evaluation file `sleepfm/pipeline/evaluate_coxph.py` is **empty**. This is a significant gap for reproducibility:
- No C-Index calculation code provided
- No confidence interval estimation
- No calibration assessment (e.g., calibration plots)
- No comparison with null models (e.g., age/sex only baseline)

**Recommendation:** Provide complete evaluation code including:
- Bootstrap confidence intervals for C-Index
- Calibration plots (predicted vs. observed risk)
- Brier scores for probabilistic calibration
- Time-dependent AUC curves

#### 3. Potential Confounding in Disease Prediction (Medium Priority)

The diagnosis model concatenates demographic features (age, sex) with PSG embeddings:

```python
# From sleepfm/models/models.py
class DiagnosisFinetuneFullLSTMCOXPHWithDemo(nn.Module):
    def forward(self, x, mask, demo_features):
        ...
        demo_embed = self.demo_embedding(demo_features)
        x = torch.cat([x, demo_embed], dim=1)
        hazards = self.disease_heads(x)
        return hazards
```

**Concern:** For conditions where demographic features are strong predictors (e.g., cardiovascular disease with age), the model may rely disproportionately on demographics rather than PSG signals. While the paper includes a `DiagnosisFinetuneDemoOnlyEmbed` baseline, the ablation results should be more prominently reported.

**Recommendation:** 
- Report feature attribution analysis (e.g., integrated gradients, SHAP) to quantify PSG vs. demographic contributions for each condition
- Include scatter plots of demo-only vs. full model C-Index for all 130 conditions

### B. Code Quality Issues

#### 1. Inconsistent Error Handling

Multiple dataset classes use recursive fallback on errors:

```python
# From sleepfm/models/dataset.py (appears multiple times)
if not x_data:
    return self.__getitem__((idx + 1) % self.total_len)
```

**Concern:** If many consecutive samples are corrupted, this could:
- Loop indefinitely
- Skip large portions of data silently
- Cause memory issues from deep recursion

**Recommendation:** Add maximum retry counter and proper logging:
```python
def __getitem__(self, idx, _retry_count=0):
    if _retry_count > 10:
        raise RuntimeError(f"Failed to load data after 10 retries starting from idx {idx}")
    ...
    if not x_data:
        logger.warning(f"Empty data at index {idx}, skipping")
        return self.__getitem__((idx + 1) % self.total_len, _retry_count + 1)
```

#### 2. Hardcoded Values in Critical Paths

Several hardcoded values reduce reproducibility:

```python
# sleepfm/pipeline/evaluate_sleep_staging.py lines 82-83
num_workers = 4  # Overrides config
batch_size = 4   # Overrides config
```

```yaml
# sleepfm/configs/config_set_transformer_contrastive.yaml
data_path: '/scratch/users/rthapa84/psg_fm/data/data_new_128/'  # Absolute path
```

**Recommendation:** 
- Remove hardcoded overrides in evaluation scripts
- Use environment variables or relative paths for data locations
- Add path validation with helpful error messages

#### 3. Missing Type Hints and Documentation

Most functions lack type hints and docstrings:

```python
def run_iter(batch, num_modalities, model, device, mode, temperature, batch_size, ij):
    # No docstring explaining parameters, return values, or expected shapes
    batch_data, mask_list, *_ = batch
    ...
```

**Recommendation:** Add comprehensive documentation:
```python
def run_iter(
    batch: Tuple[List[torch.Tensor], List[torch.Tensor], ...],
    num_modalities: int,
    model: nn.Module,
    device: torch.device,
    mode: Literal["pairwise", "leave_one_out"],
    temperature: torch.Tensor,
    batch_size: int,
    ij: Tuple[Tuple[int, int], ...]
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single training/validation iteration for contrastive learning.
    
    Args:
        batch: Tuple of (data_list, mask_list, file_paths, ...)
        num_modalities: Number of modalities (typically 4: BAS, RESP, EKG, EMG)
        ...
    
    Returns:
        loss: Scalar loss tensor
        pairwise_loss: Array of shape (num_modalities, num_modalities) or (num_modalities, 2)
        ...
    """
```

#### 4. Duplicate Code in Collate Functions

The `collate_fn` implementations in `dataset.py` are nearly identical with minor variations:
- `collate_fn`
- `sleep_event_finetune_full_collate_fn`
- `diagnosis_finetune_full_coxph_collate_fn`
- `diagnosis_finetune_full_coxph_with_demo_collate_fn`

**Recommendation:** Refactor to a base collate function with optional parameters.

### C. Reproducibility Gaps

#### 1. Incomplete Preprocessing Documentation

The channel mapping in `sleepfm/configs/channel_groups.json` lists 844 lines of channel names, but:
- No documentation of which datasets use which channels
- No handling of conflicting channel names across datasets
- The mapping logic doesn't validate channel consistency

**Recommendation:** Add a preprocessing documentation file mapping:
- Dataset → Expected channels
- Channel name variations (e.g., "EEG", "EEG1", "EEG 2")
- Validation checks for channel compatibility

#### 2. Missing Seeds in Critical Locations

While `set_seed()` is called in fine-tuning scripts:
```python
def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ...
```

The pretraining script (`pretrain.py`) does **not** set random seeds, affecting reproducibility of the foundation model itself.

#### 3. Demo Notebook Uses Synthetic Data

The demo notebook explicitly states:
> "All data in this repo is synthetically made including sleep stage annotations, demographics or diseases. The data is for demo purposes only."

While understandable for privacy, this means:
- Users cannot verify expected outputs
- Cannot validate preprocessing pipeline
- Cannot confirm model loading works correctly

**Recommendation:** Consider providing:
- A single real anonymized PSG (with consent)
- Expected output checksums for synthetic data
- Unit tests with deterministic outputs

### D. Statistical and Evaluation Concerns

#### 1. Multiple Comparison Correction

The paper reports Bonferroni-corrected p-values for 130 conditions, but:
- Correction code not provided
- Given comorbidity correlation structure, Bonferroni may be overly conservative
- Consider Benjamini-Hochberg FDR control as alternative

#### 2. Class Imbalance Handling

The sleep staging collate function contains unusual preprocessing:

```python
# From notebooks/demo.ipynb
moving_avg_tgt_sleep_no_sleep = np.convolve(tgt_sleep_no_sleep, np.ones(1080)/1080, mode='valid')
first_non_zero_index = np.where(moving_avg_tgt_sleep_no_sleep > 0.5)[0][0]
```

This finds where sleep begins and truncates wake periods, potentially introducing bias by removing challenging wake/sleep transitions.

#### 3. No Uncertainty Quantification

The model outputs point predictions for hazard ratios without uncertainty estimates. For clinical deployment:
- Confidence intervals are essential
- Consider MC dropout or ensemble methods
- Report prediction intervals, not just point estimates

### E. Architecture Considerations

#### 1. Temperature Parameter Constraints

The learnable temperature is constrained only to be non-negative:
```python
if temperature < 0:
    with torch.no_grad():
        temperature.fill_(0)
```

A temperature of 0 would cause numerical instability (effective division by zero in softmax scaling).

**Recommendation:** Add minimum threshold: `temperature.clamp_(min=0.01)`

#### 2. LSTM vs. Transformer Architectural Mismatch

- Pretraining uses Transformer encoders
- Fine-tuning uses bidirectional LSTM for temporal aggregation

This architectural mismatch may limit transfer learning effectiveness. Consider using the same architecture throughout or providing ablations.

---

## Minor Issues

| Issue | Location | Recommendation |
|-------|----------|----------------|
| Commented-out code | Throughout codebase | Remove or document |
| TODO comments in production | `pretrain.py` line 38 | Resolve or track in issues |
| Inconsistent naming | Mix of snake_case/camelCase | Standardize to PEP 8 |
| Version pinning | `requirements.txt` | Good, but document known conflicts |
| Unused imports | Multiple files | Run linter to clean up |

---

## Questions for Authors

1. **Data Splitting:** How was the train/validation/test split performed? Was it subject-level to prevent data leakage across splits?

2. **Feature Attribution:** What is the breakdown of the C-Index improvement from PSG features vs. demographic features alone for each of the 130 conditions?

3. **Subgroup Analysis:** How does model performance vary across demographic subgroups (age, sex, race)? Are there equity concerns?

4. **Class Imbalance:** Were any techniques used to handle the extreme class imbalance for rare conditions?

5. **Computational Cost:** What is the inference time per PSG recording? What hardware is required for deployment?

6. **Temporal Alignment:** How were PSG recordings aligned with outcome dates? What was the minimum follow-up period required?

---

## Recommended Priority Actions

| Priority | Issue | Suggested Fix |
|----------|-------|---------------|
| 🔴 **High** | Empty `evaluate_coxph.py` | Add complete C-Index calculation with confidence intervals |
| 🔴 **High** | Cox PH batch sorting bias | Validate implementation against `pycox` or similar |
| 🟡 **Medium** | Missing feature attribution | Add integrated gradients or SHAP analysis |
| 🟡 **Medium** | Hardcoded paths/values | Use config files and environment variables |
| 🟡 **Medium** | Missing seeds in pretraining | Add `set_seed()` call with configurable seed |
| 🟢 **Low** | Code documentation | Add type hints and docstrings |
| 🟢 **Low** | Duplicate collate functions | Refactor to shared base implementation |

---

## Conclusion

**SleepFM represents a significant contribution** to computational sleep medicine and foundation models for healthcare. The scale of data, elegant handling of variable montages, and impressive predictive performance across 130 conditions are noteworthy achievements deserving publication in a high-impact venue.

The issues identified are primarily implementation and documentation concerns that can be addressed without changing the fundamental scientific claims. Providing complete evaluation code and clearer ablation studies quantifying PSG vs. demographic contributions would significantly strengthen reproducibility and clinical interpretability.

### Final Recommendation: **Accept with Minor Revisions**

The core scientific contribution is sound and impactful. Addressing the high-priority items above would make this an exemplary open-source release for a Nature Medicine publication.

---

*Review conducted on January 6, 2026*
