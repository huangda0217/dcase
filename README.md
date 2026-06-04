# DCASE 2026 Task 2 — Unsupervised Anomalous Sound Detection

> **Before using this project**, first clone the repository and navigate into the directory:
> ```bash
> git clone https://github.com/huangda0217/dcase.git
> cd dcase
> ```

Unsupervised anomaly detection on industrial machine sounds. Trained on normal audio only (~1000 files per machine), outputs anomaly scores for new recordings.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies**: `torch`, `torchaudio`, `scikit-learn`, `scipy`, `librosa`, `soundfile`


## 2. Download the DCASE2026T2 dev_data and eval_data

Scripts download and extract files to `data/dcase2026t2/{dev_data,eval_data}/raw/` by default.

```bash
# Dev set (train+test, 7 machine types)
bash data_download_2026dev.sh

# Eval set (train, 5 machine types)
bash data_download_2026add.sh

# Eval set (test, 5 machine types)
bash data_download_2026eval.sh
```

Audio is **16 kHz dual-channel**.


## 3. Download BEATs Model Weights. (must download)
Download pre-trained weights of BEATs from [https://github.com/microsoft/unilm/tree/master/beats](https://github.com/microsoft/unilm/tree/master/beats).



Place at `inference/checkpoints/BEATs_iter3_plus_AS2M.pt` (~345 MB). GPU required (CPU works but extremely slow).

# Usage 



```bash
# Default: dev set, auto device selection
python main.py

# Specify eval set + GPU
python main.py --data_type eval_data --device cuda:0

# Custom features and threshold
python main.py --features subband beats sc --threshold 0.85
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_type` | `dev_data` | Data type: `dev_data` / `eval_data` |
| `--device` | auto | Inference device, defaults to `cuda` when GPU available |
| `--features` | `subband beats sc` | Feature list, options: `subband` `beats` `sc` `beats_ml` `cqt` |
| `--threshold` | `0.85` | Gamma PPF decision threshold percentile |
| `--result_dir` | `results` | Output directory |
| `--mono` | `False` | Force mono reading |




### Training data

- **source domain**: ~990 files
- **target domain**: ~10 files, domain-shift scenario
- Domain is inferred from file path: `"target"` in path → target domain

## Outputs

After inference, `results/{data_type}/` contains 3 CSV files per machine.

### 1. Anomaly Scores

`anomaly_score_{machine_type}_section_{section_index}_test.csv`

| File | Score |
|------|-------|
| section_00_source_test_normal_0000.wav | 0.321 |
| section_00_source_test_anomaly_0001.wav | 2.847 |

Score is a float; higher values indicate higher anomaly likelihood.

### 2. Decision Results

`decision_result_{machine_type}_section_{section_index}_test.csv`

| File | Decision |
|------|----------|
| section_00_source_test_normal_0000.wav | 0 |
| section_00_source_test_anomaly_0001.wav | 1 |

Decision = score > threshold ? 1 (anomaly) : 0 (normal). Threshold is computed by fitting a Gamma distribution to training anomaly scores and taking the PPF.


## Acknowledgement
- We thanks the authors of  for [BEATs](https://arxiv.org/abs/2212.09058) providing the pre-trained weights.
- We thanks [dcase2023_task2_baseline_ae](https://github.com/nttcslab/dcase2023_task2_baseline_ae) for providing reference code and scripts for downloading the development and evaluation datasets
