# GRACE: Privacy-centric Multimodal Seizure Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Available-2496ED.svg)](https://hub.docker.com/)

## 📋 Overview

**GRACE** is a privacy-centric multimodal framework for seizure detection and differentiation that combines video and ECG signals to achieve EEG-level performance. This repository contains the official PyTorch implementation of the paper:

> **Privacy-centric Multimodal Combination of Video and ECG Achieve EEG-level Performance for Seizure Detection and Differentiation**

<p align="center">
  <img src="https://github.com/Mr-TalhaIlyas/GRACE/blob/main/imgs/themev3.png" alt="Theme screenshot" width="500">
</p>
### Key Highlights

- 🎯 **96.2% sensitivity** (95% CI: 92.4–98.1) for seizure detection
- 🚨 **1.01 false alarms per hour** - significantly reduced through multimodal fusion
- 🌍 **Zero-shot generalization** across 4 international datasets (Australia, China, Europe, USA)
- 🔒 **Privacy-preserving** - uses fine-grained body part movements instead of identifiable video features
- 🏥 **Clinical-grade performance** - matches EEG-level accuracy (Cohen's κ=0.91, p=0.12)
- 💪 **Robust to missing modalities** - maintains 83.8% sensitivity when video or ECG unavailable

---

## 🔬 Abstract

Accurate seizure detection and diagnosis remains a critical unmet need in epilepsy management. The current gold standard is inpatient multi-day video-electroencephalography (video-EEG) monitoring. It is resource-intensive, costly, and restricted to specialized in-hospital settings, thus making it unsuitable for long-term and continuous monitoring. Alternative methods utilizing non-EEG based but in-silo modalities, such as electrocardiography (ECG)-based heart rate variability or standalone video analysis, have high false-alarm rates due to assumptions of signal stationarity and reliance on generic identifiable video features that also infringe patient privacy.

To address these limitations, we introduce a privacy-centric multimodal framework that integrates:
- **Dynamic ECG morphological features**: ictal tachycardia and wave dynamics
- **Privacy-preserving video analysis**: patient-focused fine-grained body part movements

Our model was developed on synchronized video-ECG data from an Australian cohort (78 seizures; 38 epileptic tonic–clonic, 40 functional/dissociative non-epileptic) and validated on held-out datasets from:
- **Alfred (Australia)**: 20 seizures
- **SAHZU (China)**: 36 seizures  
- **SeizeIT2 (Europe)**: 182 seizures
- **TUH (USA)**: 228 seizures

All validation sets demonstrated robust generalizability with sensitivity/specificity >90% (p<0.001 compared to in-silo single modality baselines).
<p align="center">
  <img src="https://github.com/Mr-TalhaIlyas/GRACE/blob/main/imgs/GRACE_overview4.png" alt="framework" width="500">
</p>
---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- Docker (for preprocessing pipelines)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mr-TalhaIlyas/GRACE.git
cd GRACE
```

2. **Create conda environment**
```bash
conda env create -f scripts/environment.yml
conda activate grace
```

3. **Install evaluation toolkit**
```bash
pip install seizurekit
```

### Docker Images for Preprocessing

We provide two Docker images for reproducible, privacy-preserving preprocessing:

#### 1. UniFlow - Optical Flow Extraction
```bash
docker pull talhailyas/uniflow
```
Documentation: [https://hub.docker.com/r/talhailyas/uniflow](https://hub.docker.com/r/talhailyas/uniflow)

#### 2. OpenPose-v2 - Pose Tracking & Temporal Stabilization
```bash
docker pull talhailyas/openpose-v2
```
Documentation: [https://hub.docker.com/r/talhailyas/openpose-v2](https://hub.docker.com/r/talhailyas/openpose-v2)

---

## 📁 Repository Structure

```
GRACE-main/
├── scripts/
│   ├── main_train.py              # Main training script
│   ├── train_model.script         # Training execution script
│   ├── environment.yml            # Conda environment specification
│   ├── configs/                   # Configuration files
│   ├── models/                    # Model architectures (GTN, ViT, Fusion, etc.)
│   ├── data/                      # Data loaders and augmentation
│   ├── evals/                     # Evaluation scripts for all datasets
│   ├── in_silo/                   # Single-modality training scripts
│   ├── buffer_utils/              # Feature extraction and buffering
│   ├── tools/                     # Training utilities
│   └── baselines/                 # Baseline model implementations
│       ├── david/                 # LSTM-based baseline
│       ├── hou/                   # AGCN baseline
│       ├── gestures/              # Gesture recognition baseline
│       ├── ocsvm/                 # One-class SVM baseline
│       ├── res1dcnn/              # 1D CNN baseline for ECG
│       └── vsvig/                 # Video-based seizure detection
├── datasets/
│   ├── alfred/                    # Alfred dataset utilities
│   └── tuh/                       # TUH-EEG dataset utilities
├── imgs/                          # Documentation images
└── README.md                      # This file
```

---

## 🎯 Training

### Multimodal Training (Video + ECG)

```bash
cd scripts
python main_train.py --config configs/config.py --mode multimodal
```

### Single Modality Training

**Video-only:**
```bash
cd scripts/in_silo
python main_flow.py --dataset alfred
# or
python main_pose.py --dataset alfred
```

**ECG-only:**
```bash
cd scripts/in_silo
python main_alfred.py --modality ecg
```

### Configuration

Modify `scripts/configs/config.py` to adjust:
- Model architecture (GTN, ViT, Fusion)
- Training hyperparameters
- Data augmentation settings
- Multi-GPU training options

---

## 📊 Evaluation

### Evaluate on Single Dataset

```bash
cd scripts/evals
python alfred_eval.py --checkpoint /path/to/model.pth
python sahzu_eval.py --checkpoint /path/to/model.pth
python seizeit2_eval.py --checkpoint /path/to/model.pth
python tuh_eval.py --checkpoint /path/to/model.pth
```

### Comprehensive Evaluation

```bash
cd scripts/evals
python main_eval.py --checkpoint /path/to/model.pth --all_datasets
```

This will compute:
- Sensitivity & Specificity
- AUROC & AUPRC
- False Alarm Rate (per hour)
- Cohen's Kappa (inter-rater agreement)
- Confusion matrices

---

## 🗂️ Datasets

### Supported Datasets

| Dataset | Origin | Seizures | Type |
|---------|--------|----------|------|
| **Training Cohort** | Australia | 78 | 38 epileptic, 40 non-epileptic |
| **Alfred** | Australia | 20 | Validation |
| **SAHZU** | China | 36 | Validation |
| **SeizeIT2** | Europe | 182 | Validation |
| **TUH-EEG** | USA | 228 | Validation |

### Data Preparation

1. **Extract optical flow from videos:**
```bash
docker run -v /path/to/videos:/data talhailyas/uniflow
```

2. **Extract pose keypoints:**
```bash
docker run -v /path/to/videos:/data talhailyas/openpose-v2
```

3. **Generate data arrays:**
```bash
cd datasets/tuh
python gen_data_arrays.py --input_dir /path/to/raw --output_dir /path/to/processed
```

---

## 📈 Results

### Seizure Detection Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Sensitivity** | 96.2% | 92.4–98.1% |
| **Specificity** | >90% | - |
| **False Alarm Rate** | 1.01/hour | - |
| **AUROC** | >0.95 | - |

### Zero-Shot Generalization

All validation datasets achieved:
- Sensitivity: >90%
- Specificity: >90%
- p-value: <0.001 (vs. single-modality baselines)

### Comparison with EEG

- Cohen's κ = 0.91
- p = 0.12 (no significant difference from EEG-based diagnosis)

### Robustness to Missing Modalities

- Single modality (video OR ECG): **83.8% sensitivity**
- Multimodal (video AND ECG): **96.2% sensitivity**

---


---

## 📦 Evaluation Toolkit: SeizureKit

Install our comprehensive evaluation toolkit:

```bash
pip install seizurekit
```

PyPI: [https://pypi.org/project/seizurekit/](https://pypi.org/project/seizurekit/)

**Features:**
- AUROC & AUPRC computation
- False alarm rate calculation
- Sensitivity & specificity analysis
- Statistical significance testing
- Confusion matrix generation
- Performance visualization

**Usage:**
```python
from seizurekit import evaluate_model

results = evaluate_model(
    predictions=pred_array,
    ground_truth=gt_array,
    sampling_rate=256
)
print(results.summary())
```

---

## 🔬 Key Technical Contributions

### 1. Privacy-Preserving Video Analysis
- Uses **body part movements** instead of identifiable facial features
- Temporal pose stabilization with OpenPose-v2
- Fine-grained motion dynamics extraction

### 2. Dynamic ECG Morphology
- Ictal tachycardia detection
- Wave dynamics analysis (P, QRS, T waves)
- Non-stationary signal processing

### 3. Multimodal Fusion Strategy
- Joint verification of autonomic (ECG) and motor (video) biomarkers
- Attention-based feature integration
- Robustness to missing modalities

### 4. Cross-Dataset Generalization
- Zero-shot evaluation on 4 continents
- No fine-tuning required
- Maintains >90% performance across populations

---

## 📝 Citation

If you use this code or our methodology in your research, please cite:

```bibtex
@article{GRACE,
PAPER IS CURRENTLY UNDER REVIEW
}
```

---

## 🛠️ Development & Testing

### Run Unit Tests
```bash
cd scripts/in_silo
bash run_tests.sh
```

### Code Quality
```bash
# Run pytest
python -m pytest scripts/

# Check imports
python scripts/import pytest.py
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Alfred Health, Australia
- SAHZU Hospital, China
- SeizeIT2 Consortium, Europe
- Temple University Hospital (TUH), USA
- All patients and healthcare providers who contributed to dataset collection

---

## 📧 Contact

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: [https://github.com/Mr-TalhaIlyas/GRACE/issues](https://github.com/Mr-TalhaIlyas/GRACE/issues)
- **Email**: [Your contact email]

---

## 🔗 Related Resources

| Resource | Link |
|----------|------|
| **Main Repository** | [https://github.com/Mr-TalhaIlyas/GRACE](https://github.com/Mr-TalhaIlyas/GRACE) |
| **SeizureKit (PyPI)** | [https://pypi.org/project/seizurekit/](https://pypi.org/project/seizurekit/) |
| **UniFlow Docker** | [https://hub.docker.com/r/talhailyas/uniflow](https://hub.docker.com/r/talhailyas/uniflow) |
| **OpenPose-v2 Docker** | [https://hub.docker.com/r/talhailyas/openpose-v2](https://hub.docker.com/r/talhailyas/openpose-v2) |

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=Mr-TalhaIlyas/GRACE&type=Date)](https://star-history.com/#Mr-TalhaIlyas/GRACE&Date)

---

## 📊 Project Status

- ✅ Training code released
- ✅ Evaluation code released
- ✅ Baseline models released
- ✅ Docker images released
- ✅ SeizureKit toolkit released
- 🔄 Pretrained models (coming soon)
- 🔄 Detailed tutorials (coming soon)
- 🔄 Demo application (coming soon)

---

**Made with ❤️ for advancing epilepsy care through AI**


