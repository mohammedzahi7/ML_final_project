# Speech Emotion Recognition (SER) System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20TensorFlow-orange.svg)](https://pytorch.org/)
[![Audio Engine](https://img.shields.io/badge/audio-librosa%20%7C%20soundfile-green.svg)](https://librosa.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Deep Learning pipeline designed to automatically classify human emotional states from raw speech audio signals. The system extracts discriminative acoustic signatures and evaluates them using both classical machine learning frameworks and high-capacity hybrid deep neural architectures.

---

## 📌 Project Architecture

The core pipeline processes raw time-domain audio wave files through an optimized five-stage processing architecture:

```
  [ Raw Audio (.wav) ] 
           │
           ▼
┌──────────────────────┐
│  1. Preprocessing    │ ──► Noise reduction, silence trimming, and 22kHz resampling
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│ 2. Feature Extraction│ ──► Compute MFCCs, Mel-Spectrograms, Chroma, and Prosody
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  3. Model Ensembles  │ ──► Classical Baseline (SVM/RF) OR Hybrid CNN-LSTM / wav2vec 2.0
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│ 4. Evaluation Matrix │ ──► Confusion Matrix tracking, F1-Score tuning, and Grad-CAM
└──────────────────────┘
           │
           ▼
[ Live Interactive App UI ] ──► Streamlit Interface & FastAPI Production Engine
```

---

## 📊 Supported Datasets

The models are pre-configured to ingest and parse popular public speech corpus variations out of the box:

*   **RAVDESS:** *The Ryerson Audio-Visual Database of Emotional Speech and Song.* Contains 24 professional actors vocalizing 8 emotional states with pristine studio fidelity.
*   **CREMA-D:** *Crowd-sourced Emotional Multimodal Actors Dataset.* Features 91 diverse actors generating 7,442 clips with wide demographic distributions.
*   **TESS:** *Toronto Emotional Speech Set.* High-quality recordings from 2 female speakers tracking 7 target categories.
*   **SAVEE:** *Surrey Audio-Visual表现 Emotional Exchange.* Baseline configurations trained using 4 male actors.

---

## 🛠️ Feature Extraction Breakdown

To achieve state-of-the-art diagnostic convergence, the system transforms raw signals into detailed multi-dimensional arrays:

1.  **MFCCs (Mel-Frequency Cepstral Coefficients):** Extracts 13 to 40 structural coefficients to track human vocal tract shapes, timbre variations, and acoustic envelopes.
2.  **Mel-Spectrograms:** Constructs a time-frequency visualization matrix mapped directly onto the non-linear human auditory perception scale.
3.  **Prosodic Properties:** Extracts global pitch tracking ($F_0$), Root-Mean-Square (RMS) frame energy, and localized speech cadence values to gauge arousal dynamics.
4.  **Spectral Indicators:** Measures spectral contrast, zero-crossing rates, and spectral roll-off boundaries to accurately capture harmonic brightness.

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.10 or higher
*   `libsndfile` library binaries installed on the host operating system

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/mohammedzahi7/ML_final_project.git
    cd ML_final_project
    ```

2.  Set up a virtual environment and update your base tools:
    ```bash
    python -bin m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install --upgrade pip
    ```

3.  Install the core dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the System

*   **Run Feature Extraction:**
    ```bash
    python src/extract_features.py --dataset path/to/dataset --output data/processed_features.pkl
    ```

*   **Train Models:**
    ```bash
    python src/train.py --config configs/hybrid_cnn_lstm.yaml --data data/processed_features.pkl
    ```

*   **Launch the Interactive Web UI:**
    ```bash
    streamlit run app/main.py
    ```

---

## 📈 Model Capabilities & Architecture Options

This repository includes interchangeable script modules optimized for various operating environments:

*   **Lightweight Deployment Model:** Support Vector Machines (SVM) or Random Forest configurations optimized for low-latency embedded processing nodes.
*   **Deep Learning Pipeline:** A robust, deep Hybrid **CNN-LSTM** pipeline. The spatial CNN layers decode topological 2D patterns from Mel-spectrograms, passing down internal maps to sequential LSTM layers that capture evolving time-series emotional transitions.
*   **SOTA Architectures:** Modular adaptors designed to easily fine-tune large self-supervised audio weights including **wav2vec 2.0** and **HuBERT**.

---

## 👥 Contributors

*   **Mohammed Zahi** - [mohammedzahi08@gmail.com](mailto:mohammedzahi08@gmail.com)
*   **Mupped Rasmil**
*   **Muhammed Afsal**

---

## 📄 License

This project is open-sourced under the terms of the MIT License. See the [LICENSE](LICENSE) file for deeper details.
