# AI-Solar-Flare-PINN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**Physics-Informed Machine Learning for Solar Magnetic Field Prediction**

---

## 🌞 Project Overview

AI-Solar-Flare-PINN is a state-of-the-art research framework for predicting 3D solar magnetic fields from 2D surface magnetograms using advanced neural operators. The project leverages Physics-Informed Neural Networks (PINNs), DeepONet, and Fourier Neural Operators (FNO) to enable accurate, physically consistent modeling of the solar corona—crucial for space weather forecasting and solar flare prediction.

- **Input:** 2D vector magnetograms from SDO/HMI
- **Output:** 3D magnetic field reconstructions
- **Techniques:** PINNs, DeepONet, FNO, analytical benchmarks (Low & Lou model)
- **Applications:** Space weather, solar flare/CME prediction, scientific ML

---

## 🚀 Features
- Physics-informed loss functions for robust, interpretable predictions
- Modular neural operator implementations (PINN, DeepONet, FNO)
- Scalable data pipeline for SDO/HMI and synthetic data
- Analytical benchmarks with the Low & Lou model
- Visualization and evaluation tools (field lines, MSE, SSIM)
- Ready for extension to real-time and multi-physics applications

---

## 🛠️ Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/NadavShanun-design/AI-Solar-Flare-PINN.git
cd AI-Solar-Flare-PINN
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Preprocess Data
Prepare SDO/HMI or synthetic magnetogram data:
```bash
python data/preprocess_magnetograms.py --input_dir <fits_dir> --output_file data/processed_mags.npz
```

### 4. Train a Model
Train a PINN, DeepONet, or FNO on toy or real data:
```bash
python training/train_pinn.py --optimizer adam --lambda_data 1.0 --lambda_phys 1.0
python models/deeponet_jax.py  # For DeepONet demo
python models/fno_jax.py       # For FNO demo
```

### 5. Evaluate and Visualize
```bash
python evaluation/visualize_field.py
```

---

## 📂 Project Structure
```
AI-Solar-Flare-PINN/
├── data/           # Data pipeline and preprocessing
├── models/         # PINN, DeepONet, FNO implementations
├── training/       # Training scripts and optimization
├── evaluation/     # Analytical models, visualization, metrics
├── research/       # Research notes and documentation
├── notebooks/      # Jupyter notebooks for exploration
├── docs/           # Proposals and extended documentation
└── README.md       # Project overview and instructions
```

---

## 📊 Example Results
- **PINN, DeepONet, and FNO** tested on 1D Laplace equation and synthetic Low & Lou fields
- **Metrics:** MSE, SSIM, field line visualizations
- **Ready for extension to full 3D SDO/HMI data**

---

## 📖 Citation
If you use this codebase in your research, please cite:
```bibtex
@misc{ai-solar-flare-pinn,
  author = {Nadav Shanun and contributors},
  title = {AI-Solar-Flare-PINN: Physics-Informed Machine Learning for Solar Magnetic Field Prediction},
  year = {2024},
  howpublished = {GitHub},
  url = {https://github.com/NadavShanun-design/AI-Solar-Flare-PINN}
}
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 