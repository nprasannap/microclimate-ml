[![DOI](https://zenodo.org/badge/1403808252.svg)](https://doi.org/10.5281/zenodo.23130907)
# Microclimate ML Weather Synthesis Framework (v1.0)

This repository contains the official code and data for the manuscript:
**"Microclimate-Aware Machine Learning for Cross-Site Weather Forecasting: A Zero-Shot Spatial Generalization Approach in Tropical Island Environments"**

## 📂 Repository Structure
- `data/`: Contains individual agro-climatic site Excel files for Badulla, Colombo, Galle, and Kandy.
- `CrossSite_Pipeline.py`: Main processing script for temporal phase alignment, microclimate feature engineering, LOLO cross-validation, and SHAP interpretability.

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Model Pipeline:**
   ```bash
   python CrossSite_Pipeline.py
   ```

## 📄Citation
If you use this dataset or code in your research, please cite the corresponding paper.
