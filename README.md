# PCB Thermal AI Predictor

An ML-based tool for rapid PCB temperature distribution prediction from layout features.

## ✅ Current Results

| Metric | Value |
|--------|-------|
| **Mean Absolute Error** | 6.0°C |
| **Max Error** | ~17°C |
| **Inference Time** | <50ms (PyTorch) / 27ms (ONNX) |
| **Training Dataset** | 2,000 samples |
| **Model Parameters** | 4.3M (U-Net) |

🚀 **Get thermal feedback in SECONDS instead of HOURS!**

## 🌟 Features

- **🖥️ Interactive Web Demo** - Streamlit app with real-time visualization
- **🎲 Data Augmentation** - Rotation, flip, noise injection, power scaling
- **📊 Uncertainty Quantification** - MC Dropout for confidence estimation
- **💡 Design Recommendations** - AI-powered thermal optimization suggestions
- **⚡ ONNX Export** - Cross-platform deployment, 27ms inference
- **🔌 REST API** - FastAPI endpoint for integration

## 🎯 Project Goal

Develop a machine learning model that predicts temperature fields across printed circuit boards directly from layout characteristics, enabling:
- **Seconds** instead of hours for thermal feedback
- **Early-stage** thermal issue detection during design
- **Reduced** prototype iterations due to thermal failures

## 📁 Project Structure

```
PCB-Thermal-AI/
├── README.md              # This file
├── data/                  # Training data (synthetic + simulated + real)
│   ├── synthetic/         # Analytically generated samples
│   ├── simulated/         # FEM-based (Thermca/Elmer) samples  
│   └── real/              # Thermal camera measurements
├── docs/                  # Documentation and research
│   └── ONLINE_RESOURCES_RESEARCH.md
├── emails/                # Outreach templates
│   ├── PROFESSOR_OUTREACH_TEMPLATE.md
│   ├── PROFESSOR_CONTACT_LIST.md
│   ├── COMPANY_OUTREACH_TEMPLATE.md
│   └── COMPANY_CONTACT_LIST.md
├── src/                   # Source code
│   ├── data_generation/   # Synthetic data generators
│   ├── models/            # PyTorch model definitions
│   ├── training/          # Training scripts
│   └── api/               # FastAPI inference server
└── scripts/               # Utility scripts
    └── github_scraper/    # Scrape KiCad projects for layouts
```

## 🔬 Technical Approach

### Input Features
- Copper density maps (per layer)
- Via location and thermal via patterns
- Component footprints and power dissipation
- Board stack-up (layer count, materials)
- Boundary conditions (ambient, convection coefficients)

### Output
- 2D temperature field (°C per pixel)
- Hotspot locations with peak temperatures
- Thermal design suggestions

### Architecture
- **Primary**: U-Net CNN for spatial temperature prediction
- **Alternative**: Graph Neural Network for component-level predictions

### Training Data Strategy
1. **Synthetic** (Phase 1): Analytical 2D heat equation solver
2. **Simulated** (Phase 2): Thermca/Elmer FEM for higher fidelity
3. **Real** (Phase 3): Thermal camera validation data

## 📊 Targets

- **Dataset Size**: 5,000+ samples
- **Mean Error**: <3°C
- **Inference Time**: <1 second per board
- **Max Hotspot Error**: <5°C

## 🛠️ Tech Stack

- **ML Framework**: PyTorch
- **Data Processing**: NumPy, Pandas, OpenCV
- **Thermal Simulation**: Thermca (Python FEM), Elmer FEM
- **Gerber Parsing**: gerber-parser, pcb-tools
- **API**: FastAPI
- **Visualization**: Matplotlib, PyVista

## 📚 Key Resources

See [docs/ONLINE_RESOURCES_RESEARCH.md](docs/ONLINE_RESOURCES_RESEARCH.md) for detailed analysis of:
- Available GitHub repositories
- Academic resources
- Useful tools for data generation

## 📧 Outreach

Email templates and contact lists for data acquisition:
- [Professor outreach template](emails/PROFESSOR_OUTREACH_TEMPLATE.md)
- [Professor contact list](emails/PROFESSOR_CONTACT_LIST.md)
- [Company outreach template](emails/COMPANY_OUTREACH_TEMPLATE.md)
- [Company contact list](emails/COMPANY_CONTACT_LIST.md)

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/tusharpathaknyu/PCB-Thermal-AI.git
cd PCB-Thermal-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# 🖥️ Launch interactive web demo (recommended!)
streamlit run app.py
# Visit http://localhost:8501

# Run CLI demo (visualize ML vs FEA comparison)
python scripts/demo.py --save-figure

# Start REST API server
uvicorn src.api.server:app --reload
# Visit http://localhost:8000/docs for interactive API
```

## 🔧 Training Your Own Model

```bash
# Generate synthetic training data (2000 samples)
python scripts/generate_dataset.py --num-samples 2000 --output data/synthetic

# Train the model (50 epochs, ~1 hour on M1 Mac)
python scripts/train.py --data data/synthetic --epochs 50 --batch-size 16

# Model checkpoint saved to: checkpoints/best.pth

# Export to ONNX for production deployment
python scripts/export_onnx.py --checkpoint checkpoints/best.pth --output models/pcb_thermal.onnx
```

## 📊 Advanced Features

### Uncertainty Quantification
```python
from src.inference.uncertainty import UncertaintyPredictor

predictor = UncertaintyPredictor('checkpoints/best.pth', n_samples=20)
result = predictor.predict_with_uncertainty(features)
print(f"Temperature: {result['mean_temp']:.1f}°C ± {result['temp_uncertainty']:.1f}°C")
print(f"High uncertainty regions: {result['high_uncertainty_fraction']:.1%}")
```

### Data Augmentation
```python
from src.training.augmentation import ThermalAugmentation

# Use preset or custom config
augment = ThermalAugmentation.from_preset('default')
features_aug, temp_aug = augment(features, temperature)
```

### ONNX Inference (27ms)
```python
import onnxruntime as ort
session = ort.InferenceSession('models/pcb_thermal.onnx')
output = session.run(None, {'pcb_features': features})
```

## 🌐 API Usage

```bash
# Health check
curl http://localhost:8000/health

# Quick prediction (generates random PCB)
curl -X POST "http://localhost:8000/predict/quick?total_power=3.0&copper_fill=0.6"

# Full prediction (send your own PCB features)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"copper": [[...]], "vias": [[...]], "components": [[...]], "power": [[...]]}'
```

## 📅 Roadmap

- [x] Project structure setup
- [x] Online resources research
- [x] Outreach email templates
- [x] Synthetic data generator implementation ✅
- [x] U-Net model implementation ✅
- [x] Training pipeline ✅
- [x] FastAPI deployment ✅
- [x] Inference module ✅
- [x] Demo script ✅
- [x] Interactive Streamlit web demo ✅
- [x] Data augmentation pipeline ✅
- [x] Uncertainty quantification (MC Dropout) ✅
- [x] Design recommendations ✅
- [x] ONNX export ✅
- [ ] FEM integration (Thermca/Elmer for higher fidelity)
- [ ] Validation with real thermal camera data
- [ ] Multi-layer PCB support
- [ ] Public release & paper

## 👤 Author

**Tushar Pathak**
- MS Computer Engineering, NYU (Expected 2026)
- Former Applications Intern, Texas Instruments
- Email: [YOUR_EMAIL]
- LinkedIn: [YOUR_LINKEDIN]

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Texas Instruments (industry experience and domain knowledge)
- NYU Tandon School of Engineering
- Open-source thermal simulation community

---

*This project is in active development. Contributions welcome!*
