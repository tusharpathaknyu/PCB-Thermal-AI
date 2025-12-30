# PCB Thermal AI Predictor

An ML-based tool for rapid PCB temperature distribution prediction from layout features.

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

## 🚀 Getting Started

```bash
# Clone repository
git clone https://github.com/tusharpathaknyu/PCB-Thermal-AI.git
cd PCB-Thermal-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Generate synthetic training data (quick test)
python scripts/generate_dataset.py --quick --visualize

# Generate full dataset (1000 samples)
python scripts/generate_dataset.py --num-samples 1000 --visualize

# Train the model
python scripts/train.py --data data/synthetic --epochs 50 --batch-size 16
```

## 📅 Roadmap

- [x] Project structure setup
- [x] Online resources research
- [x] Outreach email templates
- [x] Synthetic data generator implementation ✅
- [x] U-Net model implementation ✅
- [x] Training pipeline ✅
- [ ] FEM integration (Thermca)
- [ ] Validation with real thermal data
- [ ] FastAPI deployment
- [ ] Public release

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
