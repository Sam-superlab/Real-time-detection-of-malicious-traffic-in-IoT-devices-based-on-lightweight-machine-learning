# Real-time Detection of Malicious Traffic in IoT Devices Based on Lightweight Machine Learning

This project implements a real-time malicious traffic detection system for IoT devices using lightweight machine learning techniques. The system is designed to run efficiently on low-compute environments (CPU-only, model size < 5MB, training time < 2 hours).

## Features

- Real-time network traffic monitoring and analysis
- Lightweight machine learning model (LightGBM based)
- Low resource consumption (CPU-only, <100MB RAM)
- Fast inference (<5ms per prediction)
- Support for multiple IoT attack types (DDoS, Botnet, etc.)

## Project Structure

```
.
├── data/                   # Data storage and preprocessing
│   ├── raw/               # Raw dataset files
│   ├── processed/         # Processed features and labels
│   └── models/            # Trained model files
├── src/                   # Source code
│   ├── data/             # Data processing scripts
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and inference
│   └── detection/        # Real-time detection system
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Unit tests
├── configs/             # Configuration files
└── requirements.txt     # Python dependencies
```

## Requirements

- Python 3.8+
- LightGBM
- ONNX Runtime
- dpkt/scapy
- FastAPI (for API endpoints)
- pandas, numpy, scikit-learn

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Real-time-detection-of-malicious-traffic-in-IoT-devices-based-on-lightweight-machine-learning.git
cd Real-time-detection-of-malicious-traffic-in-IoT-devices-based-on-lightweight-machine-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and prepare dataset:
```bash
python src/data/download_dataset.py
python src/data/preprocess.py
```

4. Train the model:
```bash
python src/models/train.py
```

5. Start the detection system:
```bash
python src/detection/main.py
```

## Model Details

The system uses a LightGBM-based model with the following characteristics:
- Input: Network traffic features (statistical, protocol-based, temporal)
- Output: Binary classification (normal/malicious)
- Model size: <5MB after compression
- Inference time: <5ms per prediction
- Accuracy: >95% (target)
- False Positive Rate: <2% (target)

## Development

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{iot-malicious-traffic-detection,
  author = {Xuyi Ren},
  title = {Real-time Detection of Malicious Traffic in IoT Devices Based on Lightweight Machine Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/sam-superlab/Real-time-detection-of-malicious-traffic-in-IoT-devices-based-on-lightweight-machine-learning}
}
``` 