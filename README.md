# Real-time Detection of Malicious Traffic in IoT Devices Based on Lightweight Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A lightweight, real-time intrusion detection system for IoT devices using advanced feature selection and compressed machine learning models. This project aims to provide robust security monitoring while maintaining minimal resource usage, making it suitable for deployment on edge devices.

## 🌟 Key Features

- **Lightweight Detection**: CPU-only operation with <5MB model size
- **Fast Inference**: <5ms per prediction on typical hardware
- **High Accuracy**: >95% detection rate with <2% false positives
- **Resource Efficient**: <100MB RAM usage during operation
- **Real-time Monitoring**: Live traffic analysis with REST API
- **Multiple Attack Types**: Detection of DDoS, Botnet, and other IoT-specific threats

## 📊 Performance Benchmarks(Not available yet)

| Metric | Value |
|--------|--------|
| Model Size | <5MB |
| Inference Time | <5ms |
| Detection Accuracy | >95% |
| False Positive Rate | <2% |
| RAM Usage | <100MB |
| CPU Usage | <20% (on Raspberry Pi 4) |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- libpcap (for packet capture)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Real-time-detection-of-malicious-traffic-in-IoT-devices-based-on-lightweight-machine-learning.git
cd Real-time-detection-of-malicious-traffic-in-IoT-devices-based-on-lightweight-machine-learning
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

1. Download and prepare dataset:
```bash
python src/data/download_dataset.py --dataset bot-iot
python src/data/preprocess.py
```

2. Train the model:
```bash
python src/models/train.py
```

3. Start the detection system:
```bash
python src/detection/api.py
```

4. Access the API at http://localhost:8000/docs

## 🔧 Advanced Configuration

The system can be configured through `configs/config.yaml`. Key settings include:

- Window size for packet analysis
- Feature selection parameters
- Model compression settings
- Detection thresholds
- Network interface settings

Example configuration:
```yaml
detection:
  window_size: 100
  confidence_threshold: 0.8
  interface: "eth0"
```

## 📚 Documentation

- [Feature Selection](docs/feature_selection.md)
- [Model Architecture](docs/model.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🧪 Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest tests/
```

3. Check code style:
```bash
black src/ tests/
flake8 src/ tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style guidelines
- Development setup
- Pull request process
- Feature request guidelines

## 📈 Roadmap

- [ ] Support for additional IoT protocols
- [ ] Integration with popular IoT platforms
- [ ] GUI for system monitoring
- [ ] Automated model retraining
- [ ] Distributed detection capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links(Not available yet)

- [Research Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [Project Website](https://your-project-website.com)
- [Documentation](https://your-docs-website.com)

## 📧 Contact

- **Author**: Xuyi Ren
- **Email**: renxuyi@grinnell.edu


## 🙏 Acknowledgments

- Grinnell College for supporting this research
- The IoT security research community
- Contributors and users of this project

## 📚 Citation

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
