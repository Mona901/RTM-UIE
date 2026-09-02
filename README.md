# RTM-UIE: Underwater Image Enhancement Based on Retinex Prior and Transmission-Guided Residual Modulation

This repository provides the training and testing code of our RTM-UIE method.

RTM-UIE is designed for underwater image enhancement by integrating a Retinex prior, transmission estimation, background scattering cues, and transmission-guided residual modulation.

If you find this work useful, please cite our paper.

```bibtex
@article{rtmuie,
  title={Underwater Image Enhancement Based on Retinex Prior and Transmission-Guided Residual Modulation},
  author={Zhang, Bailu and Lin, Sen}
}
```

## Project Structure

```text
RTM-UIE/
├── README.md
├── requirements.txt
├── train.py
├── test.py
└── retinex/
    ├── __init__.py
    └── decomp/
        ├── __init__.py
        └── retinex_decomposer.py
```

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Please install PyTorch according to your CUDA environment.

## Dataset Preparation

The datasets used in this project are publicly available underwater image enhancement benchmarks. Please organize the training and testing data according to the required directory structure.

Detailed dataset organization and download information will be provided later.

## Training

Run the training script:

```bash
python train.py
```

The trained model will be saved according to the configuration in the training script.

## Testing

Run the testing script:

```bash
python test.py
```

The enhanced results will be saved according to the testing configuration.

## Method Overview

RTM-UIE integrates a Retinex prior, transmission estimation, background scattering cues, and transmission-guided residual modulation.

The refinement network takes the concatenation of:

```text
Concat(I, I0, B)
```

where:

- `I` denotes the original underwater image.
- `I0` denotes the Retinex prior image.
- `B` denotes the background scattering cue.

The transmission map is used for degradation-aware residual modulation rather than directly concatenated as an input channel.

The final enhanced image is obtained by adding the transmission-guided residual to the Retinex prior image.

## License

This project is released for academic research purposes only.