# RTM-UIE: Underwater Image Enhancement Based on Retinex Prior and Transmission-Guided Residual Modulation

This repository includes the training and testing code of our RTM-UIE method.

RTM-UIE is designed for underwater image enhancement by integrating Retinex prior, transmission estimation, background scattering cues, and transmission-guided residual modulation.

If you use this code, please cite our paper.

```bibtex
@article{rtmuie,
  title={Underwater Image Enhancement Based on Retinex Prior and Transmission-Guided Residual Modulation},
  author={Zhang, Bailu and Lin, Sen and Liu, Xiyao},
  journal={},
  year={}
}
```

## Environment

Python 3.10 is recommended.

Create a new conda environment:

```bash
conda create -n rtm-uie python=3.10
conda activate rtm-uie
```

Please install PyTorch and torchvision according to your CUDA version.

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
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

## Train and Test Sets

To use `train.py`, the paired training images should be organized as follows.

```text
<data directory>/
├── Train_UIEB/
│   ├── input/
│   │   ├── 0001.png
│   │   ├── ...
│   │   └── xxxx.png
│   └── GT/
│       ├── 0001.png
│       ├── ...
│       └── xxxx.png
├── Train_LSUI/
│   ├── input/
│   │   ├── 0001.png
│   │   ├── ...
│   │   └── xxxx.png
│   └── GT/
│       ├── 0001.png
│       ├── ...
│       └── xxxx.png
```

The input images and reference images should have corresponding file names.

For testing, only input images are required.

```text
<data directory>/
└── test_images/
    ├── 0001.png
    ├── 0002.png
    ├── ...
    └── xxxx.png
```

## Training

Run the following command for training:

```bash
python train.py \
  --out_dir runs/rtm_uie \
  --epochs 40 \
  --batch 6 \
  --lr 1e-4 \
  --retinex_ckpt path/to/retinex_checkpoint.pth
```

If a transmission estimation checkpoint is used for initialization, run:

```bash
python train.py \
  --out_dir runs/rtm_uie \
  --epochs 40 \
  --batch 6 \
  --lr 1e-4 \
  --retinex_ckpt path/to/retinex_checkpoint.pth \
  --init_tnet_ckpt path/to/transmission_checkpoint.pth \
  --freeze_tnet
```

The trained model will be saved in the directory specified by `--out_dir`.

## Testing

Run the following command for testing:

```bash
python test.py \
  --ckpt path/to/rtm_uie_checkpoint.pth \
  --input_dir data/test_images \
  --out_dir results/rtm_uie \
  --retinex_ckpt path/to/retinex_checkpoint.pth
```

The enhanced images will be saved in the directory specified by `--out_dir`.

## Method Overview

The proposed RTM-UIE uses a 9-channel refinement input:

```text
Concat(I, I0, B)
```

where:

```text
I      denotes the original underwater image;
I0     denotes the Retinex prior image;
B      denotes the background scattering cue.
```

The transmission map is not directly concatenated as a common input channel. It is used to construct the background scattering cue and generate the residual modulation weight.

The final enhanced result is obtained by adding the transmission-guided residual to the Retinex prior image.

## Contact

If you have any questions, please contact the authors.
