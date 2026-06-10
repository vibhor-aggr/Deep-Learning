# Deep Learning Assignments

This repository contains four Deep Learning assignment notebooks covering custom data loading, image/audio classification, autoencoders, conditional GANs, and time-series forecasting.

## Repository Layout

```text
Deep-Learning/
├── Assignment1/
│   └── code.ipynb
├── Assignment2/
│   ├── code.ipynb
│   ├── auto_encoder.pth
│   ├── auto_encoder.png
│   ├── variational_auto_encoder.pth
│   └── variational_auto_encoder.png
├── Assignment3/
│   ├── Assignment3.pdf
│   └── code.ipynb
└── Assignment4/
    └── code.ipynb
```

## Environment

The notebooks are designed for Python 3 and Google Colab-style execution. Most notebooks install their own dependencies in the first code cell.

Core packages used across the repository:

- `torch==2.1.2`
- `torchvision==0.16.2`
- `torchaudio==2.1.2`
- `numpy==1.25.2`
- `pandas==2.2.3`
- `matplotlib==3.9.4`
- `soundfile==0.13.0`
- `scikit-image==0.21.0`
- `tqdm==4.67.1`
- `scikit-learn`, `h5py`, and `gdown` for Assignment 3

Recommended workflow:

1. Open the target notebook in Google Colab or Jupyter.
2. Run the dependency installation cell.
3. Place the required data in the expected directory layout described below.
4. Run the notebook top-to-bottom.

## Assignment 1: Custom Datasets, MLPs, CNNs, and Bonus Models

Notebook: `Assignment1/code.ipynb`

Topics covered:

- Manual `SpeechCommandDataset` implementation.
- Manual `FashionMNISTDataset` implementation.
- Custom `DataLoader` and collate functions without using `torch.utils.data.DataLoader`.
- Benchmarking custom data loading.
- MLP classifiers for Speech Commands and FashionMNIST.
- CNN classifiers using ResNet-style and Inception-style blocks.
- Bonus very-deep MLP/CNN models for FashionMNIST.

Expected data layout:

```text
Assignment1/
└── data/
    ├── speech_commands_v0.01/
    │   ├── training_list.txt
    │   ├── testing_list.txt
    │   └── <label>/*.wav
    └── fashion-mnist/
        ├── fashion-mnist_train.csv
        └── fashion-mnist_test.csv
```

Main outputs:

- Training/validation loss and accuracy plots.
- Optional bonus plots:
  - `q4_bonus_mlp_fashion_mnist.png`
  - `q4_bonus_cnn_fashion_mnist.png`

## Assignment 2: Image Restoration with Autoencoders

Notebook: `Assignment2/code.ipynb`

Topics covered:

- Paired FashionMNIST image-restoration dataset.
- Generic convolutional encoder and decoder.
- AutoEncoder training and reporting.
- Variational AutoEncoder training and reporting.
- Bonus Conditional Variational AutoEncoder.
- SSIM-based evaluation and training curves.

Expected data layout:

```text
Assignment2/
└── data/
    ├── train/
    │   ├── aug/
    │   └── clean/
    └── test/
        ├── aug/
        └── clean/
```

Tracked generated artifacts:

- `Assignment2/auto_encoder.pth`
- `Assignment2/auto_encoder.png`
- `Assignment2/variational_auto_encoder.pth`
- `Assignment2/variational_auto_encoder.png`

Additional generated artifacts:

- `conditional_variational_auto_encoder.pth`
- `conditional_variational_auto_encoder.png`

## Assignment 3: Text-to-Image Synthesis with Conditional GANs

Specification: `Assignment3/Assignment3.pdf`  
Notebook: `Assignment3/code.ipynb`

Task summary:

- Train a conditional GAN for Oxford-102 flower image generation from text descriptions.
- Use a Source Encoder to produce image representations.
- Use a Target Generator conditioned on source representations and text encodings.
- Use a Discriminator to distinguish generated and real images.
- Select 20 classes for training and 5 classes for testing.
- Train all parameters from scratch; no pretrained checkpoints are used.

The notebook includes:

- Data download/setup using `gdown` and HDF5.
- Vocabulary and raw-caption text encoder.
- Source Encoder, Target Generator, and Discriminator definitions.
- Training loop with adversarial, reconstruction, feature-matching, and contrastive objectives.
- Testing section that generates a 5x5 image grid, plots 3D t-SNE embeddings, and reports parameter/model-size statistics.

Generated outputs are written under:

```text
Assignment3/checkpoints/
```

Typical files include model checkpoints, training curves, generated grids, t-SNE plots, and model statistics.

## Assignment 4: Stock Price Prediction with LSTM

Notebook: `Assignment4/code.ipynb`

Task summary:

- Train an autoregressive LSTM model for `ASIANPAINT.NS` 15-minute closing-price prediction.
- Predict the next 5 trading days, i.e. `5 * 25 = 125` values.
- Save the trained model weights for evaluation.

Expected runtime files:

```text
Assignment4/
├── past_5_days.csv
└── next_5_days.csv
```

Expected CSV columns:

- `Datetime`
- `Close`

Generated artifacts:

- `trained_lstm.pth`
- `lstm_training.png`
- `lstm_prediction.png`
- `training_prices.csv`

The notebook can also attempt to fetch recent `ASIANPAINT.NS` history from Yahoo Finance when internet access is available.

## Notes for Reproducibility

- Run each notebook from inside its own assignment directory unless paths are adjusted.
- Data directories are not committed to this repository.
- Large generated model files should be regenerated when changing model architecture or hyperparameters.
- Assignment 3 is intended for GPU-backed Colab execution.
- Assignment 4 depends on fresh market data; evaluation quality depends on the supplied `past_5_days.csv` and `next_5_days.csv`.

<!-- ## Git Hygiene

Recommended files to keep out of version control when regenerating outputs:

```text
Assignment*/data/
Assignment3/checkpoints/
Assignment4/*.csv
Assignment4/trained_lstm.pth
Assignment4/lstm_training.png
Assignment4/lstm_prediction.png
``` -->

