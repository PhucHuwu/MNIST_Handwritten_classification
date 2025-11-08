# MNIST Handwritten Digit Classification

This repository contains a minimal example for training a neural network on the MNIST handwritten digit dataset using PyTorch. It includes a Jupyter notebook (`train_model.ipynb`) that downloads the data, builds a model, trains and evaluates it, and visualizes results.

## Contents

- `train_model.ipynb` — Jupyter notebook with dataset preparation, model definition, training loop, evaluation and visualization.
- `nn.py` — (helper) neural network and/or training utilities (see file for details).
- `LICENSE` — project license.

## Features

- Simple, easy-to-follow example for beginners learning PyTorch and neural networks.
- Uses torchvision's MNIST dataset loader with automatic download.
- Notebook contains cells for data inspection, training, and visual outputs.

## Requirements

- Python 3.8+ (3.10 recommended)
- PyTorch (appropriate for your OS/CUDA setup)
- torchvision
- matplotlib
- numpy
- Jupyter Notebook or Jupyter Lab (to open the notebook)

You can install the main requirements with pip. Example (CPU-only PyTorch):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision matplotlib numpy jupyter
```

If you need CUDA-enabled PyTorch, follow the official install instructions at https://pytorch.org.

## Quick start

1. Clone the repository and change into the project folder.
2. Create and activate a virtual environment and install dependencies (see above).
3. Start Jupyter and open the notebook:

```powershell
jupyter notebook train_model.ipynb
```

4. Run the notebook cells in order. The notebook will download the MNIST dataset automatically (to `./data` by default), build the model, perform training, and display sample outputs.

Alternatively, if `nn.py` contains a runnable training script, you may run it directly (for example):

```powershell
python nn.py
```

Note: check `nn.py` for its exact interface (arguments, saving behavior).

## Expected results

- The notebook trains a small classifier on MNIST and prints training/validation metrics.
- It also displays sample images and may save a model checkpoint (if implemented in the script).

## Reproducibility tips

- To reproduce results, use a fixed random seed at the top of the notebook or script (e.g., `torch.manual_seed(0)`).
- Clear notebook outputs before committing to keep the repository lightweight.

## Contributing

Contributions are welcome. For small improvements (typos, docs), please open a pull request. For larger code changes, open an issue first to discuss the plan.

## License

This project includes a `LICENSE` file — please refer to it for license details.

## Contact

If you have questions or need help, open an issue in this repository.

---

Happy experimenting with MNIST and PyTorch!