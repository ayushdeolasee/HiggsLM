# gpt2-style-transformer

This repository provides scripts and utilities for training a GPT-2 style transformer model on the FineWeb dataset (or its variants).

## Installation

1. Install dependencies (preferably in a virtual environment):
   ```sh
   pip install -r requirements.txt
   ```

## Downloading the Dataset

You can download and preprocess the FineWeb dataset (or any of its flavors) using the CLI:

```sh
python cli.py download-dataset --local-dir <output_dir> --dataset-flavor <flavor>
```

- `--local-dir`: Directory to save the processed dataset (default: `data`)
- `--dataset-flavor`: Dataset flavor to download (default: `fineweb10B`). Example: `fineweb10B-edu`, `fineweb10B`, etc.

**Example:**
```sh
python cli.py download-dataset --local-dir data --dataset-flavor fineweb10B
```

## Training the Model

You can train the model using the CLI:

```sh
python cli.py train-cli [OPTIONS]
```

### Training Options
- `--dataset-location`: Directory containing the processed dataset (default: `data`)
- `--epochs`: Number of epochs (default: 19073)
- `--batch-size`: Batch size (default: 4)
- `--block-size`: Block size (default: 1024)
- `--total-batch-size`: Total batch size (default: 524288)
- `--lr`: Learning rate (default: 3e-4)

**Example:**
```sh
python cli.py train-cli --dataset-location data --epochs 10 --batch-size 8
```

## Direct Usage

You can also run the dataset download script directly:

```sh
python fineweb.py --local_dir data --dataset_flavor fineweb10B
```

## Notes
- The dataset is sharded and saved in the specified directory.
- Training logs and model weights are saved in the working directory.

## Requirements
See `requirements.txt` for all dependencies.

## License
[Add your license here]
