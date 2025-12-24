
# Stack: In-context modeling of single-cell biology

Stack is a large-scale encoder-decoder foundation model trained on 150 million uniformly-preprocessed single cells. It introduces a novel tabular attention architecture that enables both intra- and inter-cellular information flow, setting cell-by-gene matrix chunks as the basic input data unit. Through in-context learning, Stack offers substantial performance improvements in generalizing biological effects and enables generation of unseen cell profiles in novel contexts.

## Installation

### Using pip
```bash
# Install from PyPI
pip install arc-stack

# Or install from source for development
git clone https://github.com/ArcInstitute/stack.git
cd stack
pip install -e .
```

### Using uv
```bash
# Install from PyPI
uv pip install arc-stack

# Or install from source for development
git clone https://github.com/ArcInstitute/stack.git
cd stack
uv pip install -e .
```


## Quick Start

- Use Stack to embed your single-cell data: [Notebook](notebooks/tutorial-embed.ipynb)
- Use Stack to zero-shot predict unseen perturbation/observation profiles: [Notebook](notebooks/tutorial-predict.ipynb)

### Training Stack from Scratch

```bash
# Once installed, the console entry point becomes available
stack-train \
    --dataset_configs "/path/to/data:false:gene_symbols" \
    --genelist_path "hvg_genes.pkl" \
    --save_dir "./checkpoints" \
    --sample_size 256 \
    --batch_size 32 \
    --n_hidden 100 \
    --token_dim 16 \
    --n_layers 9 \
    --max_epochs 10

# Alternatively, invoke the module directly when working from a cloned repo
python -m stack.cli.launch_training [args...]
```

### Fine-tuning Stack with Frozen Teacher

```bash
stack-finetune \
    --checkpoint_path "./checkpoints/pretrained.ckpt" \
    --dataset_configs "human:/path/to/data:donor_id:cell_type:false" \
    --genelist_path "hvg_genes.pkl" \
    --save_dir "./finetuned_checkpoints" \
    --sample_size 512 \
    --batch_size 8 \
    --replacement_ratio 0.75 \
    --max_epochs 8

# Or use uv run
uv run stack-finetune [args...]

# Repository wrapper remains available for local development
python -m stack.cli.launch_finetuning [args...]
```

### Running Stack with configuration files

Both `launch_training.py` and `launch_finetuning.py` accept a `--config` flag that points to a YAML or JSON file. Any command line
arguments omitted after `--config` inherit their values from the file, while flags provided on the command line override the
configuration. Example configs mirroring the provided Slurm scripts live under `configs/`:

```bash
# Train with the preset configuration
stack-train --config configs/training/bc_large.yaml

# Override a single hyperparameter without editing the file
stack-train --config configs/training/bc_large.yaml --learning_rate 5e-5

# Fine-tune using a config file
stack-finetune --config configs/finetuning/ft_parsecg.yaml

# Direct module invocation is still supported if you prefer python -m
python -m stack.cli.launch_training --config configs/training/bc_large.yaml
```

> **Note:** YAML configs require [`pyyaml`](https://pyyaml.org/). Install it with `pip install pyyaml` or use a JSON config file.

### Extracting Stack Embeddings

```bash
stack-embedding \
    --checkpoint "./checkpoints/pretrained.ckpt" \
    --adata "data.h5ad" \
    --genelist "hvg_genes.pkl" \
    --output "embeddings.h5ad" \
    --batch-size 32

# Or use uv run
uv run stack-embedding \
    --checkpoint "./checkpoints/pretrained.ckpt" \
    --adata "data.h5ad" \
    --genelist "hvg_genes.pkl" \
    --output "embeddings.h5ad" \
    --batch-size 32
```

### In-Context Generation with Stack

```bash
stack-generation \
    --checkpoint "./checkpoints/pretrained.ckpt" \
    --base-adata "base_data.h5ad" \
    --test-adata "test_data.h5ad" \
    --genelist "hvg_genes.pkl" \
    --output-dir "./generations" \
    --split-column "donor_id"

# Or use uv run
uv run stack-generation \
    --checkpoint "./checkpoints/pretrained.ckpt" \
    --base-adata "base_data.h5ad" \
    --test-adata "test_data.h5ad" \
    --genelist "hvg_genes.pkl" \
    --output-dir "./generations" \
    --split-column "donor_id"
```

## Model Architecture

- **Tabular Attention**: Alternating cell-wise and gene-wise attention layers
- **Token Dimension**: Configurable token embedding dimension (default: 16)
- **Hidden Dimension**: Gene dimension reduction (default: 100)
- **Masking Strategy**: Rectangular masking with variable rates (0.1-0.8)

## Data Preparation

### Computing Highly Variable Genes (HVGs)
```python
from stack.data.datasets import DatasetConfig, compute_hvg_union

configs = [DatasetConfig(path="/data/path", filter_organism=True)]
hvg_genes = compute_hvg_union(configs, n_top_genes=1000, output_path="hvg.pkl")
```

### Dataset Configuration Format
- **Human datasets**: `human:/path:donor_col:cell_type_col[:filter_organism[:gene_col]]`
- **Drug datasets**: `drug:/path:condition_col:cell_line_col:control_condition[:filter_organism[:gene_col]]`

## Key Features

- **In-Context Learning**: Zero-shot generalization to new biological contexts
- **Multi-Dataset Training**: Simultaneous training on multiple single-cell datasets
- **Frozen Teacher Fine-tuning**: Novel fine-tuning procedure with stable teacher targets
- **Efficient Data Loading**: Optimized HDF5 loading with sparse matrix support

> **Note:** `scShiftAttentionModel` remains available as an alias for backward compatibility.

## Citation

If you use Stack in your research, please cite the Stack [paper](https://arcinstitute.org/manuscripts/Stack).
