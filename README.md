# DL Hyperparameter Tuning on HPC

This repository contains code and configuration files for hyperparameter tuning of a deep learning model (MNIST digit classification) on a High-Performance Computing (HPC) cluster using SLURM job scheduling.

## 📌 Project Overview

The goal of this project is to:
- Train a simple neural network on the MNIST dataset.
- Explore the effects of learning rate and batch size on model performance.
- Automate multiple training jobs on a GPU-based HPC system.
- Collect and analyze the results to determine optimal hyperparameters.

## 📁 Repository Structure

```
dl-hyperparam-tuning-hpc/
├── train.py             # Main training script
├── job_train.slurm      # SLURM job submission script
├── submit_all.sh        # Bash script to launch multiple jobs with different hyperparameters
├── logs/                # Output logs (.out and .err) for each job
├── results/             # Folder to store any processed results or summaries
├── mnist_data/          # MNIST dataset (auto-downloaded by PyTorch if not present)
└── README.md            # Project documentation
```

## 🚀 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/mirjunaid26/dl-hyperparam-tuning-hpc.git
cd dl-hyperparam-tuning-hpc
```

### 2. Activate Environment

Make sure you are on a GPU node and have your Python environment set up with PyTorch.

```bash
conda activate gpu-env
```

### 3. Submit Jobs

To launch jobs with various combinations of learning rates and batch sizes:

```bash
./submit_all.sh
```

This runs a loop over multiple values and submits jobs via `sbatch`.

### 4. Check Job Status

```bash
squeue --me
```

### 5. Monitor Logs

Logs will appear in the `logs/` folder:

```bash
cat logs/mnist_lr<job_id>.out
```

## 📊 Example Results

Some example results from training jobs:

| Learning Rate | Batch Size | Test Accuracy |
|---------------|------------|----------------|
| 0.005         | 64         | 98.94%         |
| 0.005         | 32         | 98.84%         |
| ...           | ...        | ...            |

## 📦 Dependencies

- Python 3.x
- PyTorch
- SLURM (for job scheduling)
- (Optional) Matplotlib / pandas for result analysis

## 📜 License

MIT License

## 🙌 Acknowledgements

This project is run on the amplitUDE which is a latest Nvidia H100 based HPC cluster of University of Duisburg-Essen. Thanks to PyTorch and the open MNIST dataset contributors.

