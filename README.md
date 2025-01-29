# Global Group Fairness in Federated Learning via Function Tracking

This repository contains the code implementation for the paper *"Global Group Fairness in Federated Learning via Function Tracking."* The implementation is structured to facilitate experimentation with the proposed method and includes scripts for running experiments on multiple datasets.

## Repository Structure

- `requirements.txt` - Lists the dependencies required to run the code.
- `setup.py` - Script to create the necessary folder structure for the project.
- `run_{***}.py` - Various scripts to execute different experiments.
- `src/` - Contains the core implementation of the method.
- `datasets/` - Folder where datasets should be placed (created by `setup.py`).

## Setup Instructions

### 1. Install Dependencies

Ensure you have Python installed (we used Python 3.10.12). Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Set Up Folder Structure

Run the setup script to create the necessary directories:

```bash
python setup.py
```

This will create the `datasets/` folder where required datasets should be placed.

### 3. Download and Place Datasets

The following datasets need to be manually downloaded and placed in the `datasets/` folder:

- [COMPAS Dataset](https://www.kaggle.com/danofer/compass)
- [Communities and Crime Dataset](https://archive.ics.uci.edu/ml/datasets/communities+and+crime)

Ensure the dataset files are correctly formatted and located in `datasets/` before running experiments.

## Running Experiments

Experiments can be run using the `run_{***}.py` scripts. Each script corresponds to a different experiment. Settings are passed as keyword arguments. For example:

```bash
python run_synthetic.py --method ours --home .
```

We use Weights & Biases (wandb) for tracking experiments. Our project names for logging are `fairFLConvergence` and `fairFL`. Ensure you have a Weights & Biases account and are logged in before running experiments:

```bash
wandb login
```

Refer to individual scripts for details on hyperparameters and configurations.

## Citation

For citing our work, you may use:

```
@article{global_fair_fl,
  title={Global Group Fairness in Federated Learning via Function Tracking},
  author={Yves Rychener, Daniel Kuhn,  Yifan Hu},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2025},
  organization={PMLR}
}
```

## Contact

For any issues or questions, please contact [Yves Rychener](https://github.com/yvesrychener).

