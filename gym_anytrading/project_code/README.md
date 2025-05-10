# Q-Learning for Algorithmic Trading

This project implements Q-Learning algorithms for trading in financial markets, supporting both tabular and tile coding (function approximation) approaches. You can train and evaluate agents using different exploration strategies and hyperparameters, and visualize the results.

---

## Table of Contents

- [Overview](#overview)
- [1. Environment Setup](#1-environment-setup)
- [2. Installing Dependencies](#2-installing-dependencies)
- [3. Preparing Configuration Files](#3-preparing-configuration-files)
- [4. Running Experiments](#4-running-experiments)
  - [Tabular Q-Learning](#tabular-q-learning)
  - [Tile Coding Q-Learning](#tile-coding-q-learning)
- [5. Using Custom Data](#5-using-custom-data)
- [6. Checking and Interpreting Results](#6-checking-and-interpreting-results)
- [7. Example Visualizations](#7-example-visualizations)
- [8. Troubleshooting](#8-troubleshooting)

---

## Overview

The system uses Q-learning, a reinforcement learning algorithm, to train agents that can trade in financial markets. Two exploration strategies are implemented:

- **Epsilon-greedy**: Balances exploration and exploitation using a decaying exploration rate
- **Thompson sampling** (tabular) or **UCB** (tile coding): Bayesian or upper-confidence-based exploration

You can choose between **tabular Q-learning** (for small, discrete state spaces) and **tile coding** (for function approximation in large/continuous state spaces).

---

## 1. Environment Setup

It's recommended to use a virtual environment to avoid dependency conflicts.

### Using `venv` (standard Python)

```bash
python3 -m venv ql-trading-env
source ql-trading-env/bin/activate
```

### Or using `conda`

```bash
conda create -n ql-trading python=3.10
conda activate ql-trading
```

---

## 2. Installing Dependencies

Install the required packages using pip:

```bash
pip install numpy matplotlib gymnasium gym-anytrading pandas
```

If you plan to use Jupyter notebooks for analysis, also install:

```bash
pip install jupyter
```

---

## 3. Preparing Configuration Files

Experiments are controlled via JSON config files. Example configs are provided in `project_code/configs/`:

- `config.json`: Standard config for tabular Q-learning
- `config_tile.json`: Config for tile coding Q-learning
- `config_multi_df.json`: Example for using different datasets for training and testing

### Example: `config.json` (tabular Q-learning)

```json
{
  "experiment_type": "single_df",
  "window_size": 10,
  "train_frame_bound": [10, 250],
  "test_frame_bound": [250, 350],
  "results_base_dir": "results_tabular",
  "learning_rates": [0.1, 0.01, 0.001],
  "exploration_strategies": ["epsilon-greedy", "thompson"],
  "num_episodes": 10000,
  "df_path": ""
}
```

### Example: `config_tile.json` (tile coding)

```json
{
  "experiment_type": "single_df",
  "window_size": 10,
  "num_tilings": 8,
  "num_tiles": 8,
  "memory_size": 4096,
  "train_frame_bound": [10, 250],
  "test_frame_bound": [250, 350],
  "results_base_dir": "results_tile",
  "learning_rates": [0.1, 0.01, 0.001],
  "exploration_strategies": ["ucb", "epsilon-greedy"],
  "num_episodes": 10000,
  "df_path": ""
}
```

- For custom data, set `df_path` to your CSV file path.
- For multi-dataset experiments, use `train_df_path` and `test_df_path`.

---

## 4. Running Experiments

### Tabular Q-Learning

Run from the `project_code` directory:

```bash
python q_learning.py --config configs/config.json
```

- You can specify a different config file with `--config`.
- Results will be saved in the directory specified by `results_base_dir` in your config.

### Tile Coding Q-Learning

Run from the `project_code` directory:

```bash
python q_learning_tile.py --config configs/config_tile.json
```

- You can specify a different config file with `--config`.
- Results will be saved in the directory specified by `results_base_dir` in your config.

---

## 5. Using Custom Data

You can use your own stock data in CSV format. The file **must** include at least these columns:

- `Open`: Opening price
- `Close`: Closing price

Optional columns: `High`, `Low`, `Volume`, `Date` (if present, will be used as index).

**Example:**

```csv
Date,Open,Close,High,Low,Volume
2020-01-01,100,105,106,99,10000
2020-01-02,105,103,107,102,12000
...
```

Set the `df_path` in your config to the path of your CSV file.

---

## 6. Checking and Interpreting Results

After running, results are saved in the directory specified by `results_base_dir`.

### What you'll find:

- **Training and testing performance plots** for each configuration
- **Comparison plots** across all configurations
- **Summary statistics** and identification of best configurations
- The **configuration** used for the experiment (`used_config.json`)

#### Example directory structure:

```
results_tabular/
├── epsilon-greedy/
│   └── lr0.1/
│       ├── q_learning_rewards_lr0.1.png
│       ├── trading_performance_lr0.1.png
│       └── ...
├── thompson/
│   └── lr0.1/
│       └── ...
├── comparison/
│   ├── learning_curves_by_strategy.png
│   ├── training_performance.png
│   ├── results_summary.txt
│   └── ...
└── used_config.json
```

### How to interpret:

- **Learning curves**: Show how agent reward improves over episodes.
- **Trading performance**: Visualizes trades and profit over time.
- **Comparison plots**: Help you pick the best learning rate and strategy.
- **Summary file**: Lists the best configurations and their performance.

---

## 7. Example Visualizations

The system generates several types of visualizations:

1. **Learning curves** by strategy and learning rate
2. **Bar charts** comparing training and testing performance
3. **Heatmaps** showing performance across all configurations
4. **Generalization plots** showing training vs testing performance
5. **Complete trading performance charts**
6. (Tile coding only) **Weight visualizations** for feature importance

---

## 8. Troubleshooting

- **Module not found**: Make sure you installed all dependencies in the correct environment.
- **No results generated**: Check for errors in the terminal and ensure your config file paths are correct.
- **Custom data errors**: Ensure your CSV has at least `Open` and `Close` columns, and that the path is correct.
- **Plots not showing**: All plots are saved as PNG files in the results directory; open them with any image viewer.

---

## Need Help?

If you encounter issues, check the error messages and ensure your environment and config files are set up as described above. For further help, open an issue or contact the project maintainer.
