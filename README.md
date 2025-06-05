# Poly Control

This repository simulates a continuous stirred tank reactor used for copolymerisation and provides utilities for generating PI control trajectories. The rollouts can serve as offline data for reinforcement learning experiments.

## Repository structure

- `env.py` implements the `PolyCSTR` Gymnasium environment.
- `data.py` defines `PolyCSTRDataset` for loading rollout data.
- `utils.py` contains helper classes for scaling and plotting.
- `main.py` runs a sample simulation and stores results in `rollouts.csv`.

## Usage

1. Install the Python dependencies using `pip install -r requirements.txt`.
2. Execute `python main.py` to generate rollout data and display basic plots.
3. Use `data.PolyCSTRDataset` to load the generated CSV file for offline RL.
