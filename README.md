
# DeepXDE PINN Framework

This repository provides a modular framework for solving physics-informed neural network (PINN) problems using DeepXDE and PyTorch. It supports multiple physical domains (mechanic, heat, moisture and there Coupling).

## Project Structure

- `DeepXDE/`
  - `main.py`: Entry point. Parses command-line arguments and manages the workflow.
  - `config.py`: Configuration settings for diffrent models.
  - `domain_vars.py`: Domain-specific variables and parameters.
  - `manager.py`: Handles argument parsing and execution logic.
  - `MAP.py`: Implements a Routing logic.
  - `material.py`: Material property definitions.
  - `model.py`: PINN model architecture creation.
  - `points.py`: Point sampling/generation for training/testing.


- Each physical problem (e.g., mechanic, heat) is organized in its own folder, with relevant configuration, MAP, and material files.

- `utils/`: Shared utility scripts for data handling, model management, training, and more.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/7Solomon/PINNs.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Change directory to DeepXDE:
   ```bash
   cd DeepXDE
   ```

2. Run the main script with arguments. Example:
   ```bash
   python main.py add --type mechanic einspannung_2d --vis all --save
   ```

   - `add`: Action to perform (e.g., add a new model, load to load a existing folder).
   - `--type mechanic`: Selects the problem type/domain.
   - `einspannung_2d`: Specifies the particular problem/case.
   - `--vis all`: Enables all visualizations.
   - `--save`: Saves the results.

3. All problems are defined in their respective folders, each containing:
   - Residual Definition
   - Domain Definition
   - Scaling
   - gnd Definition
   - Visualisations functions


## Utilities

The `utils/` folder contains also the Callbacks that are used for Complexer Training Staging

