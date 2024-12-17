# CartPole DQN

## Overview

This project implements a Deep Q-Network (DQN) to solve the CartPole environment using reinforcement learning techniques. The DQN algorithm is a value-based method that combines Q-learning with deep neural networks to approximate the Q-value function.

## Requirements

To run this project, you need the following libraries:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Gym

You can install the required libraries using pip:

```bash
pip install torch numpy matplotlib gym
```

## Project Structure

```plaintext
.
├── lib
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── memory.py
│   ├── plotting.py
│   └── trainer.py
└── train.py
```

- **lib/**: Contains the core components of the DQN implementation.

  - **config.py**: Configuration settings for the DQN training.
  - **model.py**: Defines the neural network architecture.
  - **memory.py**: Implements the replay memory for storing experiences.
  - **plotting.py**: Functions for plotting training results.
  - **trainer.py**: Contains the DQN training logic.

- **train.py**: The main script to train the DQN agent.

## Usage

To train the DQN agent, run the following command:

```bash
python train.py
```

The training process will begin, and the results will be plotted at the end of the training.

## Configuration

The configuration for the DQN can be modified in the `train.py` file. Key parameters include:

- `run_name`: Name of the training run.
- `env_id`: The environment ID (default is "CartPole-v1").
- `device`: Device to run the training on (either "cpu" or "cuda").
- `n_steps`: Number of steps for TD(n) learning.
- `buffer_size`: Size of the replay buffer.
- `batch_size`: Number of samples per training step.
- `eps_start`, `eps_end`, `eps_decay`: Parameters for epsilon-greedy action selection.

## Training Process

The training process consists of the following steps:

1. **Initialization**: Set up the environment and the DQN agent.
2. **Training Loop**: For a specified number of steps:
   - Reset the environment and initialize variables.
   - Select actions using the epsilon-greedy policy.
   - Store experiences in the replay memory.
   - Update the Q-network based on experiences sampled from the replay memory.
   - Update the target network periodically.
3. **Evaluation**: After training, the mean reward from the last 10 episodes is printed.

## Plotting Results

The training results are plotted using Matplotlib, showing the episode rewards over time. The plot includes a horizontal line indicating the maximum score achievable in the CartPole environment.

## Acknowledgments

This implementation is based on the DQN algorithm as described in the original paper by Mnih et al. (2015).
