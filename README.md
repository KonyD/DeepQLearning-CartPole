# Deep Q-Learning with CartPole

This project implements a Deep Q-Learning (DQL) agent to solve the CartPole-v1 environment using the Gymnasium library. The agent leverages a neural network to learn the optimal policy for balancing the pole on the cart.

## Features

- **Environment:** The project uses the CartPole-v1 environment from Gymnasium.
- **Neural Network:** A fully connected feedforward neural network is implemented using TensorFlow/Keras.
- **Replay Memory:** The agent uses experience replay to train on past experiences.
- **Epsilon-Greedy Strategy:** Balances exploration and exploitation during training.
- **Training and Testing Modes:** The trained model is tested to demonstrate its performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KonyD/DeepQLearning-CartPole.git
   cd DeepQLearning-CartPole 
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## File Structure

- `main.py`: The main script containing the implementation of the DQL agent and training/testing logic.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: Project documentation.

## Usage

1. **Train the Agent:**
   Run the script to train the agent:
   ```bash
   python main.py
   ```

   The agent will train for a specified number of episodes and log its performance.

2. **Test the Trained Model:**
   The script includes a testing phase where the trained model is evaluated in the CartPole environment.

## Key Parameters

- `gamma`: Discount factor for future rewards (default: `0.95`).
- `epsilon`: Initial exploration rate (default: `1`).
- `epsilon_decay`: Decay rate for exploration (default: `0.995`).
- `epsilon_min`: Minimum exploration rate (default: `0.01`).
- `batch_size`: Size of the minibatch used for training (default: `32`).
- `episodes`: Number of training episodes (default: `50`).

## Dependencies

- Python 3.8+
- Gymnasium
- TensorFlow/Keras
- NumPy
- tqdm

You can install the required libraries using the following command:
```bash
pip install -r requirements.txt
```

## Results

The agent is trained to balance the pole for as long as possible. During training, the epsilon-greedy strategy ensures the agent explores the environment initially and focuses on exploitation as training progresses.

## Acknowledgments

- [OpenAI Gymnasium](https://gymnasium.farama.org/) for providing the CartPole-v1 environment.
- [TensorFlow](https://www.tensorflow.org/) for neural network implementation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
