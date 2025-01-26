import gymnasium as gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import random
from tqdm import tqdm

# Define the Deep Q-Learning Agent
class DQLAgent:
    def __init__(self, env):
        # Initialize the state size and action size based on the environment
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        # Hyperparameters for training
        self.gamma = 0.95  # Discount factor for future rewards
        self.learning_rate = 0.001
        self.epsilon = 1  # Initial exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.epsilon_min = 0.01  # Minimum exploration rate

        # Memory to store experiences (state, action, reward, next state, done)
        self.memory = deque(maxlen=1000)

        # Build the neural network model
        self.model = self.build_model()
    
    # Build the Q-network
    def build_model(self):
        model = Sequential()
        # Input layer
        model.add(Dense(48, input_dim=self.state_size, activation="relu"))
        # Hidden layer
        model.add(Dense(24, activation="relu"))
        # Output layer
        model.add(Dense(self.action_size, activation="linear"))
        # Compile the model using Mean Squared Error loss and Adam optimizer
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    # Store an experience in memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    # Choose an action using an epsilon-greedy strategy
    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            # Explore: choose a random action
            return env.action_space.sample()
        # Exploit: predict the action with the highest Q-value
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    # Train the model using a random sample of experiences
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Skip training if memory doesn't have enough samples

        # Sample a random batch of experiences
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            if done:
                # If the episode ended, the target is just the reward
                target = reward
            else:
                # Otherwise, use the Bellman equation to compute the target
                target = reward + self.gamma * np.max(self.model.predict(next_state, verbose=0)[0])
            
            # Update the Q-value for the chosen action
            train_target = self.model.predict(state, verbose=0)
            train_target[0][action] = target
            
            # Train the model with the updated Q-value
            self.model.fit(state, train_target, verbose=0)
    
    # Gradually reduce epsilon to reduce exploration over time
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")
agent = DQLAgent(env)

# Hyperparameters
batch_size = 32
episodes = 50

# Train the agent
for e in tqdm(range(episodes)):
    state = env.reset()[0]
    state = np.reshape(state, [1, 4])  # Reshape the state for the neural network
    
    time = 0
    
    while True:
        # Select an action
        action = agent.act(state)
        
        # Perform the action and observe the next state and reward
        (next_state, reward, done, _, _) = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        
        # Store the experience in memory
        agent.remember(state, action, reward, next_state, done)
        
        # Update the state
        state = next_state
        
        # Train the model
        agent.replay(batch_size)
        
        # Adjust epsilon for exploration/exploitation tradeoff
        agent.adaptiveEGreedy()
        
        time += 1
        
        # End the episode if done
        if done:
            print(f"Episode: {e}, time: {time}")
            break

# Test the trained model
import time 

trained_model = agent

# Reload the environment for testing
env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()[0]
state = np.reshape(state, [1, 4])

time_t = 0

while True:
    # Render the environment
    env.render()

    # Select an action using the trained model
    action = trained_model.act(state)

    # Perform the action and observe the next state
    (next_state, reward, done, _, _) = env.step(action)
    next_state = np.reshape(next_state, [1, 4])
    state = next_state

    time_t += 1
    
    print(f"Time: {time_t}")
    
    # Slow down rendering for visualization
    time.sleep(0.5)
    
    # End the loop if the episode finishes
    if done:
        break

print("Done")
