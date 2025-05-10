import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import pandas as pd
import json
import argparse
import sys
import hashlib

import gym_anytrading
from gym_anytrading.envs import StocksEnv, Actions, Positions 
from gym_anytrading.datasets import STOCKS_GOOGL


class TileCoder:
    """
    Tile coding for function approximation in Q-Learning.
    Uses hashing to prevent memory explosion with high-dimensional states.
    """
    def __init__(self, 
                 num_tilings=8, 
                 num_tiles=8, 
                 state_low=None, 
                 state_high=None, 
                 state_dimensions=None,
                 memory_size=4096):  # Use a reasonable fixed memory size
        """
        Initialize the tile coder
        
        Args:
            num_tilings: Number of overlapping tilings to use
            num_tiles: Number of tiles per dimension
            state_low: Lower bounds for each dimension of the observation space
            state_high: Upper bounds for each dimension of the observation space
            state_dimensions: Number of dimensions to consider from the state
            memory_size: Size of the memory array (number of possible tiles)
        """
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.state_low = np.array(state_low)
        self.state_high = np.array(state_high)
        self.state_dimensions = state_dimensions
        self.memory_size = memory_size
        
        # Calculate the size of each tile in each dimension
        self.tile_size = (self.state_high - self.state_low) / self.num_tiles
        
        # Set total_tiles to the memory size (this is what the function approximation will use)
        self.total_tiles = self.memory_size
        
        # Offset for each tiling to create the overlap
        self.offsets = np.zeros((self.num_tilings, self.state_dimensions))
        for i in range(self.num_tilings):
            for j in range(self.state_dimensions):
                self.offsets[i, j] = -i * self.tile_size[j] / self.num_tilings
    
    def _hash_coordinates(self, coordinates, tiling):
        """Hash the tile coordinates to a fixed-size memory index"""
        # Convert coordinates to string and add tiling as a prefix
        state_str = f"{tiling}:{','.join(map(str, coordinates))}"
        
        # Hash the string using MD5 (any deterministic hash would work)
        hash_value = int(hashlib.md5(state_str.encode()).hexdigest(), 16)
        
        # Map the hash to the memory size
        return hash_value % self.memory_size
    
    def get_tiles(self, state):
        """
        Convert a continuous state to active tile indices
        
        Args:
            state: Continuous state vector
            
        Returns:
            List of active tile indices
        """
        # Handle different observation types
        if isinstance(state, tuple):
            state = state[0]
        
        # For trading environment, we'll use the last row (most recent data)
        if len(state.shape) > 1:
            flat_state = state[-1]
        else:
            flat_state = state
        
        # Ensure the flattened state matches the expected dimensions
        if len(flat_state) != self.state_dimensions:
            if len(flat_state) > self.state_dimensions:
                flat_state = flat_state[:self.state_dimensions]
            else:
                flat_state = np.pad(flat_state, (0, self.state_dimensions - len(flat_state)), 'constant')
        
        # Clip the state to be within bounds
        clipped_state = np.clip(flat_state, self.state_low, self.state_high)
        
        # Normalize the state to [0, num_tiles)
        normalized_state = (clipped_state - self.state_low) / self.tile_size
        
        active_tiles = []
        for tiling in range(self.num_tilings):
            # Apply the offset for this tiling
            offset_state = normalized_state + self.offsets[tiling]
            
            # Get the tile coordinates for this tiling
            tile_coordinates = np.floor(offset_state).astype(int)
            
            # Ensure coordinates are within valid range
            tile_coordinates = np.clip(tile_coordinates, 0, self.num_tiles - 1)
            
            # Hash the coordinates to get a tile index
            tile_index = self._hash_coordinates(tile_coordinates, tiling)
            
            active_tiles.append(tile_index)
        
        return active_tiles


class QLearningTileAgent:
    """
    Q-Learning agent with tile coding and function approximation.
    """
    def __init__(self, action_space, tile_coder, 
                 learning_rate=0.1, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.995, 
                 min_exploration_rate=0.02):
        """
        Initialize the Q-Learning agent with tile coding
        
        Args:
            action_space: The action space of the environment
            tile_coder: The tile coder to convert continuous observations to feature vectors
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Rate at which exploration rate decays
            min_exploration_rate: Minimum exploration rate
        """
        self.action_space = action_space
        self.tile_coder = tile_coder
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize weights for function approximation
        # For each action, we have a weight vector of the size of the total tiles
        self.weights = np.zeros((self.action_space.n, self.tile_coder.total_tiles))
    
    def get_q_value(self, state, action):
        """
        Calculate the Q-value for a state-action pair using function approximation
        
        Args:
            state: The current state
            action: The action
            
        Returns:
            The estimated Q-value
        """
        active_tiles = self.tile_coder.get_tiles(state)
        return np.sum(self.weights[action, active_tiles])
    
    def get_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        if np.random.random() < self.exploration_rate:
            return self.action_space.sample()
        else:
            q_values = np.array([self.get_q_value(state, a) for a in range(self.action_space.n)])
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """Update the weights using Q-learning update rule with function approximation"""
        # Get the active tiles for the current state
        active_tiles = self.tile_coder.get_tiles(state)
        
        # Calculate the current Q-value
        current_q = self.get_q_value(state, action)
        
        # Calculate the target value
        if done:
            target = reward
        else:
            # Find the best action in the next state
            next_q_values = np.array([self.get_q_value(next_state, a) for a in range(self.action_space.n)])
            best_next_action = np.argmax(next_q_values)
            target = reward + self.discount_factor * self.get_q_value(next_state, best_next_action)
        
        # Calculate the TD error
        td_error = target - current_q
        
        # Update weights for the active tiles
        for tile in active_tiles:
            self.weights[action, tile] += self.learning_rate * td_error / len(active_tiles)
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(self.min_exploration_rate, 
                                      self.exploration_rate * self.exploration_decay)


class QLearningTileAgentUCB:
    """
    Q-Learning agent with tile coding and Upper Confidence Bound (UCB) exploration.
    """
    def __init__(self, action_space, tile_coder, learning_rate=0.1, discount_factor=0.99, 
                 exploration_weight=0.8, ucb_constant=2.0):
        """
        Initialize the Q-Learning agent with tile coding and UCB exploration
        
        Args:
            action_space: The action space of the environment
            tile_coder: The tile coder to convert continuous observations to feature vectors
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_weight: Weight given to exploration term vs exploitation (0.5 = equal weight)
            ucb_constant: Constant factor for UCB exploration term (higher = more exploration)
        """
        self.action_space = action_space
        self.tile_coder = tile_coder
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_weight = exploration_weight
        self.ucb_constant = ucb_constant
        
        # Initialize weights for function approximation
        self.weights = np.zeros((self.action_space.n, self.tile_coder.total_tiles))
        
        # Track visits for each tile-action pair for UCB strategy
        self.tile_visits = {}   # (tile_index, action) -> visit count
        
        # Track step count for reward normalization and exploration adjustment
        self.step_count = 0
        
        # Normalize rewards to [0, 1] for scaling
        self.max_reward_seen = 1.0
        self.min_reward_seen = 0.0
        
        # Action counts for tracking action distribution
        self.action_counts = np.zeros(self.action_space.n)
    
    def get_q_value(self, state, action):
        """
        Calculate the Q-value for a state-action pair using function approximation
        
        Args:
            state: The current state
            action: The action
            
        Returns:
            The estimated Q-value
        """
        active_tiles = self.tile_coder.get_tiles(state)
        return np.sum(self.weights[action, active_tiles])
    
    def get_action(self, state):
        """Choose an action using UCB policy"""
        # Get the active tiles for this state
        active_tiles = self.tile_coder.get_tiles(state)
        
        # Get Q-values for all actions in this state
        q_values = np.array([self.get_q_value(state, a) for a in range(self.action_space.n)])
        
        # Calculate UCB term for each action
        ucb_values = np.zeros(self.action_space.n)
        for a in range(self.action_space.n):
            # Count total visits for all tiles for this action
            total_visits = 1  # Initialize with 1 to avoid division by zero
            
            for tile in active_tiles:
                total_visits += self.tile_visits.get((tile, a), 0)
            
            # Calculate average visits per tile
            avg_visits = total_visits / len(active_tiles)
            
            # UCB formula: exploration term is proportional to sqrt(ln(total_steps) / visits)
            # Higher for less visited actions, ensures exploration
            ucb_term = self.ucb_constant * np.sqrt(np.log(max(1, self.step_count)) / avg_visits)
            
            # Store UCB value
            ucb_values[a] = ucb_term
        
        # Normalize both Q-values and UCB values to [0,1]
        if np.max(q_values) > np.min(q_values):
            normalized_q = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
        else:
            normalized_q = np.ones_like(q_values) / len(q_values)
            
        if np.max(ucb_values) > 0:
            normalized_ucb = ucb_values / np.max(ucb_values)
        else:
            normalized_ucb = np.ones_like(ucb_values) / len(ucb_values)
        
        # Combine Q-values and UCB (weighted by exploration weight)
        combined_values = (1 - self.exploration_weight) * normalized_q + self.exploration_weight * normalized_ucb
        
        # Select action
        action = np.argmax(combined_values)
        
        # Update action counts for debugging
        self.action_counts[action] += 1
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Update the weights and exploration counts"""
        # Increment step count
        self.step_count += 1
        
        # Get the active tiles for the current state
        active_tiles = self.tile_coder.get_tiles(state)
        
        # Calculate the current Q-value
        current_q = self.get_q_value(state, action)
        
        # Calculate the target value
        if done:
            target = reward
        else:
            # Find the best action in the next state
            next_q_values = np.array([self.get_q_value(next_state, a) for a in range(self.action_space.n)])
            best_next_action = np.argmax(next_q_values)
            target = reward + self.discount_factor * self.get_q_value(next_state, best_next_action)
        
        # Calculate the TD error
        td_error = target - current_q
        
        # Update weights for the active tiles
        for tile in active_tiles:
            self.weights[action, tile] += self.learning_rate * td_error / len(active_tiles)
            
            # Update visit counts for UCB strategy
            self.tile_visits[(tile, action)] = self.tile_visits.get((tile, action), 0) + 1
        
        # Update reward scaling
        self.max_reward_seen = max(self.max_reward_seen, reward)
        self.min_reward_seen = min(self.min_reward_seen, reward)
    
    def get_action_distribution(self):
        """Return the normalized distribution of selected actions"""
        total = np.sum(self.action_counts)
        if total > 0:
            return self.action_counts / total
        return np.ones_like(self.action_counts) / len(self.action_counts)




def create_environment(env_id='stocks-v0', window_size=10, frame_bound=(50, 200), df=None):
    """Create and return a configured trading environment
    
    Args:
        env_id: ID of the gym environment to create
        window_size: Size of the observation window
        frame_bound: Tuple of (start, end) indices for the data
        df: Optional pandas DataFrame with custom data. If None, uses the default dataset
    """
    if df is not None:
        env = gym.make(env_id, df=df, frame_bound=frame_bound, window_size=window_size)
    else:
        env = gym.make(env_id, frame_bound=frame_bound, window_size=window_size)
    
    return env


def create_tile_coder(window_size=10, num_tilings=8, num_tiles=8, 
                     price_range=(190, 320), diff_range=(-20, 20),
                     memory_size=4096):
    """
    Create and return a configured tile coder
    
    Args:
        window_size: Size of the observation window
        num_tilings: Number of overlapping tilings to use
        num_tiles: Number of tiles per dimension
        price_range: Tuple of (min_price, max_price)
        diff_range: Tuple of (min_diff, max_diff)
        memory_size: Size of the memory array (number of possible tiles)
    """
    # In the trading environment, each row of the observation has 2 features: price and diff
    # So for a window_size of 10, we have 20 dimensions (10 prices and 10 diffs)
    state_dimensions = window_size * 2
    
    # Create bounds arrays for all dimensions
    price_lower, price_upper = price_range
    diff_lower, diff_upper = diff_range
    
    lower_bounds = np.array([[price_lower, diff_lower]] * window_size).flatten()
    upper_bounds = np.array([[price_upper, diff_upper]] * window_size).flatten()
    
    return TileCoder(
        num_tilings=num_tilings,
        num_tiles=num_tiles,
        state_low=lower_bounds,
        state_high=upper_bounds,
        state_dimensions=state_dimensions,
        memory_size=memory_size
    )


def train_agent(env, agent, num_episodes=10000, max_steps=1000):
    """Train the agent and return training metrics"""
    episode_rewards = []
    episode_profits = []
    
    # For tracking the smoothed reward during training
    smoothed_rewards = []
    window_size = 50  # Window size for real-time smoothing
    
    # Calculate print frequency to show exactly 10 progress updates
    print_freq = max(1, num_episodes // 10)
    
    for episode in range(num_episodes):
        observation, _ = env.reset(seed=episode)
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(observation, action, reward, next_observation, done)
            observation = next_observation
            total_reward += reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_profits.append(info['total_profit'])
        
        # Calculate smoothed reward for the current window
        if len(episode_rewards) >= window_size:
            current_smoothed = np.mean(episode_rewards[-window_size:])
        else:
            current_smoothed = np.mean(episode_rewards)
            
        smoothed_rewards.append(current_smoothed)
        
        if (episode + 1) % print_freq == 0 or episode == num_episodes - 1:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_profit = np.mean(episode_profits[-print_freq:])
            
            # Check if the agent has an exploration_rate attribute (epsilon-greedy)
            if hasattr(agent, 'exploration_rate'):
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, "
                      f"Smoothed: {current_smoothed:.2f}, "
                      f"Avg Profit: {avg_profit:.4f}, Exploration Rate: {agent.exploration_rate:.4f}")
            else:
                # For Thompson sampling agents
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, "
                      f"Smoothed: {current_smoothed:.2f}, "
                      f"Avg Profit: {avg_profit:.4f}")
    
    return episode_rewards, episode_profits, smoothed_rewards


def evaluate_agent(agent, test_env, seed=1000):
    """Evaluate the trained agent on a different time period and return evaluation metrics"""
    observation, _ = test_env.reset(seed=seed)
    done = False
    total_reward = 0
    
    while not done:
        # Get Q-values for all actions
        q_values = np.array([agent.get_q_value(observation, a) for a in range(agent.action_space.n)])
        action = np.argmax(q_values)
        
        observation, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    return total_reward, info['total_profit']


def plot_training_results(episode_rewards, learning_rate, results_dir):
    """Plot and save training results with smoothed moving average"""
    plt.figure(figsize=(12, 5))
    
    # Plot raw rewards
    plt.plot(episode_rewards, alpha=0.3, color='lightblue', label='Raw Rewards')
    
    # Calculate and plot smoothed moving average
    window_size = min(100, len(episode_rewards) // 10)  # Adaptive window size
    if window_size < 2:
        window_size = 2
    
    smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(episode_rewards)), smoothed_rewards, 
             linewidth=2, color='blue', label=f'Moving Avg (window={window_size})')
    
    plt.title(f'Total Reward vs Episode (Learning Rate = {learning_rate})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'q_learning_tile_rewards_lr{learning_rate}.png'))
    plt.close()


def plot_trading_performance(env, learning_rate, total_profit, results_dir):
    """Plot and save trading performance"""
    plt.figure(figsize=(15, 6))
    env.unwrapped.render_all()
    plt.title(f"Trading Performance (LR = {learning_rate}) - Total Profit: {total_profit:.4f}")
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'trading_performance_lr{learning_rate}.png'))
    plt.close()


def plot_weight_visualization(agent, results_dir, learning_rate):
    """
    Visualize the weights of the agent to understand what features are important
    
    Args:
        agent: The trained agent
        results_dir: Directory to save the visualization
        learning_rate: Learning rate used for training (for filename)
    """
    plt.figure(figsize=(12, 6))
    
    # Get average weight magnitude for each action
    avg_weights = np.mean(np.abs(agent.weights), axis=1)
    
    # Plot weight distribution for each action
    actions = ['Buy/Long', 'Sell/Short']
    for i, action_name in enumerate(actions):
        plt.subplot(1, 2, i+1)
        plt.hist(agent.weights[i], bins=50, alpha=0.7)
        plt.title(f'Weights for {action_name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'weight_distribution_lr{learning_rate}.png'))
    plt.close()
    
    # Plot the most significant weights for each action
    plt.figure(figsize=(14, 6))
    num_top_weights = min(50, agent.weights.shape[1])  # Show top 50 weights or all if less
    
    for i, action_name in enumerate(actions):
        plt.subplot(1, 2, i+1)
        # Get indices of largest magnitude weights
        top_indices = np.argsort(np.abs(agent.weights[i]))[-num_top_weights:]
        top_weights = agent.weights[i][top_indices]
        
        # Plot horizontally
        plt.barh(range(len(top_weights)), top_weights, color='blue' if i == 0 else 'red')
        plt.title(f'Top {num_top_weights} Weights for {action_name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Feature Index')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'top_weights_lr{learning_rate}.png'))
    plt.close()


def run_experiment(learning_rate, 
                  window_size=10,
                  num_tilings=8,
                  num_tiles=8,
                  train_frame_bound=(50, 200),
                  test_frame_bound=(200, 350),
                  num_episodes=10000, 
                  exploration_strategy='epsilon-greedy',
                  df=None,
                  price_range=None,
                  diff_range=None,
                  memory_size=4096,
                  results_base_dir='results_tile',
                  **kwargs):
    """
    Run a complete experiment with the specified learning rate and exploration strategy
    
    Args:
        learning_rate: Learning rate for Q-learning
        window_size: Size of the observation window (same for both environments)
        num_tilings: Number of tilings for tile coding
        num_tiles: Number of tiles per dimension for tile coding
        train_frame_bound: Tuple of (start, end) indices for training data
        test_frame_bound: Tuple of (start, end) indices for testing data
        num_episodes: Number of episodes to train
        exploration_strategy: Either 'epsilon-greedy' or 'ucb'
        df: Optional pandas DataFrame with custom data
        price_range: Optional tuple of (min_price, max_price) for discretization
        diff_range: Optional tuple of (min_diff, max_diff) for discretization
        memory_size: Size of the memory array (number of possible tiles)
        results_base_dir: Base directory to save results
        **kwargs: Additional arguments for the agent initialization
    """
    # Create training and testing environments
    train_env = create_environment(window_size=window_size, frame_bound=train_frame_bound, df=df)
    test_env = create_environment(window_size=window_size, frame_bound=test_frame_bound, df=df)
    
    # Determine price and diff ranges based on the dataframe
    if price_range is None or diff_range is None:
        # Get min and max prices from the environment's dataframe
        if hasattr(train_env.unwrapped, 'prices'):
            prices = train_env.unwrapped.prices
            min_price, max_price = np.min(prices), np.max(prices)
            
            # Calculate approximate price difference range
            price_diffs = np.diff(prices)
            min_diff, max_diff = np.min(price_diffs), np.max(price_diffs)
            
            # Add margins to ensure all values are covered
            price_margin = (max_price - min_price) * 0.1
            diff_margin = max(abs(min_diff), abs(max_diff)) * 0.2
            
            if price_range is None:
                price_range = (min_price - price_margin, max_price + price_margin)
            
            if diff_range is None:
                diff_range = (min_diff - diff_margin, max_diff + diff_margin)
        else:
            # Default fallback ranges
            price_range = price_range or (190, 320)
            diff_range = diff_range or (-20, 20)
    
    # Create tile coder
    tile_coder = create_tile_coder(
        window_size=window_size,
        num_tilings=num_tilings,
        num_tiles=num_tiles,
        price_range=price_range,
        diff_range=diff_range,
        memory_size=memory_size
    )
    
    # Create agent based on exploration strategy
    if exploration_strategy == 'epsilon-greedy':
        agent = QLearningTileAgent(
            action_space=train_env.action_space,
            tile_coder=tile_coder,
            learning_rate=learning_rate,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.995,
            min_exploration_rate=0.02,
            **kwargs
        )
    elif exploration_strategy == 'ucb':
        agent = QLearningTileAgentUCB(
            action_space=train_env.action_space,
            tile_coder=tile_coder,
            learning_rate=learning_rate,
            discount_factor=0.99,
            exploration_weight=0.8,
            ucb_constant=2.0,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown exploration strategy: {exploration_strategy}")
    
    # Train agent
    print(f"Training agent with {exploration_strategy}, learning rate = {learning_rate}")
    print(f"Tile coding: {num_tilings} tilings with {num_tiles} tiles per dimension")
    print(f"Total features: {tile_coder.total_tiles}")
    
    episode_rewards, episode_profits, smoothed_rewards = train_agent(train_env, agent, num_episodes)
    
    # Create results directory
    results_dir = os.path.join(results_base_dir, exploration_strategy, f'lr{learning_rate}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot training results
    plot_training_results(episode_rewards, learning_rate, results_dir)
    
    # Evaluate agent on training environment
    print(f"\nEvaluating trained agent on training environment ({exploration_strategy}, Learning Rate = {learning_rate})...")
    train_reward, train_profit = evaluate_agent(agent, train_env)
    print(f"Train Evaluation - Total Reward: {train_reward:.2f}, Total Profit: {train_profit:.4f}")
    
    # Plot training performance
    plot_trading_performance(train_env, learning_rate, train_profit, 
                             os.path.join(results_dir, 'train'))
    
    # Evaluate agent on testing environment
    print(f"\nEvaluating trained agent on testing environment ({exploration_strategy}, Learning Rate = {learning_rate})...")
    test_reward, test_profit = evaluate_agent(agent, test_env)
    print(f"Test Evaluation - Total Reward: {test_reward:.2f}, Total Profit: {test_profit:.4f}")
    
    # Plot testing performance
    plot_trading_performance(test_env, learning_rate, test_profit, 
                             os.path.join(results_dir, 'test'))
    
    # Plot weight visualization
    plot_weight_visualization(agent, results_dir, learning_rate)
    
    # Plot smoothed rewards separately
    plt.figure(figsize=(12, 5))
    plt.plot(smoothed_rewards, linewidth=2)
    plt.title(f'Smoothed Reward vs Episode (Learning Rate = {learning_rate})')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward (window=50)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'smoothed_rewards_lr{learning_rate}.png'))
    plt.close()
    
    # If the agent has an action distribution method, plot it
    if hasattr(agent, 'get_action_distribution'):
        action_dist = agent.get_action_distribution()
        plt.figure(figsize=(8, 5))
        plt.bar(['Buy/Long', 'Sell/Short'], action_dist)
        plt.title('Action Distribution')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(results_dir, f'action_distribution_lr{learning_rate}.png'))
        plt.close()
    
    return {
        'learning_rate': learning_rate,
        'exploration_strategy': exploration_strategy,
        'num_tilings': num_tilings,
        'num_tiles': num_tiles,
        'episode_rewards': episode_rewards,
        'episode_profits': episode_profits,
        'smoothed_rewards': smoothed_rewards,
        'final_avg_reward': np.mean(episode_rewards[-100:]),
        'final_avg_profit': np.mean(episode_profits[-100:]),
        'train_reward': train_reward,
        'train_profit': train_profit,
        'test_reward': test_reward,
        'test_profit': test_profit,
        'total_features': tile_coder.total_tiles
    } 


def load_custom_data(file_path):
    """
    Load a custom dataframe from a CSV file
    
    Args:
        file_path: Path to the CSV file containing the data
        
    Returns:
        A pandas DataFrame formatted for use with the trading environment
    """
    # Load the data from CSV
    df = pd.read_csv(file_path)
    
    # Ensure the dataframe has the required columns for the trading environment
    # At minimum, we need 'Open' and 'Close' columns, but additional columns like
    # 'High', 'Low', and 'Volume' can be helpful
    required_columns = ['Open', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataframe")
            
    # Handle datetime column if needed
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    return df


def compare_results(results_dict, results_base_dir='results'):
    """
    Create comprehensive comparison plots for the experiment results
    
    Args:
        results_dict: Dictionary containing results from multiple experiments
        results_base_dir: Base directory to save comparison plots
    """
    # Ensure comparison directory exists
    comparison_dir = os.path.join(results_base_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract strategies and learning rates
    strategies = sorted(list(set([s for _, s in results_dict.keys()])))
    learning_rates = sorted(list(set([lr for lr, _ in results_dict.keys()])))
    
    # Print summary
    print("\nResults Summary:")
    print("="*80)
    print(f"{'Strategy':<15} {'Learning Rate':<15} {'Train Reward':<15} {'Train Profit':<15} {'Test Reward':<15} {'Test Profit':<15}")
    print("-"*80)
    for (lr, strategy), results in sorted(results_dict.items(), key=lambda x: (x[0][1], x[0][0])):
        train_reward = results.get('train_reward', 0)
        train_profit = results.get('train_profit', 0)
        test_reward = results.get('test_reward', 0)
        test_profit = results.get('test_profit', 0)
        print(f"{strategy:<15} {lr:<15.4f} {train_reward:<15.2f} {train_profit:<15.4f} {test_reward:<15.2f} {test_profit:<15.4f}")
    print("="*80)
    
    # Save the summary to a text file
    with open(os.path.join(comparison_dir, 'results_summary.txt'), 'w') as f:
        f.write("Results Summary:\n")
        f.write("="*80 + "\n")
        f.write(f"{'Strategy':<15} {'Learning Rate':<15} {'Train Reward':<15} {'Train Profit':<15} {'Test Reward':<15} {'Test Profit':<15}\n")
        f.write("-"*80 + "\n")
        for (lr, strategy), results in sorted(results_dict.items(), key=lambda x: (x[0][1], x[0][0])):
            train_reward = results.get('train_reward', 0)
            train_profit = results.get('train_profit', 0)
            test_reward = results.get('test_reward', 0)
            test_profit = results.get('test_profit', 0)
            f.write(f"{strategy:<15} {lr:<15.4f} {train_reward:<15.2f} {train_profit:<15.4f} {test_reward:<15.2f} {test_profit:<15.4f}\n")
        f.write("="*80 + "\n")
    
    # Create comparison plots
    # 1. Learning rate vs test profit by strategy
    plt.figure(figsize=(10, 6))
    for strategy in strategies:
        lr_values = []
        test_profits = []
        for (lr, strat), results in results_dict.items():
            if strat == strategy:
                lr_values.append(lr)
                test_profits.append(results.get('test_profit', 0))
        plt.plot(lr_values, test_profits, 'o-', label=strategy)
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Profit')
    plt.title('Test Profit vs Learning Rate by Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(comparison_dir, 'test_profit_vs_lr.png'))
    plt.close()
    
    # 2. Training curves for each strategy (best learning rate)
    plt.figure(figsize=(12, 8))
    for strategy in strategies:
        best_test_profit = -float('inf')
        best_results = None
        best_lr = None
        
        for (lr, strat), results in results_dict.items():
            if strat == strategy and results.get('test_profit', 0) > best_test_profit:
                best_test_profit = results.get('test_profit', 0)
                best_results = results
                best_lr = lr
        
        if best_results and 'smoothed_rewards' in best_results:
            plt.plot(best_results['smoothed_rewards'], label=f"{strategy} (lr={best_lr})")
    
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Training Curves for Best Learning Rate by Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(comparison_dir, 'training_curves_best_lr.png'))
    plt.close()
    
    # 3. Bar chart comparing train vs test profit for each strategy-lr combo
    n_configs = len(results_dict)
    indices = np.arange(n_configs)
    bar_width = 0.35
    
    train_profits = []
    test_profits = []
    config_labels = []
    
    for (lr, strategy), results in sorted(results_dict.items(), key=lambda x: (x[0][1], x[0][0])):
        train_profits.append(results.get('train_profit', 0))
        test_profits.append(results.get('test_profit', 0))
        config_labels.append(f"{strategy}\nlr={lr}")
    
    plt.figure(figsize=(max(10, n_configs), 6))
    plt.bar(indices - bar_width/2, train_profits, bar_width, label='Train Profit')
    plt.bar(indices + bar_width/2, test_profits, bar_width, label='Test Profit')
    
    plt.xlabel('Configuration')
    plt.ylabel('Profit')
    plt.title('Train vs Test Profit by Configuration')
    plt.xticks(indices, config_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(comparison_dir, 'train_vs_test_profit.png'))
    plt.close()
    
    # 4. Heatmap of test profits by learning rate and strategy
    if len(strategies) > 1 and len(learning_rates) > 1:
        profit_matrix = np.zeros((len(strategies), len(learning_rates)))
        
        for i, strategy in enumerate(strategies):
            for j, lr in enumerate(learning_rates):
                if (lr, strategy) in results_dict:
                    profit_matrix[i, j] = results_dict[(lr, strategy)].get('test_profit', 0)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(profit_matrix, interpolation='nearest', cmap='viridis')
        plt.colorbar(label='Test Profit')
        plt.xticks(np.arange(len(learning_rates)), [str(lr) for lr in learning_rates], rotation=45)
        plt.yticks(np.arange(len(strategies)), strategies)
        plt.xlabel('Learning Rate')
        plt.ylabel('Strategy')
        plt.title('Test Profit Heatmap')
        
        # Add text annotations in the heatmap
        for i in range(len(strategies)):
            for j in range(len(learning_rates)):
                text_color = 'black' if profit_matrix[i, j] < np.max(profit_matrix) * 0.7 else 'white'
                plt.text(j, i, f"{profit_matrix[i, j]:.4f}", 
                        ha="center", va="center", color=text_color)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'test_profit_heatmap.png'))
        plt.close()
    
    # 5. Compare final reward distributions
    plt.figure(figsize=(12, 6))
    for (lr, strategy), results in sorted(results_dict.items(), key=lambda x: (x[0][1], x[0][0])):
        if 'episode_rewards' in results:
            # Get the last N rewards for distribution
            last_n = min(100, len(results['episode_rewards']))
            final_rewards = results['episode_rewards'][-last_n:]
            plt.hist(final_rewards, alpha=0.5, label=f"{strategy}, lr={lr}", bins=20)
    
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'final_reward_distributions.png'))
    plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Q-Learning Tile Coding experiments for trading')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config_tile.json',
        help='Path to JSON configuration file'
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{config_path}' contains invalid JSON.")
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration from JSON file
    config = load_config(args.config)
    
    # Extract configuration parameters
    window_size = config.get('window_size', 10)
    num_tilings = config.get('num_tilings', 8)
    num_tiles = config.get('num_tiles', 8)
    memory_size = config.get('memory_size', 4096)
    train_frame_bound = tuple(config.get('train_frame_bound', (50, 200)))
    test_frame_bound = tuple(config.get('test_frame_bound', (200, 350)))
    results_base_dir = config.get('results_base_dir', 'results_tile')
    experiment_type = config.get('experiment_type', 'single_df')
    learning_rates = config.get('learning_rates', [0.1, 0.01, 0.001])
    exploration_strategies = config.get('exploration_strategies', ['epsilon-greedy', 'thompson'])
    num_episodes = config.get('num_episodes', 10000)
    
    # Optional discretizer parameters
    price_range = config.get('price_range')
    diff_range = config.get('diff_range')
    
    # Load dataframes if specified
    custom_df = None
    
    if experiment_type == 'single_df':
        df_path = config.get('df_path')
        if df_path:
            print(f"Loading custom dataframe from {df_path}")
            custom_df = load_custom_data(df_path)
    
    # Create results directory if it doesn't exist
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Save the used configuration
    config_out_path = os.path.join(results_base_dir, 'used_config.json')
    with open(config_out_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Starting experiments with configuration from {args.config}")
    print(f"Results will be saved to {results_base_dir}")
    print(f"Using tile coding with {num_tilings} tilings, {num_tiles} tiles per dimension, and memory size {memory_size}")
    
    # Run experiments with different learning rates and exploration strategies
    results = {}
    
    for strategy in exploration_strategies:
        for lr in learning_rates:
            print(f"\nRunning experiment with {strategy}, learning rate = {lr}")
            
            # Use same dataframe for training and testing
            results[(lr, strategy)] = run_experiment(
                learning_rate=lr,
                window_size=window_size,
                num_tilings=num_tilings,
                num_tiles=num_tiles,
                train_frame_bound=train_frame_bound,
                test_frame_bound=test_frame_bound,
                num_episodes=num_episodes,
                exploration_strategy=strategy,
                df=custom_df,
                price_range=price_range,
                diff_range=diff_range,
                memory_size=memory_size,
                results_base_dir=results_base_dir
            )
    
    # Compare results
    if results:
        compare_results(results, results_base_dir) 