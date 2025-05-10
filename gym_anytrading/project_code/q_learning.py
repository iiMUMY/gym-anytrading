import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import pandas as pd
import json
import argparse
import sys

import gym_anytrading
from gym_anytrading.envs import StocksEnv, Actions, Positions 
from gym_anytrading.datasets import STOCKS_GOOGL


class Discretizer:
    def __init__(self, lower_bounds, upper_bounds, num_bins):
        """
        Initialize the discretizer for continuous state space
        
        Args:
            lower_bounds: Lower bounds for each dimension of the observation space
            upper_bounds: Upper bounds for each dimension of the observation space
            num_bins: Number of bins for each dimension
        """
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.num_bins = np.array(num_bins)
        self.bin_sizes = (upper_bounds - lower_bounds) / num_bins
    
    def discretize(self, observation):
        """
        Convert a continuous observation to a discrete state
        
        Args:
            observation: Continuous observation from the environment
            
        Returns:
            A tuple representing the discretized state
        """
        # Handle different observation types
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # For trading environment, we'll use the last row (most recent data)
        if len(observation.shape) > 1:
            flat_obs = observation[-1]
        else:
            flat_obs = observation
        
        # Ensure the flattened observation matches the expected dimensions
        if len(flat_obs) != len(self.lower_bounds):
            if len(flat_obs) > len(self.lower_bounds):
                flat_obs = flat_obs[:len(self.lower_bounds)]
            else:
                flat_obs = np.pad(flat_obs, (0, len(self.lower_bounds) - len(flat_obs)), 'constant')
        
        # Clip the observation to be within bounds
        clipped_obs = np.clip(flat_obs, self.lower_bounds, self.upper_bounds)
        
        # Calculate the bin indices for each dimension
        bin_indices = np.floor((clipped_obs - self.lower_bounds) / self.bin_sizes).astype(int)
        
        # Ensure indices are within valid range
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        # Convert multi-dimensional indices to a single index
        return tuple(map(int, bin_indices.flatten()))

class QLearningAgent:
    def __init__(self, action_space, discretizer, learning_rate=0.1, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.02):
        """
        Initialize the Q-Learning agent
        
        Args:
            action_space: The action space of the environment
            discretizer: The discretizer to convert continuous observations to discrete states
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Rate at which exploration rate decays
            min_exploration_rate: Minimum exploration rate
        """
        self.action_space = action_space
        self.discretizer = discretizer
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = {}
    
    def get_action(self, observation):
        """Choose an action using epsilon-greedy policy"""
        state = self.discretizer.discretize(observation)
        
        if np.random.random() < self.exploration_rate:
            return self.action_space.sample()
        else:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.action_space.n)
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Update the Q-table using the Q-learning update rule"""
        discrete_state = self.discretizer.discretize(state)
        discrete_next_state = self.discretizer.discretize(next_state)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space.n)
        
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_space.n)
        
        best_next_action = np.argmax(self.q_table[discrete_next_state])
        td_target = reward + (1 - int(done)) * self.discount_factor * self.q_table[discrete_next_state][best_next_action]
        td_error = td_target - self.q_table[discrete_state][action]
        self.q_table[discrete_state][action] += self.learning_rate * td_error
        
        if done:
            self.exploration_rate = max(self.min_exploration_rate, 
                                      self.exploration_rate * self.exploration_decay)

class QLearningAgentThompson:
    def __init__(self, action_space, discretizer, learning_rate=0.1, discount_factor=0.99, alpha_prior=1.0, beta_prior=1.0):
        """
        Initialize the Q-Learning agent with Thompson Sampling exploration
        
        Args:
            action_space: The action space of the environment
            discretizer: The discretizer to convert continuous observations to discrete states
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            alpha_prior: Alpha parameter for the Beta distribution prior
            beta_prior: Beta parameter for the Beta distribution prior
        """
        self.action_space = action_space
        self.discretizer = discretizer
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Initialize Q-table with zeros
        self.q_table = {}
        
        # Initialize success and failure counts for each state-action pair
        self.success_count = {}
        self.failure_count = {}
        
        # Normalize rewards to [0, 1] for Beta distribution
        self.max_reward_seen = 1.0
        self.min_reward_seen = 0.0
    
    def get_action(self, observation):
        """Choose an action using Thompson Sampling policy"""
        state = self.discretizer.discretize(observation)
        
        # Initialize state if not in tables
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
            self.success_count[state] = np.ones(self.action_space.n) * self.alpha_prior
            self.failure_count[state] = np.ones(self.action_space.n) * self.beta_prior
        
        # Sample from Beta distribution for each action
        samples = np.zeros(self.action_space.n)
        for a in range(self.action_space.n):
            samples[a] = np.random.beta(
                self.success_count[state][a],
                self.failure_count[state][a]
            )
        
        return np.argmax(samples)
    
    def update(self, state, action, reward, next_state, done):
        """Update the Q-table and Beta parameters"""
        discrete_state = self.discretizer.discretize(state)
        discrete_next_state = self.discretizer.discretize(next_state)
        
        # Initialize Q-values if not in table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space.n)
            self.success_count[discrete_state] = np.ones(self.action_space.n) * self.alpha_prior
            self.failure_count[discrete_state] = np.ones(self.action_space.n) * self.beta_prior
        
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_space.n)
            self.success_count[discrete_next_state] = np.ones(self.action_space.n) * self.alpha_prior
            self.failure_count[discrete_next_state] = np.ones(self.action_space.n) * self.beta_prior
        
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[discrete_next_state])
        td_target = reward + (1 - int(done)) * self.discount_factor * self.q_table[discrete_next_state][best_next_action]
        td_error = td_target - self.q_table[discrete_state][action]
        self.q_table[discrete_state][action] += self.learning_rate * td_error
        
        # Update success and failure counts based on reward
        self.max_reward_seen = max(self.max_reward_seen, reward)
        self.min_reward_seen = min(self.min_reward_seen, reward)
        
        # Normalize reward to [0, 1] range
        reward_range = max(1e-5, self.max_reward_seen - self.min_reward_seen)
        normalized_reward = (reward - self.min_reward_seen) / reward_range
        
        # Update Beta distribution parameters
        if normalized_reward > 0.5:
            self.success_count[discrete_state][action] += normalized_reward
        else:
            self.failure_count[discrete_state][action] += (1 - normalized_reward)

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

def create_discretizer(window_size=10, price_range=(190, 320), diff_range=(-20, 20), bins=(20, 20)):
    """
    Create and return a configured discretizer
    
    Args:
        window_size: Size of the observation window
        price_range: Tuple of (min_price, max_price)
        diff_range: Tuple of (min_diff, max_diff)
        bins: Tuple of (price_bins, diff_bins)
    """
    price_lower, price_upper = price_range
    diff_lower, diff_upper = diff_range
    price_bins, diff_bins = bins
    
    lower_bounds = np.array([[price_lower, diff_lower]] * window_size).flatten()
    upper_bounds = np.array([[price_upper, diff_upper]] * window_size).flatten()
    num_bins = np.array([[price_bins, diff_bins]] * window_size).flatten()
    
    return Discretizer(lower_bounds, upper_bounds, num_bins)

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
        state = agent.discretizer.discretize(observation)
        if state in agent.q_table:
            action = np.argmax(agent.q_table[state])
        else:
            action = test_env.action_space.sample()
        
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
    plt.savefig(os.path.join(results_dir, f'q_learning_rewards_lr{learning_rate}.png'))
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

def run_experiment(learning_rate, 
                  window_size=10,
                  train_frame_bound=(50, 200),
                  test_frame_bound=(200, 350),
                  num_episodes=10000, 
                  exploration_strategy='epsilon-greedy',
                  df=None,
                  price_range=None,
                  diff_range=None,
                  bins=None,
                  results_base_dir='results',
                  **kwargs):
    """
    Run a complete experiment with the specified learning rate and exploration strategy
    
    Args:
        learning_rate: Learning rate for Q-learning
        window_size: Size of the observation window (same for both environments)
        train_frame_bound: Tuple of (start, end) indices for training data
        test_frame_bound: Tuple of (start, end) indices for testing data
        num_episodes: Number of episodes to train
        exploration_strategy: Either 'epsilon-greedy' or 'thompson'
        df: Optional pandas DataFrame with custom data
        price_range: Optional tuple of (min_price, max_price) for discretization
        diff_range: Optional tuple of (min_diff, max_diff) for discretization
        bins: Optional tuple of (price_bins, diff_bins) for discretization
        results_base_dir: Base directory to save results
        **kwargs: Additional arguments for the agent initialization
    """
    # Create training and testing environments with the same window size
    train_env = create_environment(window_size=window_size, frame_bound=train_frame_bound, df=df)
    test_env = create_environment(window_size=window_size, frame_bound=test_frame_bound, df=df)
    
    # Determine price and diff ranges based on the dataframe
    if price_range is None or diff_range is None or bins is None:
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

    if bins is None:
        bins = (20, 20)
    
    # Create discretizer with appropriate ranges for the data
    discretizer = create_discretizer(window_size=window_size, 
                                     price_range=price_range,
                                     diff_range=diff_range,
                                     bins=bins)
    
    # Create agent based on exploration strategy
    if exploration_strategy == 'epsilon-greedy':
        agent = QLearningAgent(
            action_space=train_env.action_space,
            discretizer=discretizer,
            learning_rate=learning_rate,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.995,
            min_exploration_rate=0.02,
            **kwargs
        )
    elif exploration_strategy == 'thompson':
        agent = QLearningAgentThompson(
            action_space=train_env.action_space,
            discretizer=discretizer,
            learning_rate=learning_rate,
            discount_factor=0.99,
            alpha_prior=1.0,
            beta_prior=1.0,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown exploration strategy: {exploration_strategy}")
    
    # Train agent
    episode_rewards, episode_profits, smoothed_rewards = train_agent(train_env, agent, num_episodes)
    
    # Plot training results
    results_dir = os.path.join(results_base_dir, exploration_strategy, f'lr{learning_rate}')
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
    
    # Plot smoothed rewards separately
    plt.figure(figsize=(12, 5))
    plt.plot(smoothed_rewards, linewidth=2)
    plt.title(f'Smoothed Reward vs Episode (Learning Rate = {learning_rate})')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward (window=50)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'smoothed_rewards_lr{learning_rate}.png'))
    plt.close()
    
    return {
        'learning_rate': learning_rate,
        'exploration_strategy': exploration_strategy,
        'episode_rewards': episode_rewards,
        'episode_profits': episode_profits,
        'smoothed_rewards': smoothed_rewards,
        'final_avg_reward': np.mean(episode_rewards[-100:]),
        'final_avg_profit': np.mean(episode_profits[-100:]),
        'train_reward': train_reward,
        'train_profit': train_profit,
        'test_reward': test_reward,
        'test_profit': test_profit
    }

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
    
    # Create a color map for different strategies
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    strategy_colors = {strategy: colors[i] for i, strategy in enumerate(strategies)}
    
    # Function to get smoothed rewards
    def get_smoothed_rewards(rewards, window_size=None):
        if window_size is None:
            window_size = min(100, len(rewards) // 10)
        if window_size < 2:
            window_size = 2
        return np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    # 1. Plot learning curves (rewards) by strategy
    plt.figure(figsize=(12, 8))
    for strategy in strategies:
        plt.subplot(len(strategies), 1, strategies.index(strategy) + 1)
        for lr in learning_rates:
            if (lr, strategy) in results_dict:
                rewards = results_dict[(lr, strategy)]['episode_rewards']
                # Plot raw rewards with low alpha
                plt.plot(rewards, alpha=0.1, color=strategy_colors[strategy])
                # Plot smoothed rewards
                smoothed = get_smoothed_rewards(rewards)
                window_size = min(100, len(rewards) // 10)
                plt.plot(range(window_size-1, len(rewards)), smoothed, 
                        label=f'LR = {lr}')
        plt.title(f'Learning Curves: {strategy}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'learning_curves_by_strategy.png'))
    plt.close()
    
    # 2. Plot learning curves (rewards) by learning rate
    plt.figure(figsize=(12, 8))
    for i, lr in enumerate(learning_rates):
        plt.subplot(len(learning_rates), 1, i + 1)
        for strategy in strategies:
            if (lr, strategy) in results_dict:
                rewards = results_dict[(lr, strategy)]['episode_rewards']
                # Plot raw rewards with low alpha
                plt.plot(rewards, alpha=0.1, color=strategy_colors[strategy])
                # Plot smoothed rewards
                smoothed = get_smoothed_rewards(rewards)
                window_size = min(100, len(rewards) // 10)
                plt.plot(range(window_size-1, len(rewards)), smoothed, 
                        label=strategy, color=strategy_colors[strategy])
        plt.title(f'Learning Curves: LR = {lr}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'learning_curves_by_lr.png'))
    plt.close()
    
    # 3. Plot final training performance comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(learning_rates))
    width = 0.35 / len(strategies)
    
    for i, strategy in enumerate(strategies):
        train_profits = [results_dict.get((lr, strategy), {}).get('train_profit', 0) 
                      for lr in learning_rates]
        plt.bar(x + (i - len(strategies)/2 + 0.5) * width, train_profits, width, 
                label=strategy, color=strategy_colors[strategy])
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Training Profit')
    plt.title('Training Performance Comparison')
    plt.xticks(x, [str(lr) for lr in learning_rates])
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(comparison_dir, 'training_performance.png'))
    plt.close()
    
    # 4. Plot testing performance comparison
    plt.figure(figsize=(10, 6))
    for i, strategy in enumerate(strategies):
        test_profits = [results_dict.get((lr, strategy), {}).get('test_profit', 0) 
                     for lr in learning_rates]
        plt.bar(x + (i - len(strategies)/2 + 0.5) * width, test_profits, width, 
                label=strategy, color=strategy_colors[strategy])
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Testing Profit')
    plt.title('Testing Performance Comparison')
    plt.xticks(x, [str(lr) for lr in learning_rates])
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(comparison_dir, 'testing_performance.png'))
    plt.close()
    
    # 5. Create a heatmap for train-test performance comparison
    plt.figure(figsize=(12, 5))
    
    # Training performance heatmap
    plt.subplot(1, 2, 1)
    train_data = np.zeros((len(strategies), len(learning_rates)))
    for i, strategy in enumerate(strategies):
        for j, lr in enumerate(learning_rates):
            if (lr, strategy) in results_dict:
                train_data[i, j] = results_dict[(lr, strategy)].get('train_profit', 0)
                
    plt.imshow(train_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Profit')
    plt.xticks(range(len(learning_rates)), [str(lr) for lr in learning_rates])
    plt.yticks(range(len(strategies)), strategies)
    plt.xlabel('Learning Rate')
    plt.ylabel('Strategy')
    plt.title('Training Performance')
    
    # Testing performance heatmap
    plt.subplot(1, 2, 2)
    test_data = np.zeros((len(strategies), len(learning_rates)))
    for i, strategy in enumerate(strategies):
        for j, lr in enumerate(learning_rates):
            if (lr, strategy) in results_dict:
                test_data[i, j] = results_dict[(lr, strategy)].get('test_profit', 0)
                
    im = plt.imshow(test_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Profit')
    plt.xticks(range(len(learning_rates)), [str(lr) for lr in learning_rates])
    plt.yticks(range(len(strategies)), strategies)
    plt.xlabel('Learning Rate')
    plt.ylabel('Strategy')
    plt.title('Testing Performance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'performance_heatmaps.png'))
    plt.close()
    
    # 6. Plot generalization capability (testing vs training performance)
    plt.figure(figsize=(10, 8))
    
    for strategy in strategies:
        train_profits = []
        test_profits = []
        for lr in learning_rates:
            if (lr, strategy) in results_dict:
                train_profits.append(results_dict[(lr, strategy)].get('train_profit', 0))
                test_profits.append(results_dict[(lr, strategy)].get('test_profit', 0))
        
        plt.scatter(train_profits, test_profits, label=strategy, color=strategy_colors[strategy], s=100)
        
        # Add text labels for each point (learning rate)
        for i, lr in enumerate(learning_rates):
            if i < len(train_profits):
                plt.annotate(f'LR={lr}', 
                           xy=(train_profits[i], test_profits[i]),
                           xytext=(5, 5),
                           textcoords='offset points')
    
    # Add diagonal line (y=x) for reference
    min_val = min(min([p for p in train_profits if p]), min([p for p in test_profits if p]))
    max_val = max(max(train_profits), max(test_profits))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('Training Profit')
    plt.ylabel('Testing Profit')
    plt.title('Generalization: Testing vs Training Performance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'generalization.png'))
    plt.close()
    
    # 7. Plot smoothed reward convergence comparison
    plt.figure(figsize=(12, 8))
    for strategy in strategies:
        for lr in learning_rates:
            if (lr, strategy) in results_dict:
                rewards = results_dict[(lr, strategy)]['episode_rewards']
                smoothed = get_smoothed_rewards(rewards, window_size=min(500, len(rewards) // 5))
                window_size = min(500, len(rewards) // 5)
                plt.plot(range(window_size-1, len(rewards)), smoothed, 
                       label=f'{strategy}, LR={lr}', 
                       color=strategy_colors[strategy],
                       linestyle='-' if strategy == strategies[0] else '--')
    
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Reward Convergence Comparison')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'reward_convergence.png'))
    plt.close()
    
    # 8. Find the best performing configurations
    best_train_profit = -float('inf')
    best_test_profit = -float('inf')
    best_train_config = None
    best_test_config = None
    best_overall_config = None
    best_overall_score = -float('inf')
    
    for (lr, strategy), results in results_dict.items():
        train_profit = results.get('train_profit', 0)
        test_profit = results.get('test_profit', 0)
        
        if train_profit > best_train_profit:
            best_train_profit = train_profit
            best_train_config = (lr, strategy)
            
        if test_profit > best_test_profit:
            best_test_profit = test_profit
            best_test_config = (lr, strategy)
            
        # For overall, consider a weighted average favoring test performance
        overall_score = 0.3 * train_profit + 0.7 * test_profit
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_overall_config = (lr, strategy)
    
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
    
    print("\nBest Configurations:")
    if best_train_config:
        print(f"Best Training Performance: Strategy={best_train_config[1]}, LR={best_train_config[0]}, Profit={best_train_profit:.4f}")
    if best_test_config:
        print(f"Best Testing Performance: Strategy={best_test_config[1]}, LR={best_test_config[0]}, Profit={best_test_profit:.4f}")
    if best_overall_config:
        print(f"Best Overall Performance: Strategy={best_overall_config[1]}, LR={best_overall_config[0]}")
        print(f"  Training Profit={results_dict[best_overall_config]['train_profit']:.4f}, Testing Profit={results_dict[best_overall_config]['test_profit']:.4f}")
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
        f.write("="*80 + "\n\n")
        
        f.write("Best Configurations:\n")
        if best_train_config:
            f.write(f"Best Training Performance: Strategy={best_train_config[1]}, LR={best_train_config[0]}, Profit={best_train_profit:.4f}\n")
        if best_test_config:
            f.write(f"Best Testing Performance: Strategy={best_test_config[1]}, LR={best_test_config[0]}, Profit={best_test_profit:.4f}\n")
        if best_overall_config:
            f.write(f"Best Overall Performance: Strategy={best_overall_config[1]}, LR={best_overall_config[0]}\n")
            f.write(f"  Training Profit={results_dict[best_overall_config]['train_profit']:.4f}, Testing Profit={results_dict[best_overall_config]['test_profit']:.4f}\n")
        f.write("="*80 + "\n")

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

def run_multi_df_experiment(learning_rate, 
                        window_size=10,
                        train_df=None, train_frame_bound=(50, 200),
                        test_df=None, test_frame_bound=(200, 350),
                        num_episodes=10000, 
                        exploration_strategy='epsilon-greedy',
                        price_range=None,
                        diff_range=None,
                        bins=None,
                        results_base_dir='results',
                        **kwargs):
    """
    Run an experiment with different dataframes for training and testing
    
    Args:
        learning_rate: Learning rate for Q-learning
        window_size: Size of the observation window
        train_df: DataFrame for training. If None, uses the default dataset
        train_frame_bound: Frame bounds for training
        test_df: DataFrame for testing. If None, uses the default dataset
        test_frame_bound: Frame bounds for testing
        num_episodes: Number of episodes to train
        exploration_strategy: Either 'epsilon-greedy' or 'thompson'
        price_range: Optional tuple of (min_price, max_price) for discretization
        diff_range: Optional tuple of (min_diff, max_diff) for discretization
        bins: Optional tuple of (price_bins, diff_bins) for discretization
        results_base_dir: Base directory to save results
        **kwargs: Additional arguments for the agent initialization
    
    Returns:
        Dictionary with experiment results
    """
    # Create training and testing environments with their respective dataframes
    train_env = create_environment(window_size=window_size, frame_bound=train_frame_bound, df=train_df)
    test_env = create_environment(window_size=window_size, frame_bound=test_frame_bound, df=test_df)
    
    # Determine price and diff ranges from training environment
    if price_range is None or diff_range is None:
        # Get min and max prices from the training environment's dataframe
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
    
    # Create discretizer with appropriate ranges for the training data
    discretizer = create_discretizer(window_size=window_size, 
                                     price_range=price_range,
                                     diff_range=diff_range,
                                     bins=bins)
    
    # Create agent based on exploration strategy
    if exploration_strategy == 'epsilon-greedy':
        agent = QLearningAgent(
            action_space=train_env.action_space,
            discretizer=discretizer,
            learning_rate=learning_rate,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.995,
            min_exploration_rate=0.02,
            **kwargs
        )
    elif exploration_strategy == 'thompson':
        agent = QLearningAgentThompson(
            action_space=train_env.action_space,
            discretizer=discretizer,
            learning_rate=learning_rate,
            discount_factor=0.99,
            alpha_prior=1.0,
            beta_prior=1.0,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown exploration strategy: {exploration_strategy}")
    
    # Train agent
    episode_rewards, episode_profits, smoothed_rewards = train_agent(train_env, agent, num_episodes)
    
    # Plot training results
    results_dir = os.path.join(results_base_dir, exploration_strategy, f'lr{learning_rate}', 'multi_df')
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
    
    # Plot smoothed rewards separately
    plt.figure(figsize=(12, 5))
    plt.plot(smoothed_rewards, linewidth=2)
    plt.title(f'Smoothed Reward vs Episode (Learning Rate = {learning_rate})')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward (window=50)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'smoothed_rewards_lr{learning_rate}.png'))
    plt.close()
    
    return {
        'learning_rate': learning_rate,
        'exploration_strategy': exploration_strategy,
        'episode_rewards': episode_rewards,
        'episode_profits': episode_profits,
        'smoothed_rewards': smoothed_rewards,
        'final_avg_reward': np.mean(episode_rewards[-100:]),
        'final_avg_profit': np.mean(episode_profits[-100:]),
        'train_reward': train_reward,
        'train_profit': train_profit,
        'test_reward': test_reward,
        'test_profit': test_profit
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Q-Learning experiments for trading')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.json',
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
    train_frame_bound = tuple(config.get('train_frame_bound', (50, 200)))
    test_frame_bound = tuple(config.get('test_frame_bound', (200, 350)))
    results_base_dir = config.get('results_base_dir', 'results')
    experiment_type = config.get('experiment_type', 'single_df')
    learning_rates = config.get('learning_rates', [0.1, 0.01, 0.001])
    exploration_strategies = config.get('exploration_strategies', ['epsilon-greedy', 'thompson'])
    num_episodes = config.get('num_episodes', 10000)
    
    # Optional discretizer parameters
    price_range = config.get('price_range')
    diff_range = config.get('diff_range')
    bins = tuple(config.get('bins', (20, 20))) if config.get('bins') else None
    
    # Load dataframes if specified
    custom_df = None
    train_df = None
    test_df = None
    
    if experiment_type == 'single_df':
        df_path = config.get('df_path')
        if df_path:
            print(f"Loading custom dataframe from {df_path}")
            custom_df = load_custom_data(df_path)
    else:  # multi_df
        train_df_path = config.get('train_df_path')
        test_df_path = config.get('test_df_path')
        
        if train_df_path:
            print(f"Loading training dataframe from {train_df_path}")
            train_df = load_custom_data(train_df_path)
        
        if test_df_path:
            print(f"Loading testing dataframe from {test_df_path}")
            test_df = load_custom_data(test_df_path)
    
    # Create results directory if it doesn't exist
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Save the used configuration
    config_out_path = os.path.join(results_base_dir, 'used_config.json')
    with open(config_out_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Starting experiments with configuration from {args.config}")
    print(f"Results will be saved to {results_base_dir}")
    
    # Run experiments with different learning rates and exploration strategies
    results = {}
    
    for strategy in exploration_strategies:
        for lr in learning_rates:
            print(f"\nRunning experiment with {strategy}, learning rate = {lr}")
            
            if experiment_type == "single_df":
                # Use same dataframe for training and testing
                results[(lr, strategy)] = run_experiment(
                    learning_rate=lr,
                    window_size=window_size,
                    train_frame_bound=train_frame_bound,
                    test_frame_bound=test_frame_bound,
                    num_episodes=num_episodes,
                    exploration_strategy=strategy,
                    df=custom_df,
                    price_range=price_range,
                    diff_range=diff_range,
                    bins=bins,
                    results_base_dir=results_base_dir
                )
            else:
                # Use different dataframes for training and testing
                results[(lr, strategy)] = run_multi_df_experiment(
                    learning_rate=lr,
                    window_size=window_size,
                    train_df=train_df, 
                    train_frame_bound=train_frame_bound,
                    test_df=test_df, 
                    test_frame_bound=test_frame_bound,
                    num_episodes=num_episodes,
                    exploration_strategy=strategy,
                    price_range=price_range,
                    diff_range=diff_range,
                    bins=bins,
                    results_base_dir=results_base_dir
                )
    
    # Compare results
    if results:
        compare_results(results, results_base_dir)