import numpy as np
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import seaborn as sns

# Log in to W&B
wandb.login()

# Project name for W&B
project = f"gridworld_dqn_1"

# ============ Grid World setting ===========
ROWS, COLS = 3, 4      # number of rows and columns
WIN_STATE = (0,3)      # goal state coordinates
LOSE_STATE = (1,3)     # lose state coordinates
START = (2,0)          # start state coordinates
WALL = (1,1)           # wall state coordinates

NUM_EPISODES = 1500     # number of training episodes
EVAL_EPISODES = 500     # number of evaluation episodes

GOAL_REWARD = 1        # Reward for reachign goal
LOSE_REWARD = -1       # Penalty for reaching lose

# ============= Hyperparameters =============
LEARNING_RATE = 3e-1   # Learning Rate
DISCOUNT_FACTOR = 0.99 # Discount Factor
EPSILON_DECAY = 0.995  # Epsilon decay factor
EPSILON_RATE = 0.9     # Epsilon Rate
BATCH_SIZE = 128        # Mini-Batch size for replay memory


class State:
    """
    Represnets the environment state in the grid world
    Handles position, rewards, termination, and movement
    """
    def __init__(self, state=START):
        self.grid = np.zeros([ROWS, COLS])  # Initialize grid
        self.state = state                  # current agent position
        self.isEnd = False                  # flag for end of episode

    def reward(self):
        """
        Define reward

        - WIN_STATE: +1
        - LOSE_STATE: -1
        - Other state: -0.04
        """
        if self.state == WIN_STATE:
            return GOAL_REWARD
        elif self.state == LOSE_STATE:
            return LOSE_REWARD
        else:
            return -0.04
    
    def isEndFunc(self):
        """
        Check if the episode ended
        When the stat hits WIN and LOSE state, episode ends
        """
        if (self.state == WIN_STATE):
            self.isEnd = True

        if (self.state == LOSE_STATE):
            self.isEnd = True

    def move(self, action):

        """
        Define movement with stochastic outcomes
        - 80% desired direction
        - 10% left
        - 10% right
        
        Agent stay in the same cell when it hits walls/boundaries
        """
        
        # deifine probabilites
        probabilities = [0.8, 0.1, 0.1]

        # sample action
        if action == "up":
            action = np.random.choice(["up", "left", "right"], p=probabilities)
        elif action == "down":
            action = np.random.choice(["down", "left", "right"], p=probabilities)
        elif action == "left":
            action = np.random.choice(["left", "up", "down"], p=probabilities)
        elif action == "right":
            action = np.random.choice(["right", "up", "down"], p=probabilities)

        # compute the new position
        i, j = self.state
        if action == "up":
            i -= 1
        elif action == "down":
            i += 1
        elif action == "left":
            j -= 1
        elif action == "right":
            j += 1

        # check boundaries and wall
        if 0 <= i < ROWS:
            if 0 <= j < COLS:
                if (i, j) != WALL:
                    return (i,j)

        return self.state    # stay in the same position


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) with 2 hidden layers and ReLU activations

    - First hidden layer: input_dim -> 64 neurons, followed by ReLU activations
    - Second hidden layer: 64 -> 64 nerons, followed by ReLU activation
    - Output layer: 64 -> output_dim neurons (Q-values for each action)

    Args:
        input_dim (int): Dimension of input features (state space)
        output_dim (int): Number of actions (action space)
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)
        

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Agent:
    """
    DQN Agent
    """
    def __init__(self):
        self.actions = ["up", "down", "left", "right"]  # actions
        self.State = State()                            # initialize environment state
        self.lr = LEARNING_RATE                         # learning rate
        self.exp_rate = EPSILON_RATE                    # exploration rate
        self.decay_gamma = DISCOUNT_FACTOR              # discount factor
        self.exp_decay = EPSILON_DECAY                  # epsilon decay per episode
        self.min_exp_rate = 0.01                        # miminum epsilon

        self.memory = deque(maxlen=1000)               # replay buffer size
        self.batch_size = BATCH_SIZE                    # batch size
        self.target_update_freq = 10                   # target network update frequency

        # Neural network initializaiton
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = ROWS * COLS
        output_dim = len(self.actions)
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()                                           # Huber loss for better handling outliers
        
        self.episode_rewards = []
        self.episode_steps = []
        self.losses = []
        self.episode_successes = []

    def state_to_tensor(self, state):
        """
        Convert the state (i, j) into a one-hot encoded tensor
        """
        # Unpack the 2D state coordinates (row, column)
        i, j = state

        # Convert the 2D state position into a single flat index
        state_flat = i * COLS + j

        # Initialize a zero vector of length ROWS * COLS
        state_one_hot = np.zeros(ROWS * COLS)

        # Set the element corresponding to the current position to 1
        state_one_hot[state_flat] = 1

        # Convert the numpy array to a pytorch tensor and move it to the device (CPU/GPU)
        return torch.FloatTensor(state_one_hot).to(self.device)

    def chooseAction(self, epsilon=None):
        """
        Chooses an action using an ε-greedy policy.
        With probability ε: random action (exploration)
        Otherwise: action with the highest Q-value (exploitation)
        """
        if self.State.isEnd:
            # If the agent hit terminate, return None
            return None

        # If epsilon value is defined, use it otherwise use exp_rate for epsilon
        current_epsilon = epsilon if epsilon is not None else self.exp_rate

        # Choose an action using the epsilon-greedy strategy
        if np.random.uniform(0, 1) < current_epsilon:
            # with probability epsilon, choose a random action (exploration)
            return np.random.choice(self.actions)
        else:
            # Otherwise, choose the best action to teh Q-network (exploit)
            state_tensor = self.state_to_tensor(self.State.state)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return self.actions[torch.argmax(q_values).item()]

    def takeAction(self, action):
        """
        Execute the given action and return the new state
        """
        if action is None:
            # If no action is given, return the current state
            return self.State
        # Compute the new position after taking the action
        new_position = self.State.move(action)
        # Update the internal state with the new posiion
        self.State.state = new_position
        # Check whether the new state is a terminal state
        self.State.isEndFunc()
        # Return the updated state
        return self.State
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store one transition (state, action, reward, next_state, done) in replay memory
        Each trasnsition represents one step in the environment
        """
        # Convert current and next states into one-hot encoded tensors
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)
        # Convert actin into index
        action_idx = self.actions.index(action)
        # Append the experience tuple to memory
        self.memory.append((state_tensor, action_idx, reward, next_state_tensor, done))

    def learn(self):
        """
        Perform one learning step from replay memory
        Samples a batch and updates the Q-network using the Bellman equaiton
        """
        # If not enough experiences in memory, skip learning
        if len(self.memory) < self.batch_size:
            return None
        
        # Randomly sample a batch of transitions (state, action, reward, next_state, done)
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)   # Unpack

        # Convert lists of tensors to a single batched tensor
        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values for current states using the policy network
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0].detach()

        # Apply the Bellman equation
        target_q_values = rewards + (1 - dones) * self.decay_gamma * next_q_values

        # Compute the loss between predicted Q-values and target Q-values
        loss = self.loss_fn(q_values, target_q_values)

        # Perform backpropagation to update the policy network
        self.optimizer.zero_grad()       # Reset previous gradients
        loss.backward()                  # Compute new gradients
        self.optimizer.step()            # Update network parameters

        # Return the scalar loss value of logging
        return loss.item()

    def reset(self):
        """
        Reset environment for new episode
        """
        self.State = State()

        
    def train(self, num_episodes=NUM_EPISODES):
        """
        Train the DQN agent
        """

        # Initialize a step counter
        step_count = 0

        for i in range(num_episodes):
            

            # Reset the env to the initial state
            self.reset()

            # Initialize
            episode_reward = 0
            steps = 0

            while not self.State.isEnd:
                # Current state and chosen action
                s = self.State.state
                a = self.chooseAction()

                # take the chosen action, get next state and reward
                self.State = self.takeAction(a)
                s_next = self.State.state
                r = self.State.reward()

                # Terminal flag
                done = 1 if self.State.isEnd else 0

                episode_reward += r
                steps += 1
                step_count += 1

                # Store the transition in replay memory
                self.store_transition(s, a, r, s_next, done)

                # Perform one learning step
                loss = self.learn()

                if loss is not None:
                    self.losses.append(loss)

                # Update target network periodically
                if step_count % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            success = 1 if self.State.state == WIN_STATE else 0

            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)
            self.episode_successes.append(success)

            self.exp_rate = max(self.min_exp_rate, self.exp_rate * self.exp_decay)

            # Log to wandb
            if loss is not None:
                wandb.log({
                "episode": i,
                "reward": episode_reward,
                "steps": steps,
                "epsilon": self.exp_rate,
                "loss": loss,
                "train success": success,
                "avg_train_reward": np.mean(self.episode_rewards),
                "avg_train_success": np.mean(self.episode_successes)
            })
            
            else:
                wandb.log({
                "episode": i,
                "reward": episode_reward,
                "steps": steps,
                "epsilon": self.exp_rate,
                "train success": success,
                "avg_train_reward": np.mean(self.episode_rewards)
            })


            if i % 100 == 0:
                print(f"Episode {i}, ε: {self.exp_rate:.3f}, Reward: {episode_reward:.3f}, Step: {steps}")
           
    def evaluate(self, num_eval_episodes=EVAL_EPISODES):
        """
        Evaluates the trained policy
        """
        # Initialize
        eval_rewards = []
        eval_steps = []
        eval_successes = []

        for episode in range(num_eval_episodes):
            # Reset the env to the initial state
            self.reset()

            # Initialize reward and steps
            episode_reward = 0
            steps = 0

            while not self.State.isEnd:

                # select an action with epsilon = 0
                a = self.chooseAction(epsilon=0.0)

                # Take the selected action and transition to the next state
                self.State = self.takeAction(a)

                # Get the reward from the current state
                r = self.State.reward()

                episode_reward += r
                steps += 1
            
            success = 1 if self.State.state == WIN_STATE else 0

            # Append
            eval_rewards.append(episode_reward)
            eval_steps.append(steps)
            eval_successes.append(success)

            # Log each episode separately
            print(f"Evaluation Episode {episode+1}: Reward = {episode_reward}, Steps = {steps}")

            wandb.log({
                "eval_episode": episode + 1,
                "eval_reward": episode_reward,
                "eval_steps": steps,
                "eval_success": success,
                "avg_eval_reward": np.mean(eval_rewards)
            })

        return eval_rewards, eval_steps


def plot_policy(agent):
    """
    Visualize the learned policy
    """
    
    # Initialize a dictonary to store
    policy={}

    for i in range(ROWS):
        for j in range(COLS):
            if (i,j) in [WALL, WIN_STATE, LOSE_STATE]:
                continue
            
            state_tensor = agent.state_to_tensor((i, j))
            with torch.no_grad():
                qvals = agent.policy_net(state_tensor).cpu().numpy()
            policy[(i, j)] = agent.actions[np.argmax(qvals)]

    # Set arrow dictionaly
    arrow_dic = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→"
    }

    fig, ax = plt.subplots(figsize=(8,6))    # Create a Figure and Axes for plottig
    ax.set_xlim(0, COLS)                     # Set the x-axis limits from 0  to number of columns
    ax.set_ylim(0, ROWS)                     # Set the y-axis limits from 0 to number of rows
    ax.set_xticks([])                        # Remove x-axis ticks
    ax.set_yticks([])                        # Remoce y-axis ticks
    ax.set_aspect('equal')                   # Set each cell square

    # Draw grid lines
    for i in range(ROWS):
        for j in range(COLS):
            rect = plt.Rectangle((j, ROWS-i-1), 1, 1, fill=False, edgecolor='black', linewidth=1)
            ax.add_patch(rect)

    # Add Q-Value and arrow in each cell
    for i in range(ROWS):
        for j in range(COLS):
            # Convert python coordinates to matplotlib coordinates
            y = ROWS - i - 0.5 
            x = j + 0.5

            # Skip WALL, WIN, and LOSE cells
            if (i,j) == WALL:
                continue

            if (i,j) == WIN_STATE:
                continue

            if (i,j) == LOSE_STATE:
                continue
                
            state_tensor = agent.state_to_tensor((i, j))
            with torch.no_grad():
                qvals = agent.policy_net(state_tensor).cpu().numpy()
            action = policy[(i, j)]
            arrow = arrow_dic[action]

            # Display Q-Values in four directions
            ax.text(x, y + 0.3, f"{qvals[agent.actions.index('up')]:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')
            ax.text(x, y - 0.3, f"{qvals[agent.actions.index('down')]:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')
            ax.text(x - 0.3, y, f"{qvals[agent.actions.index('left')]:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')
            ax.text(x + 0.3, y, f"{qvals[agent.actions.index('right')]:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')

            # Display arrow in the center of cell
            ax.text(x, y, arrow, horizontalalignment='center', verticalalignment='center', fontsize=16, color='black')

    # Text for goal
    gx = WIN_STATE[1] + 0.5
    gy = ROWS - WIN_STATE[0] - 0.5
    ax.text(gx, gy, "G", ha='center', va='center', fontsize=16, color="green")

    # Text for lose
    lx = LOSE_STATE[1] + 0.5
    ly = ROWS - LOSE_STATE[0] - 0.5
    ax.text(lx, ly, "L", ha='center', va='center', fontsize=16, color="red")

    # Text for wall
    wx = WALL[1] + 0.5
    wy = ROWS - WALL[0] - 0.5
    ax.text(wx, wy, "Wall", ha='center', va='center', fontsize=16, color="black")

    return fig
    
def main():
    """
    Main training and evaluation
    """
    # Hyperparameters
    config = {
        "learning_rate": LEARNING_RATE,
        "gamma": DISCOUNT_FACTOR,
        "epsilon_decay": EPSILON_DECAY,
        "epsilon_rate": EPSILON_RATE,
        "episodes": NUM_EPISODES,
        "lose_reward": LOSE_REWARD
    }

    # Initialize W&B
    wandb.init(project=project, config=config)
    config = wandb.config

    run_name = f"learning_rate_{config.learning_rate}_gamma_{config.gamma}_epsilon_decay_{config.epsilon_decay}_epsilon_rate_{config.epsilon_rate}"
    
    wandb.run.name = run_name

    # Create Q-Learning agent and set hyperparameters from the configration
    agent = Agent()
    agent.lr = config.learning_rate
    agent.decay_gamma = config.gamma
    agent.exp_decay = config.epsilon_decay
    agent.exp_rate = config.epsilon_rate

    # Train
    agent.train(config.episodes)

    # # Compute the final average reward over last 200 episodes
    # Using last 100 rewards becasue the agent has already learned most of its policy by then
    avg_train_reward = np.mean(agent.episode_rewards[-200])
    wandb.log({f"avg_train_reward_200": avg_reward})

    # Evalate
    agent.evaluate()

    # Generate a policy map
    fig = plot_policy(agent)
    # Log policy map image to W&B
    wandb.log({f"policy_map": wandb.Image(fig)})
    plt.close(fig)

    # Store the policy
    policy = {}

    for i in range(ROWS):
        for j in range(COLS):
            state = (i,j)

            # Skip wall, win, and lose state
            if state in [WALL, WIN_STATE, LOSE_STATE]:
                continue

            state_tensor = agent.state_to_tensor(state)
            with torch.no_grad():
                qvals = agent.policy_net(state_tensor)

            best_action = agent.actions[torch.argmax(qvals).item()]

            # Add it to the policy dictionary
            policy[state] = best_action

    wandb.finish()


if __name__ == "__main__":
    main()
