import numpy as np
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Log in to W&B
wandb.login()

# Project name for W&B
project = f"gridworld_dqn"

# Grid World setting
ROWS, COLS = 3, 4      # number of rows and columns
WIN_STATE = (0,3)      # goal state coordinates
LOSE_STATE = (1,3)     # lose state coordinates
START = (2,0)          # start state coordinates
WALL = (1,1)           # wall state coordinates

NUM_EPISODES = 500     # number of training episodes

GOAL_REWARD = 1        # Reward for reachign goal
LEARNING_RATE = 0.1    # Learning Rate
DISCOUNT_FACTOR = 0.7  # Discount Factor
EPSILON_DECAY = 0.999  # Epsilon decay factor
EPSILON_RATE = 0.1     # Epsilon Rate


class State:
    """
    Represnets the environment state in the grid world
    Handles position, rewards, termination, and movement
    """
    def __init__(self, state=START, lose_reward=-1):
        self.grid = np.zeros([ROWS, COLS])  # Initialize grid
        self.state = state                  # current agent position
        self.isEnd = False                  # flag for end of episode
        self.lose_reward = lose_reward      # losing reward

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
            return self.lose_reward
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
        self.layer1 = nn.Linear(input_dim, 64),
        self.layer2 = nn.Linear(64, 64),
        self.layer3 = nn.Linear(64, output_dim)
        

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Agent:
    """
    DQN agent
    """
    def __init__(self, lose_reward=-1):
        self.actions = ["up", "down", "left", "right"]  # actions
        self.State = State(lose_reward=lose_reward)     # initialize environment state
        self.lr = 0.1                                   # learning rate
        self.exp_rate = 0.1                             # exploration rate
        self.decay_gamma = 0.9                          # discount factor
        self.exp_decay = 0.995                          # epsilon decay per episode
        self.min_exp_rate = 0.01                        # miminum epsilon
        self.lose_reward = lose_reward                  # reward for losing

        self.memory = deque(maxlen=10000)               # replay buffer size
        self.batch_size = 64                            # batch size
        self.target_updata_freq = 100                   # target network update frequency

        # Neural network initializaiton
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = ROW * COLS
        output_dim = len(self.actions)
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.,loss_fn = nn.SmoothL1Loss()                                           # Huber loss for better handling outliers
        
        self.episode_rewards = []
        self.episode_steps = []
        self.losses = []

    def state_to_tensor(self, state):
        """
        Convert the environment state (i, j) into a one-hot encoded tensor.
        This allows the neural network to process discreate grid positions as input
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
        Choose actions using ε-greedy
        If epsilon value is defined, use it. Otherwise, use exp_rate for epsilon
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
        Store one experience (state, action, reward, next_state, done) in replay memory
        Each trasnsition represents one step in the environment
        """
        # Convert current and next states into one-hot encoded tenosors
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)
        # Convert actin into index
        action_idx = self.actions.index(action)
        # Append the experience tuple to memory
        self.memory.append((state_tenosor, action_idx, reward, next_state_tensor, done))

    def learn(self):
        # If not enough experiences in memory, skip learning
        if len(self.memory) < self.batch_size:
            return None
        
        # Randomly sample a batch of transitions (state, action, reward, next_state, done)
        batch = random.sample(self.memroy, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)   # Unpack

        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values for current states
        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.decay_gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def reset(self):
        """
        Reset environment for new episode
        """
        self.State = State(lose_reward=self.lose_reward)

        
    def train(self, NUM_EPISODES):
        """
        TODO: implement training
        """
        pass   

def plot_policy(agent):
    """
    TODO: Define agent.Q_values
    """
    # # Initialize a dictonary to store
    # policy={}

    # for i in range(ROWS):
    #     for j in range(COLS):
    #         if (i,j) in [WALL, WIN_STATE, LOSE_STATE]:
    #             continue

    #         # Select the action with highest Q-Value
    #         policy[(i,j)] = max(agent.Q_values[(i,j)], key=agent.Q_values[(i,j)].get)

    # # Set arrow dictionaly
    # arrow_dic = {
    #     "up": "↑",
    #     "down": "↓",
    #     "left": "←",
    #     "right": "→"
    # }

    # fig, ax = plt.subplots(figsize=(8,6))    # Create a Figure and Axes for plottig
    # ax.set_xlim(0, COLS)                     # Set the x-axis limits from 0  to number of columns
    # ax.set_ylim(0, ROWS)                     # Set the y-axis limits from 0 to number of rows
    # ax.set_xticks([])                        # Remove x-axis ticks
    # ax.set_yticks([])                        # Remoce y-axis ticks
    # ax.set_aspect('equal')                   # Set each cell square

    # # Draw grid lines
    # for i in range(ROWS):
    #     for j in range(COLS):
    #         rect = plt.Rectangle((j, ROWS-i-1), 1, 1, fill=False, edgecolor='black', linewidth=1)
    #         ax.add_patch(rect)

    # # Add Q-Value and arrow in each cell
    # for i in range(ROWS):
    #     for j in range(COLS):
    #         # Convert python coordinates to matplotlib coordinates
    #         y = ROWS - i - 0.5 
    #         x = j + 0.5

    #         # Skip WALL, WIN, and LOSE cells
    #         if (i,j) == WALL:
    #             continue

    #         if (i,j) == WIN_STATE:
    #             continue

    #         if (i,j) == LOSE_STATE:
    #             continue
                
    #         # Get Q-Value in (i, j)
    #         qvals = agent.Q_values[(i,j)]

    #         # Determine arrow based on policy 
    #         if  (i,j) in policy:
    #             action = policy[(i, j)]
    #             arrow = arrow_dic[action]
    #         else:
    #             arrow = " "

    #         # Display Q-Values in four directions
    #         ax.text(x, y + 0.3, f"{qvals['up']:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')
    #         ax.text(x, y - 0.3, f"{qvals['down']:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')
    #         ax.text(x - 0.3, y, f"{qvals['left']:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')
    #         ax.text(x + 0.3, y, f"{qvals['right']:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')

    #         # Display arrow in the center of cell
    #         ax.text(x, y, arrow, horizontalalignment='center', verticalalignment='center', fontsize=16, color='black')

    # # Text for goal
    # gx = WIN_STATE[1] + 0.5
    # gy = ROWS - WIN_STATE[0] - 0.5
    # ax.text(gx, gy, "G", ha='center', va='center', fontsize=16, color="green")

    # # Text for lose
    # lx = LOSE_STATE[1] + 0.5
    # ly = ROWS - LOSE_STATE[0] - 0.5
    # ax.text(lx, ly, "L", ha='center', va='center', fontsize=16, color="red")

    # # Text for wall
    # wx = WALL[1] + 0.5
    # wy = ROWS - WALL[0] - 0.5
    # ax.text(wx, wy, "Wall", ha='center', va='center', fontsize=16, color="black")

    # return fig
    
    pass

def main():

    # Hyperparameters
    config = {
        "learning_rate": LEARNING_RATE,
        "gamma": DISCOUNT_FACTOR,
        "epsilon_decay": EPSILON_DECAY,
        "epsilon_rate": EPSILON_RATE,
        "episodes": NUM_EPISODES
    }

    lose_rewards = [-1, -200]

    for lose_reward in lose_rewards:
      # Initialize W&B
      wandb.init(project=project, config={**config, "lose_reward": lose_reward})
      config = wandb.config
      run_name = f"learning_rate_{config.learning_rate}_gamma_{config.gamma}_epsilon_decay_{config.epsilon_decay}_epsilon_rate_{config.epsilon_rate}_lose_{config.lose_reward}"
      wandb.run.name = run_name

      # Create Q-Learning agent and set hyperparameters from the sweep configration
      agent = Agent(lose_reward=config.lose_reward)
      agent.learning_rate = config.learning_rate
      agent.gamma = config.gamma
      agent.epsilon_decay = config.epsilon_decay
      agent.epsilon_rate = config.epsilon_rate

    #   agent.train(NUM_EPISODES)

    #   # Generate a policy map
    #   fig = plot_policy(agent)
    #   # Log policy map image to W&B
    #   wandb.log({f"policy_map_L_{agent.lose_reward}": wandb.Image(fig)})
    #   plt.close(fig)

    #   # Store the policy
    #   policy = {}

    #   for i in range(ROWS):
    #       for j in range(COLS):
    #           state = (i,j)

    #           # Skip wall, win, and lose state
    #           if state in [WALL, WIN_STATE, LOSE_STATE]:
    #             continue

    #           with torch.no_grad():
    #             qvals = agent.
              
    #           best_action = 

    #           # Add it to the policy dictionary
    #           policy[state] = best_action

    #   # convert teh policy dictionary to a W&B table and log it
    #   policy_table_data = []
    #   for state, action in policy.items():
    #       policy_table_data.append([str(state), action])

    #   wandb.log({
    #       f"policy_table_L_{agent.lose_reward}": wandb.Table(data=policy_table_data, columns=["state", "action"]
    #       )
    #   })

      wandb.finish()


if __name__ == "__main__":
    main()
