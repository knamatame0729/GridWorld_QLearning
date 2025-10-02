import numpy as np                # needed for numerica computaion
import matplotlib.pyplot as plt   # needed for plotting
import wandb                      # needed for tracking metrics

# Log in to W&B
wandb.login()

# Project name for W&B
project = "gridworld_q_learning_run3"

# Grid World setting
ROWS, COLS = 3, 4      # number of rows and columns
WIN_STATE = (0,3)      # goal state coordinates
LOSE_STATE = (1,3)     # lose state coordinates
START = (2,0)          # start state coordinates
WALL = (1,1)           # wall state coordinates

NUM_EPISODES = 500     # number of training episodes

GOAL_REWARD = 1        # Reward for reachign goal

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


class Agent:
    """
    Q-Learning agent for teh grid world
    """
    def __init__(self, lose_reward=-1):
        self.actions = ["up", "down", "left", "right"]  # actions
        self.State = State()                            # initialize environment state
        self.lr = 0.1                                   # learning rate
        self.exp_rate = 1.0                             # exploration rate
        self.decay_gamma = 0.9                          # discount factor
        self.exp_decay = 0.995                          # epsilon decay per episode
        self.min_exp_rate = 0.01                        # miminum epsilon
        self.lose_reward = lose_reward                  # reward for losing


        # Initialize Q-Values: Q(s,a) for each state-action pair
        self.Q_values = {}
        for i in range(ROWS):
            for j in range(COLS):
                self.Q_values[(i,j)] = {}
                for a in self.actions:
                    self.Q_values[(i,j)][a] = 0.0


        self.episode_rewards = []                       # Store rewards per episode
        self.episode_steps = []                         # Store steps per episode
        self.q_deltas = []                              # Store Q-Value difference


    def get_policy(self):
        """
        Return the current policy based on Q_values
        """
        policy = {}

        for i in range(ROWS):
            for j in range(COLS):
                state = (i,j)
                if state in [WALL, WIN_STATE, LOSE_STATE]:
                    continue

                qvals = self.Q_values[state]                                             # Get Q-Values for the curent state
                best_action = max(qvals, key=qvals.get)                                  # Find the action with the highest Q-value
                policy[state] = best_action                                              # Add policy dictionary
        return policy

    def q_delta(self):
        """
        Calculate teh average maximum difference between Q-Values and the best Q-Value
        """
        q_deltas = []
        for i in range(ROWS):
            for j in range(COLS):
                if (i,j) in [WALL, WIN_STATE, LOSE_STATE]:
                    continue
                
                qvals = self.Q_values[(i,j)]                                             # Get Q-Values for the current state
                best_value = max(qvals.values())                                         # Find the maximum Q-Value
                differences = [abs(value - best_value) for value in qvals.values()]      # Compute
                max_difference = max(differences)                                        # Take the largest                 
                q_deltas.append(max_difference)

        # Return the mean delta
        if q_deltas:
            return np.mean(q_deltas)
        else:
            return 0.0

    def chooseAction(self):
        """
        Choose action using epsilon-greedy
        """
        if np.random.uniform(0,1) <= self.exp_rate:
            return np.random.choice(self.actions)

        qvals = self.Q_values[self.State.state]

        return max(qvals, key=qvals.get)

    def takeAction(self, action):
        """
        Take action and return new state
        """
        position = self.State.move(action)
        return State(state=position, lose_reward=self.lose_reward)

    def reset(self):
        """
        Reset environment for new episode
        """
        self.State = State(lose_reward=self.lose_reward)
        self.State.isEndFunc()
        self.isEnd = self.State.isEnd

    def train(self, num_episodes=NUM_EPISODES):
        """
        Train agent using Q learning
        Updates Q-Values and tracks episode metrics
        Logs metrics to wandb (study)
        """
        previous_policy = None
        policy_changes = []

        for i in range(num_episodes):
            self.reset()
            episode_reward = 0
            steps = 0

            while not self.isEnd:
                # Current state and chosen action
                s = self.State.state
                a = self.chooseAction()

                # take action, get next state and reward
                self.State = self.takeAction(a)
                s_next = self.State.state
                r = self.State.reward()
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd

                episode_reward += r
                steps += 1

                # Q-learning update
                if self.isEnd:
                    target = r    # Terminal state (no future reward)
                else:
                    target = r + self.decay_gamma * max(self.Q_values[s_next].values())
                self.Q_values[s][a] += self.lr * (target - self.Q_values[s][a])
            
            # Record episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)
            self.q_deltas.append(self.q_delta())

            # Get current policy (best action for each state)
            current_policy = self.get_policy()

            policy_change = 0

            # Compare with previous policy
            if previous_policy is not None:
                for state in current_policy:                                   # Iterate all states in teh curret policy
                    if state in previous_policy:                               # Ensure the state exists in the previous policy
                        if current_policy[state] != previous_policy[state]:    # Check if the action chaged
                            policy_change += 1                                 # Increment

            # Append
            policy_changes.append(policy_change)
            previous_policy = current_policy

            # Log to wandb
            wandb.log({
                "episode": i,
                f"reward_L_{self.lose_reward}": episode_reward,
                f"steps_L_{self.lose_reward}": steps,
                "epsilon": self.exp_rate,
                f"policy_change_L_{self.lose_reward}": policy_change,
                f"q_delta_L_{self.lose_reward}": self.q_deltas[-1]
            })

            # Decay exploraion rate
            self.exp_rate = max(self.min_exp_rate, self.exp_rate * self.exp_decay)

            # Progress update
            if i % 100 == 0:
                print(f"Episode {i}, ε: {self.exp_rate:.3f}, Reward: {episode_reward:.3f}, Step: {steps}")
        
        print(f"Training completed after {num_episodes} episodes.")
        return policy_change

def plot_policy(agent):
    # Initialize a dictonary to store
    policy={}

    for i in range(ROWS):
        for j in range(COLS):
            if (i,j) in [WALL, WIN_STATE, LOSE_STATE]:
                continue

            # Select the action with highest Q-Value
            policy[(i,j)] = max(agent.Q_values[(i,j)], key=agent.Q_values[(i,j)].get)

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
                
            # Get Q-Value in (i, j)
            qvals = agent.Q_values[(i,j)]

            # Determine arrow based on policy 
            if  (i,j) in policy:
                action = policy[(i, j)]
                arrow = arrow_dic[action]
            else:
                arrow = " "

            # Display Q-Values in four directions
            ax.text(x, y + 0.3, f"{qvals['up']:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')
            ax.text(x, y - 0.3, f"{qvals['down']:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')
            ax.text(x - 0.3, y, f"{qvals['left']:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')
            ax.text(x + 0.3, y, f"{qvals['right']:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=10, color='blue')

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
    # Initiallize W&B
    run = wandb.init(project=project)
    config = wandb.config

    run.name = f"lr_{config.lr}_gamma_{config.decay_gamma}_exp_decay_{config.exp_decay}_exp_rate_{config.exp_rate}_lose_{config.lose_reward}"

    # Create Q-Learning agent and set hyperparameters from the sweep configration
    agent = Agent(lose_reward=config.lose_reward)
    agent.lr = config.lr
    agent.decay_gamma = config.decay_gamma
    agent.exp_decay = config.exp_decay
    agent.exp_rate = config.exp_rate

    # Update the W&B config with the agnet's hyperarameters for reference
    wandb.config.update({
        "learning_rate": agent.lr,
        "gamma": agent.decay_gamma,
        "epsilon_decay": agent.exp_decay,
        "epsilon_start": agent.exp_rate,
        "lose_reward": agent.lose_reward,
        "episodes": NUM_EPISODES
    })

    # Train the agent
    agent.train(NUM_EPISODES)

    # Compute the final average reward over last 200 episodes
    # Using last 100 rewards becasue the agent has already learned most of its policy by then
    avg_reward = np.mean(agent.episode_rewards[-200:])
    wandb.log({f"final_avg_reward_L_{agent.lose_reward}": avg_reward})

    # Generate a policy map
    fig = plot_policy(agent)
    # Log policy map image to W&B
    wandb.log({f"policy_map_L_{agent.lose_reward}": wandb.Image(fig)})
    plt.close(fig)

    # Store the policy
    policy = {}

    for i in range(ROWS):
        for j in range(COLS):
            state = (i,j)
            
            # Skip wall, win, and lose state
            if state in [WALL, WIN_STATE, LOSE_STATE]:
                continue
                        
            # For the current state, find the action with the highest Q-value
            qvals = agent.Q_values[state]
            best_action = max(qvals, key=qvals.get)

            # Add it to the policy dictionary
            policy[state] = best_action
    
    # convert teh policy dictionary to a W&B table and log it
    policy_table_data = []
    for state, action in policy.items():
        policy_table_data.append([str(state), action])

    wandb.log({
        f"policy_table_L_{agent.lose_reward}": wandb.Table(data=policy_table_data, columns=["state", "action"]
        )
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()

    


