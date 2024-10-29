import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import random


window_size = 50
SIZE = 20
EPISODES = 3000
RENDER = 3000
ALPHA = 0.1
GAMMA = 0.99  # try 0.9: the snake we'll probably be stuck
DECAY = 0.999
REWARD_INIT = 3000
STEP_PENALTY = 3
PAUSE_DURATION = 0.000001 
START = np.array(random.choice([[1, 1], [1, SIZE-2], [SIZE-2, 1], [SIZE-2, SIZE-2]]))
TARGET = np.array(random.choice([[1, 1], [1, SIZE-2], [SIZE-2, 1], [SIZE-2, SIZE-2]]))
while TARGET[0] == START[0] or TARGET[1] == START[1]:
    TARGET = np.array(random.choice([[1, 1], [1, SIZE-2], [SIZE-2, 1], [SIZE-2, SIZE-2]]))



def generate_maze(rows, cols):
    maze = np.ones((rows, cols), dtype=int)
    stack = [(0, 0)]
    
    while stack:
        current_cell = stack[-1]
        maze[current_cell] = 0
        neighbors = []
        
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = current_cell[0] + dx, current_cell[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 1:
                neighbors.append((nx, ny))
        
        if neighbors:
            next_cell = random.choice(neighbors)
            wall_cell = (
                current_cell[0] + (next_cell[0] - current_cell[0]) // 2,
                current_cell[1] + (next_cell[1] - current_cell[1]) // 2
            )
            maze[wall_cell] = 0
            stack.append(next_cell)
        else:
            stack.pop()
    
    # Set the starting position for the bot at (0, 0)
    maze[START] = 0
    maze[TARGET] = 0

    return maze


MAZE = generate_maze(SIZE, SIZE)


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=SIZE, shape=(10,), dtype=np.int32)

        self.maze = MAZE
        
        self.state = START
        self.target = TARGET

        self.trajectory = []

    def reset(self):
        self.state = START
        self.trajectory = [self.state.copy()]
        return self.state

    def estimate_next_state(self, action):
        state = self.state.copy()
        if action == 0 and self.state[1] < SIZE:
            state[1] += 1
        elif action == 1 and self.state[1] > 0:
            state[1] -= 1
        elif action == 2 and self.state[0] < SIZE:
            state[0] += 1
        elif action == 3 and self.state[0] > 0:
            state[0] -= 1
        return state

    def is_valid_state(self, state):
        return (
            0 <= state[0] < self.maze.shape[0] and
            0 <= state[1] < self.maze.shape[1] and
            self.maze[state[0], state[1]] == 0
        )

    def step(self, action):
        
        prev_state = self.state.copy()
        state = self.estimate_next_state(action)
        if self.is_valid_state(state):
            self.state = state
            
        self.trajectory.append(self.state.copy())
        reward = - np.linalg.norm(self.state - self.target) + np.linalg.norm(prev_state - self.target) - STEP_PENALTY # time is important, but less than taking a good direction.
        done = np.linalg.norm(self.state - self.target) < 1
        return self.state, reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            plt.clf()
            plt.xlim(0, SIZE)
            plt.ylim(0, SIZE)

            # Plot the maze grid
            for i in range(SIZE):
                for j in range(SIZE):
                    if self.maze[i, j] == 1:
                        plt.plot(j, i, 'ks', markersize=10)  # Mark walls as black squares

            plt.plot(self.target[1], self.target[0], 'go', markersize=10, label='Target')
            trajectory = np.array(self.trajectory)
            plt.plot(trajectory[:, 1], trajectory[:, 0], 'b-', label='Agent trajectory')
            plt.plot(trajectory[-1, 1], trajectory[-1, 0], 'ro', label='Current position')
            plt.legend()
            plt.pause(PAUSE_DURATION)
            plt.draw()
        else:
            super(CustomEnv, self).render(mode=mode)




def state_to_col(state):
    return state[0] * SIZE + state[1]

def col_to_state(col):
    return [col//SIZE, col%SIZE]


class QLearningAgent:
    def __init__(self, action_space_size, observation_space_size):
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        self.q_table = np.zeros((observation_space_size, action_space_size))

        self.alpha = ALPHA # Learning rate
        self.gamma = GAMMA # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = DECAY

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_size)
        return np.argmax(self.q_table[state_to_col(state), :])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[state_to_col(next_state), :])

        self.q_table[state_to_col(state), action] += self.alpha * (
            reward + self.gamma * self.q_table[state_to_col(next_state), best_next_action] - self.q_table[state_to_col(state), action]
        )
        
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')




if __name__ == "__main__":
    env = CustomEnv()
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    obs_bins = [SIZE+1, SIZE+1]  # Number of bins for each dimension
    agent = QLearningAgent(env.action_space.n, np.prod(obs_bins))
    episode_rewards = []
    episode_epsilons = []  # Create a list to store epsilon values

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = REWARD_INIT - np.linalg.norm(env.state - env.target)
        while not done and total_reward > 0:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            if (episode % RENDER == 0 and episode >= RENDER) or episode == EPISODES - 1:
                env.render('human')
        episode_rewards.append(total_reward)
        episode_epsilons.append(agent.epsilon)  # Append epsilon value

        agent.decay_epsilon()
        print(f"Episode {episode + 1} | Total Reward: {total_reward}, Epsilon: {agent.epsilon}, Sucess: {done}")
    
    smoothed_rewards = moving_average(episode_rewards, window_size)
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(smoothed_rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Epsilon', color=color)  # we already handled the x-label with ax1
    ax2.plot(episode_epsilons, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Evolution of Rewards and Epsilon')
    plt.show()

    
    env.close()
