import gym
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.pyplot import figure, draw, pause
import numpy as np
from collections import defaultdict
import time
import operator
import pickle

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
s = time.time()

fg = figure()
ax = fg.gca()



def clamp(lower, val, upper):
    if val > upper:
        return upper
    elif val < lower:
        return lower
    else:
        return val 

global grid 
grid = np.zeros((32,17))
h = ax.imshow(grid)
#Setting up environment

rows = [ (0,13), 
        (0,14), 
        (0,14), 
        (0,15), 
        (0,16), 
        (0,16), 
        (7,16), 
        (8,16), 
        (8,16), 
        (8,16), 
        (8,16), 
        (8,16), 
        (8,16), 
        (8,16),
        (8,15),
        (8,15),
        (8,15),
        (8,15),
        (8,15),
        (8,15),
        (8,15),
        (8,15),
        (8,14),
        (8,14),
        (8,14),
        (8,14),
        (8,14),
        (8,14),
        (8,14),
        (8,13),
        (8,13),
        (8,13)]

start = [(len(rows)-1, i) for i in range(rows[-1][0], rows[-1][1]+1)]

end = [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0)]

grid_dict = {}

for i, row in enumerate(rows):
    for j in range(row[0], row[1]+1):
        grid[i][j] = 1
        grid_dict[(i,j)] = 1

h = ax.imshow(grid)

#plt.imshow(np.fliplr(grid))
#plt.show()

class RaceTrackenv:
    def __init__(self, start, end, grid, grid_dict):
        self.start = start
        self.end = {e:1 for e in end}
        self.grid = grid
        self.grid_dict = grid_dict
        self.action_space = [(i,j) for i in [-1,0,1] for j in [0,-1,1]]
        self.done = False
    
    def reset(self):
        start_pos = np.random.choice(range(len(self.start)))
        self.state = (start[start_pos], (0,0))
        self.done = False
        return self.state
    
    def step(self, action):
        next_pos = tuple(map(lambda x, y: x + y, self.state[0], self.state[1]))
        if next_pos in self.end:
            self.done = True

        next_vel = tuple(map(lambda x, y: x + y, self.action_space[action], self.state[1]))
        next_vel = (clamp(-4,next_vel[0],0), clamp(-4,next_vel[1],0))
        if next_vel == (0,0):
            next_vel = self.state[1]
        if next_pos not in self.grid_dict:
            next_pos = start[np.random.choice(range(len(self.start)))]
            next_vel = (0,0)
        next_state = (next_pos, next_vel)
        env.state = next_state
        return next_state, -1, self.done

def make_epsilon_greedy_policy(epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    Q = defaultdict(lambda: np.zeros(nA))

    def policy_fn(state):
        new_policy = np.zeros(nA)
        action_vals = Q[state]
        maxidx = np.argmax(action_vals)
        for i in range(nA):
            if i != maxidx:
                new_policy[i] = epsilon/nA
            else:
                new_policy[i] = 1- epsilon + epsilon/nA
        return new_policy

    return policy_fn

def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn

def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.
    
    Args:
        Q: A dictionary that maps from state -> action values
        
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """
    
    def policy_fn(state):
        action_vals = Q[state]
        return np.eye(len(action_vals))[np.argmax(action_vals)]

    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, Q, C, discount_factor=1.0 ):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Gamma discount factor.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """
    
    # The final action-value function.
    # A dictionary that maps state -> action values
    
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)

    for i_episode in range(num_episodes):

        #if i_episode%100 == 0 or i_episode == 1:
        #s = time.time()
        state = env.reset()
        episode = []
        c = 0
        while True:
            action = np.random.choice(range(len(env.action_space)), p=behavior_policy(state))
            next_state, reward, done = env.step(action)
            episode.append((tuple(state), action, reward))
            #print(np.array((tuple(state), action, reward)))
            if done:
                break
            state = next_state
            point = state[0]
            #grid[point[0]][point[1]] = 0
            #h.set_data(np.fliplr(grid))
            #draw(), pause(1e-3)
            #grid[point[0]][point[1]] = 1
            c +=1
        if i_episode%500 == 0:
            print(i_episode, c)
        
        g = 0
        W = 1
        for t in reversed(range(len(episode))):
            state = episode[t][0]
            action = episode[t][1]
            reward = episode[t][2]
            g = discount_factor*g + reward
            C[state][action] += W
            Q[state][action] += (W/C[state][action])*(g-Q[state][action])
            target_policy = create_greedy_policy(Q)
            if action != np.argmax(target_policy(state)):
                break
            W = W/behavior_policy(state)[action]
        #f = time.time()
        #print(f-s)

    return Q, C, target_policy



env = RaceTrackenv(start, end, grid, grid_dict)
#Q= defaultdict(lambda: np.zeros(len(env.action_space)))
a_file = open(dir_path + "\Q_dict", "rb")
Q= defaultdict(lambda: np.zeros(len(env.action_space)),pickle.load(a_file))
#C = defaultdict(lambda: np.zeros(len(env.action_space)))
b_file = open(dir_path + "\C_dict", "rb")
C= defaultdict(lambda: np.zeros(len(env.action_space)),pickle.load(b_file))


epsilon_policy = make_epsilon_greedy_policy(0.9, len(env.action_space))
epsilon_policy2 = make_epsilon_greedy_policy(0.1, len(env.action_space))
random_policy = create_random_policy(len(env.action_space))
Q1,C, policy = mc_control_importance_sampling(env, num_episodes=300000, behavior_policy=epsilon_policy, Q = Q, C = C, discount_factor=0.9)

new_Q1 = {key:Q1[key] for key in Q1}

Q_dict = open(dir_path + '\Q_dict', 'wb')
pickle.dump(new_Q1, Q_dict)
Q_dict.close()

new_C = {key:C[key] for key in C}

C_dict = open(dir_path + '\C_dict', 'wb')
pickle.dump(new_C, C_dict)
C_dict.close()

f = time.time()
print(len(new_Q1))
print(f"Exectuion time: {f-s}")
'''
Q2, policy2 = mc_control_importance_sampling(env, num_episodes=100, behavior_policy=epsilon_policy2, Q = Q1, discount_factor=0.9)

new_Q2 = {key:Q2[key] for key in Q2}

Q_dict2 = open('Q_dict2', 'wb')
pickle.dump(new_Q2, Q_dict2)
Q_dict2.close()

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q2.items():
    action_value = np.max(action_values)
    V[state] = action_value

test_start = (31,9)

episode = []
state = env.reset()
state = (test_start, (0,0))
path = []
while True:
    action = np.argmax(Q2[state])
    next_state, reward, done = env.step(action)
    episode.append((tuple(state), action, reward))
    print(np.array((tuple(state), action, reward)))
    path.append(state[0])
    if done:
        break
    state = next_state


print(path)

for point in path:
    grid[point[0]][point[1]] = 0


plt.imshow(np.fliplr(grid))
plt.show()
'''













