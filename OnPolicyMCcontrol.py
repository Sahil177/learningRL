import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
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


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        state = env.reset()
        episode = []
        for t in range(100):
            action = np.random.choice(range(env.action_space.n), p=policy(state))
            next_state, reward, done, _ = env.step(action)
            episode.append((tuple(state), action, reward))
            if done:
                break
            state = next_state

        states_visited = set([x[0] for x in episode])
        states_ep = [x[0] for x in episode]
        for state in states_visited:
            first_occ = min([idx for idx, statep in enumerate(states_ep) if state == statep ])
            g2 = sum([x[2]*discount_factor**i for i, x in enumerate(episode[first_occ:])])
            state_action_pair = (state, episode[first_occ][1])
            returns_sum[state_action_pair] += g2
            returns_count[state_action_pair] += 1
            Q[state][episode[first_occ][1]] = returns_sum[state_action_pair]/returns_count[state_action_pair]
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    return Q, policy

    
Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)


# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")