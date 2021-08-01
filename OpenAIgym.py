import gym
import random
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMsprop

def NN(input_shape, action_space):
    X_input = Input(input_shape)

    X = Dense(512, input_shape= input_shape, activation = 'relu', kernel_initializer = 'he_uniform')(X_input)
    X = Dense(256, activation = 'relu', kernel_initializer = 'he_uniform')(X)
    X = Dense(64, activation = 'relu', kernel_initializer = 'he_uniform')(X)
    X = Dense(action_space, activation = 'linear', kernel_initializer = 'he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='DQN for cartpole')
    model.compile(loss= 'mse', optimizer=RMsprop(lr=0.00025, rho=0.95, epsilon = 0.01), metrics = ["accuracy"])
    model.summary()
    return model

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        


env = gym.make('CartPole-v1')

def Random_games():

    for episode in range(10):
        env.reset()

        for t in range(500):
            env.render()
            action = env.action_space.sample()

            next_state, reward, done, info = env.step(action)

            print(t, next_state, reward, done, info, action)
            if done:
                break


Random_games()
