
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.python.client import device_lib
tf.test.gpu_device_name()


# In[225]:


import retro
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, Conv3D, Conv1D
from keras.optimizers import Adam
import os # for creating directories


# In[3]:


retro.data.list_games()


# In[4]:


print(retro.__file__)


# In[5]:


env = ""
env = retro.make(game="SuperMarioBros-Nes", state="Level1-1")
# retro.data.get_file_path("SuperMarioBros-Nes", "Scenario2")
retro.data.list_scenarios("SuperMarioBros-Nes")


# In[6]:


state_size = env.observation_space.shape[1] * env.observation_space.shape[0] * env.observation_space.shape[2]
state_size0 = env.observation_space.shape[0]
state_size1 = env.observation_space.shape[1]
state_size2 = env.observation_space.shape[2]
# Might need to make this not rgb or osmething?


# In[138]:


action_size = env.action_space.n
action_size
env.unwrapped.get_action_meaning([0, 1, 1, 0, 0, 0, 1, 1, 1])


# In[101]:


batch_size = 32
n_episodes = 15 # n games we want agent to play (default 1001)
output_dir = 'model\SuperMarioBros-nes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    


# In[308]:


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # double-ended queue; acts like list, but elements can be added/removed from either end
        self.gamma = 0.95 # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon = 1.0 # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995 # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.1 # minimum amount of random exploration permitted
        self.learning_rate = 0.001 # rate at which NN adjusts models parameters via SGD to reduce cost 
        self.model = self._build_model() # private method 
    
    def _build_model(self):
        # neural net to approximate Q-value function: Need to figure out inputs and outputs.
        model = Sequential()
        model.add(Conv2D(32, 3, activation='relu')) # 1st hidden layer; 
        model.add(Dense(32, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear')) #This is the problem. My output from the NN needs to be... 
        #model.add(Flatten())
        
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later

    def act(self, state):
        # We need state to be a ((224*240), 3) format?
        if np.random.rand() <= self.epsilon: # if acting randomly, take random action
            return env.action_space.sample()
        
        act_values = self.model.predict(state) # if not acting randomly, predict reward value based on current state
        act_values = np.where(act_values > 1, 1, 0)
        #print(env.unwrapped.get_action_meaning(act_values))
        return act_values[0][0] # pick the action that will give the highest reward, reutrn them as array.

    def replay(self, batch_size): # method that trains NN with experiences sampled from memory
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
            target = reward # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            if not done: # if not done, then predict future discounted reward
                target = (reward + self.gamma * # (target) = reward + (discount rate gamma) * 
                          np.amax(self.model.predict(next_state))) # (maximum target Q based on future action a')
            target_f = self.model.predict(state) # approximately map current state to future discounted reward
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# In[309]:


state_size = env.observation_space.shape[1] * env.observation_space.shape[0] * env.observation_space.shape[2]
agent = DQNAgent(state_size, action_size) # initialise agent


# In[310]:


done = False
for e in range(n_episodes): # iterate over new episodes of the game
    print(e)
    state = env.reset() # reset state at start of each new episode of the game
    state = np.reshape(state, [state_size0, state_size1, state_size2])
    for time in range(100000):  # time represents a frame of the game; goal is to keep pole upright as long as possible up to range, e.g., 500 or 5000 timesteps
        # env.render()
        action = agent.act(state) # action is either 0 or 1 (move cart left or right); decide on one or other here
        # next_state, reward, done, _ = env.step(str(action)) # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position
        next_state, reward, done, _ = env.step(action) # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position 
        if time % 1000 == 0:
            print(_['time'], reward, action)
            
            
        reward = reward if not done else -10 # reward +1 for each additional frame with pole upright        
        next_state = np.reshape(next_state, [state_size0, state_size1, state_size2])
        agent.remember(state, action, reward, next_state, done) # remember the previous timestep's state, actions, reward, etc.        
        state = next_state # set "current state" for upcoming iteration to the current next state        
        if done: # episode ends if agent drops pole or we reach timestep 5000
            print("episode: {}/{}, score: {}, e: {:.2}" # print the episode's score and agent's epsilon
                  .format(e, n_episodes, time, agent.epsilon))
            break # exit loop
            
    if len(agent.memory) > batch_size:
        agent.replay(batch_size) # train the agent by replaying the experiences of the episode
    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")


# In[275]:


done = False

state = env.reset() # reset state at start of each new episode of the game
state = np.reshape(state, [state_size0, state_size1, state_size2])

action = agent.act(state) # action is either 0 or 1 (move cart left or right); decide on one or other here

next_state, reward, done, _ = env.step(str(action)) # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position
action


# In[289]:


state = env.reset() # reset state at start of each new episode of the game

action = agent.act(state) # action is either 0 or 1 (move cart left or right); decide on one or other here
#print(len(action[0]))

for i in range(0, 10):
    print(action)
    #print(env.unwrapped.get_action_meaning(action))
    


# In[109]:


action_size


# In[326]:


state = env.reset() # reset state at start of each new episode of the game
len(state[0])


# In[331]:


state = env.reset() # reset state at start of each new episode of the game
state[200]
#state[0] is a line of pixels, must be the top line of the screen. All blue or whatever color this is. this is width.
#state[223] is the bottom line of pixels.
#state[0][0] is a single pixel

