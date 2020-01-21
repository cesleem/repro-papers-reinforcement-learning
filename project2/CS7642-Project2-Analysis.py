#!/usr/bin/env python
# coding: utf-8

# # Project 2: Lunar Lander
# 
# ----  
# 
# 

# ## Resources
# 
# The concepts explored in this project are covered by:
# - Lectures
#     - Generalization
# - Readings
#     - Sutton Ch. 9 On-Policy Prediction with Approximation ○ http://incompleteideas.net/book/the-book-2nd.html
# - Source Code
#     - https://github.com/openai/gym
#     - https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
# - Documentation
#     - https://gym.openai.com/docs
# - Examples (use for inspiration, you are still required to write ​your own​ code)
#     - https://gym.openai.com/envs/LunarLander-v2

# ## Background
# 
# TODO

# ## Problem
# TODO
# <img src="img/img-problem-description.png" alt="Drawing" style="width: 750px;" align="left"/>

# ## Instructions
# 
# ### Procedure
# 1. Create an agent capable of solving the Lunar Lander problem found in OpenAI gym
#     - Upload/maintain your code in your private repo at https://github.gatech.edu/gt-omscs-rldm
# 2. Use any RL agent discussed in the class as your inspiration and basis for your program
# 3. Create graphs demonstrating
#     - The reward for each training episode while training your agent
#     - The reward per trial for 100 trials using your trained agent
#     - The effect of hyper-parameters (alpha, lambda, epsilon) on your agent
#         - You pick the ranges
#         - Be prepared to explain why you chose them
#     - Anything else you may think appropriate
# 4. Write a paper describing your agent and the experiments you ran
#     - Include the hash for your last commit to the GitHub repository in the paper’s header.
#     - The rubric includes a few points for formatting. Make sure your graphs are legible and you cite sources properly. While it is not required, we recommend you use a conference paper format. Just pick any one.
#     - 5 pages maximum -- really, you will lose points for longer papers.
#     - Explain your experiments
#     - Graph: Reward at each training episode while training your agent and discussion of results
#     - Graph: Reward per trial for 100 trials using you trained agent and discussion of the results
#     - Graph: Effect of hyperparameters and discussion of the results
#     - Explanation of pitfalls and problems you encountered
#     - Explanation of algorithms used, what worked best, what didn't work
#     - What didn't, what would you try if you had more time?
#     - Discuss your results 
# 
# ### Tips
# 
# 1. If you worked on HW4, you most likely checked out an older version of OpenAI gym. Please remember to update to the latest version for this assignment.
# 2. If you get a Box2D error when running gym.make(‘LunarLander-v2’), you will have to compile Box2D from source. Please follow these steps:
# Try running Lunar Lander again. Source: ​https://github.com/openai/gym/issues/100
# 3. Even if you don't get perfect results or even build an agent that solves the problem you can still write a solid paper.

# ## Step 1: Agent Implementation & Training
# 
# ### Agent: Deep Q-Learning Networks (DQN)
# 
# Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (n.d.). Playing Atari with Deep Reinforcement Learning. 9.

# #### 0. Imports

# In[1]:


#1. Install OpenAI Gym + Setup Env
import gym
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')
#
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# #### 1. Instantiate Environment and Agent

# In[2]:


seed = 0

env = gym.make('LunarLander-v2')
env.seed(seed)

print('Env Seed={}'.format(seed))
print('State shape: ', env.observation_space.shape)
print('Num. of actions: ', env.action_space.n)


# ### Part I: Implement DQ Network
# 
# *Build out `_build_model` for NN model*
# 
# ### Part II: Implement DQN RL Agent (`dqn.agent`)
# 
# *Build out key functions: `add_experience` and batch experience `learn` and `get_action`*

# In[3]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

np.random.seed(seed)
random.seed(seed)

class DQNAgent():
    
    def __init__(self,
                 state_size, 
                 action_size, 
                 gamma=.99, 
                 epsilon=1.0, 
                 epsilon_min=.01, 
                 epsilon_decay=.995,
                 hidden_sizes=[64,128,64], 
                 learning_rate=5e-4, 
                 BATCH_SIZE=32,
                 BUFFER_SIZE=int(1e5)):
        
        """Initialize parameters and build agent w/DQNetwork."""

        self.state_size = state_size
        self.action_size = action_size 
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.batch_size = BATCH_SIZE
        self.model = self._build_model(self.state_size, 
                                self.action_size,  
                                hidden_sizes)
        self.t_step = 0
        
    def _build_model(self, input_size, 
                        output_size, 
                        hidden_sizes=[64,128,64]):
        
        """Initialize parameters and build model.
            Hidden layers = relu activation 
            Output layer = linear activation
            
            Loss function: Mean Squared Error
            Optimizer: Adam
         """

        
        #setup NN
        model = Sequential()
        
        #input layer
        model.add(Dense(hidden_sizes[0], input_dim=input_size, activation='relu'))
        
        #hidden layers
        for h_size in hidden_sizes[1:]:
            model.add(Dense(h_size, activation='relu'))
        
        #output layer
        model.add(Dense(output_size, activation='linear'))
        
        #compile model
        model.compile(loss='mean_squared_error', 
                      optimizer=Adam(lr=self.learning_rate))
        
        return model
        
    
    def get_action(self, state, test=False):
        if test:                                                    #turn off exploration for testing agent
            self.epsilon = 0.0

        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])
    
    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def experience_replay(self, minibatch):
        
        states, actions, rewards, next_states, dones = minibatch.T
        
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)
        
        #predict over next_states to calculate Q_targets
        Q_targets_next = self.model.predict(next_states)
        Q_targets = rewards + (1 - dones) * (self.gamma * np.max(Q_targets_next, axis=1))
        
        #update with new target for given action over predicted current state
        Q_expected = self.model.predict(states)
        Q_expected[range(Q_expected.shape[0]), actions.astype(int)] = Q_targets
        
        self.model.fit(states, Q_expected, epochs=1, verbose=0)
        
    
    def learn(self, state, action, reward, next_state, done, UPDATE_EVERY=1):
        
        #add experience to memory
        self.add_experience(state, action, reward, next_state, done)
        
        self.t_step += 1
        
        if len(self.memory) > self.batch_size and self.t_step % UPDATE_EVERY == 0:
        
            #sample minibatch of memory to train
            minibatch = random.sample(self.memory, self.batch_size)
        
            #train
            self.experience_replay(np.array(minibatch))
        
            #Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
    def save(self, filename):
        self.model.save_weights(filename)
            
    def load(self, filename):
        self.model.load_weights(filename)


# ### Part III: Train

# #### Hyperparameters

# In[4]:


state_size = env.observation_space.shape[0]
action_size = env.action_space.n
gamma = 0.99    
epsilon = 1.0  
epsilon_min = 0.01
epsilon_decay = 0.995

learning_rate = 5e-4
BATCH_SIZE = 24                                 #TODO - 32

BUFFER_SIZE = int(1e5)
UPDATE_EVERY = 1

n_episodes=5000
max_t=1000


# #### Train and Collect Stats

# In[16]:


def train_agent(agent, n_episodes=5000, max_t=1000, UPDATE_EVERY=1, verbose=True, filename='lunarlander-dqn.h5'):

    scores = []                        # store rewards from each episode for graphing
    epsilons = []                      # Store eps for tracking exploration

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        score = 0
        for t in range(max_t):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            #one step in env = add to memory + replay
            agent.learn(state, action, reward, next_state, done, UPDATE_EVERY)

            state = next_state
            score += reward

            if done:
                break

        epsilons.append(agent.epsilon)
        scores.append(score)
        
        if verbose:
            curr_mean_score = np.mean(scores[-100:])

            print('\n\rEpisode {}\tAverage Score: {:.2f} \t Epsilon {:.6f}'.format(i_episode, curr_mean_score, agent.epsilon), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f} \t Epsilon {:.6f}'.format(i_episode, curr_mean_score, agent.epsilon))
                agent.save("project2/model/{}".format(filename))
            if curr_mean_score >= 200:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, curr_mean_score))
                break

    return scores, epsilons


# In[5]:


agent = DQNAgent(state_size=state_size,
                 action_size=action_size,
                 gamma=gamma,
                 epsilon=epsilon,
                 epsilon_min=epsilon_min,
                 epsilon_decay=epsilon_decay,
                 learning_rate=learning_rate,
                 BATCH_SIZE=BATCH_SIZE,
                 BUFFER_SIZE=BUFFER_SIZE)

scores, epsilons = train_agent(agent, n_episodes=5000, max_t=1000, UPDATE_EVERY=1, verbose=True)


## Step 2: Analysis

### I. Graph: Reward at each training episode while training your agent and discussion of results

# In[8]:


rolling_average = np.convolve(scores, np.ones(100)/100)

# plot the scores
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.title('Average Rewards for Lunar Lander in Training over {} Episodes'.format(i_episode))

plt.plot(np.arange(len(scores)), scores, label='reward total')
plt.plot(rolling_average, color='black', label='reward total (smoothed)')
plt.axhline(y=200, color='r', linestyle='-', label='solved line') #Solved Line

#Scale Epsilon (0.001 - 1.0) to match reward (0 - 200) range
eps_graph = [200*x for x in epsilons]
plt.plot(eps_graph, color='g', linestyle='-', label='epsilon (scaled)')

ax.legend()

plt.ylabel('Score')
plt.xlabel('Episode #')

plt.xlim( (0, i_episode) )
plt.ylim( (0, 220) )

fig.savefig('./figures/Project2-LunarLander-v1.0-fig1-training-tot-rewards.png', dpi=fig.dpi)


### II. Graph: Reward per trial for 100 trials using you trained agent and discussion of the results

# In[10]:


#Trained agent runs
try:
    agent
except:
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    agent.load("project2/model/lunarlander-dqn.h5")

n_episodes_TEST = 100

scores_TEST = []                        # store rewards from each episode for graphing

for i_episode in range(1, n_episodes_TEST+1):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    score = 0
    for t in range(max_t):
        action = agent.get_action(state, test=True)              #TEST MODE: Run trained agent
        next_state, reward, done, _ = env.step(action)

        env.render()                                             #Show viz of agent

        next_state = np.reshape(next_state, [1, state_size])

        state = next_state
        score += reward

        if done:
            break

    scores_TEST.append(score)
    print('\rEpisode {}\tAverage Score: {:.2f} \t Epsilon {:.6f}'.format(i_episode, score, agent.epsilon))


# In[14]:


# plot the scores
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.title('100 Episodes w/Trained Lunar Lander Agent')

plt.plot(np.arange(len(scores_TEST[-100:])), scores_TEST[-100:])
plt.axhline(y=200, color='r', linestyle='-', label='solved line') #Solved Line
plt.ylabel('Score')
plt.xlabel('Episode #')

fig.savefig('./figures/Project2-LunarLander-v1.0-fig2-trained-agent-tot-rewards.png', dpi=fig.dpi)


### III. Graph: Effect of hyperparameters (`UPDATE_EVERY`, size of NN `model`) and discussion of the results

# In[17]:


rolling_average = lambda scores: np.convolve(scores, np.ones(100)/100)

UPDATE_EVERY_1 = 1
UPDATE_EVERY_2 = 4
UPDATE_EVERY_3 = 8

print('------------- running experiments: UPDATE_EVERY -------------')

exp1_scores_1, _ = train_agent(DQNAgent(state_size=state_size, action_size=action_size), n_episodes = 1000, UPDATE_EVERY = UPDATE_EVERY_1, filename='lunarlander-dqn-update-{}.h5'.format(UPDATE_EVERY_1))
exp1_scores_2, _ = train_agent(DQNAgent(state_size=state_size, action_size=action_size), n_episodes = 1000, UPDATE_EVERY = UPDATE_EVERY_2, filename='lunarlander-dqn-update-{}.h5'.format(UPDATE_EVERY_2))
exp1_scores_3, _ = train_agent(DQNAgent(state_size=state_size, action_size=action_size), n_episodes = 1000, UPDATE_EVERY = UPDATE_EVERY_3, filename='lunarlander-dqn-update-{}.h5'.format(UPDATE_EVERY_3))

# plot the scores
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.title('DQN Agent Training vs. UPDATE_EVERY training schedule')

plt.axhline(y=200, color='r', linestyle='-') #Solved Line

plt.plot(np.arange(len(exp1_scores_1)), exp1_scores_1, label='1')
plt.plot(rolling_average(exp1_scores_1)[:len(exp1_scores_1)], color='black')

plt.plot(np.arange(len(exp1_scores_2)), exp1_scores_2, label='4')
plt.plot(rolling_average(exp1_scores_2)[:len(exp1_scores_2)], color='black')

plt.plot(np.arange(len(exp1_scores_3)), exp1_scores_3, label='8')
plt.plot(rolling_average(exp1_scores_3)[:len(exp1_scores_3)], color='black')

plt.xlim( (0, 1000) )
plt.ylim( (-220, 220) )

ax.legend()

plt.ylabel('Score')
plt.xlabel('Episode #')

fig.savefig('project2/figures/Project2-LunarLander-v1.0-fig3-hyperparams-update-every.png', dpi=fig.dpi)


# In[ ]:


batch_size_1 = 24
batch_size_2 = 32
batch_size_3 = 64

print('------------- running experiments: BATCH_SIZE -------------')

exp2_scores_1, _ = train_agent(DQNAgent(state_size=state_size, action_size=action_size, BATCH_SIZE = batch_size_1), n_episodes = 1000, filename='lunarlander-dqn-batch_size-{}.h5'.format(batch_size_1))
exp2_scores_2, _ = train_agent(DQNAgent(state_size=state_size, action_size=action_size, BATCH_SIZE = batch_size_2), n_episodes = 1000, filename='lunarlander-dqn-batch_size-{}.h5'.format(batch_size_2))
exp2_scores_3, _ = train_agent(DQNAgent(state_size=state_size, action_size=action_size, BATCH_SIZE = batch_size_3), n_episodes = 1000, filename='lunarlander-dqn-batch_size-{}.h5'.format(batch_size_3))

# plot the scores
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.title('DQN Agent Training vs. batch size')

plt.axhline(y=200, color='r', linestyle='-') #Solved Line

plt.plot(np.arange(len(exp2_scores_1)), exp2_scores_1, label='24')
plt.plot(rolling_average(exp2_scores_1)[:len(exp2_scores_1)], color='black')

plt.plot(np.arange(len(exp2_scores_2)), exp2_scores_2, label='32')
plt.plot(rolling_average(exp2_scores_2)[:len(exp2_scores_2)], color='black')

plt.plot(np.arange(len(exp2_scores_3)), exp2_scores_3, label='64')
plt.plot(rolling_average(exp2_scores_3)[:len(exp2_scores_3)], color='black')

plt.xlim( (0, 1000) )
plt.ylim( (-220, 220) )

ax.legend()

plt.ylabel('Score')
plt.xlabel('Episode #')

fig.savefig('project2/figures/Project2-LunarLander-v1.0-fig3-hyperparams-batch-size.png', dpi=fig.dpi)


# In[ ]:


hidden_sizes_1 = [32, 32]
hidden_sizes_2 = [32, 64, 32]
hidden_sizes_3 = [64, 128, 64]

hidden_sizes_4 = [256, 256]

print('------------- running experiments: HIDDEN_SIZES -------------')

exp3_scores_1, _ = train_agent(DQNAgent(state_size=state_size, action_size=action_size, hidden_sizes = hidden_sizes_1), n_episodes = 1000, filename='lunarlander-dqn-nn_hidden_size-{}.h5'.format('32x32'))
exp3_scores_2, _ = train_agent(DQNAgent(state_size=state_size, action_size=action_size, hidden_sizes = hidden_sizes_2), n_episodes = 1000, filename='lunarlander-dqn-nn_hidden_size-{}.h5'.format('32x64x32'))
exp3_scores_3, _ = train_agent(DQNAgent(state_size=state_size, action_size=action_size, hidden_sizes = hidden_sizes_3), n_episodes = 1000, filename='lunarlander-dqn-nn_hidden_size-{}.h5'.format('64x128x64'))

# just for fun lets see what happens when you spread same # edges amongst two layers (collapse one hidden layer)
exp3_scores_4, _ = train_agent(DQNAgent(state_size=state_size, action_size=action_size, hidden_sizes = hidden_sizes_4), n_episodes = 1000, filename='lunarlander-dqn-nn_hidden_size-{}.h5'.format('256x256'))

# plot the scores
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.title('DQN Agent Training vs. NN size')

plt.axhline(y=200, color='r', linestyle='-') #Solved Line

plt.plot(np.arange(len(exp3_scores_1)), exp3_scores_1, label='32x32')
plt.plot(rolling_average(exp3_scores_1)[:len(exp3_scores_1)], color='black')

plt.plot(np.arange(len(exp3_scores_2)), exp3_scores_2, label='32x64x32')
plt.plot(rolling_average(exp3_scores_2)[:len(exp3_scores_2)], color='black')

plt.plot(np.arange(len(exp3_scores_3)), exp3_scores_3, label='64x128x64')
plt.plot(rolling_average(exp3_scores_3)[:len(exp3_scores_3)], color='black')

plt.plot(np.arange(len(exp3_scores_4)), exp3_scores_4, label='256x256')
plt.plot(rolling_average(exp3_scores_4)[:len(exp3_scores_4)], color='black')

plt.xlim( (0, 1000) )
plt.ylim( (-220, 220) )

ax.legend()

plt.ylabel('Score')
plt.xlabel('Episode #')

fig.savefig('project2/figures/Project2-LunarLander-v1.0-fig3-hyperparams-nn-size.png', dpi=fig.dpi)


# ## Possible Extensions
# 0. Avg of 10 runs per agent training in experiment as training varies wildly
# Exp 1 - 1,2,3,4,5 updates
# 1. Fixed Q Targets
# 2. Fancy Replay Buffer
# 3. PyTorxch/GPU speedups?
# 4. Updated Python 3.7+
