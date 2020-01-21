#!/usr/bin/env python
# coding: utf-8

# # Project #1: Desperately Seeking Sutton

# ## TODOs
# 2. Add Git commit

# ## Resources

# The concepts explored in this project are covered by:
# - Lectures
#     - [Lesson 4: TD and Friends](https://classroom.udacity.com/courses/ud600/lessons/4178018883/concepts/43095686540923)
# - Readings
#     - [Learning to Predict by the Methods of Temporal Differences](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)

# ## Instructions

# #### 1. Read Sutton's Paper
# #### 2. Write the code necessary to replicate Sutton's experiments
#     - You will be replicating figures 3, 4, and 5 (Check Erratum at the end of paper and [Ch. 7 Sutton Textbook](http://incompleteideas.net/book/ebook/node73.html)) 
# #### 3. Create the graphs
#     - Replicate figures 3, 4, and 5
# #### 4. Write a paper describing the experiments, how you replicated them, and any other relevant information.
#     - Reminders:
#         - Include the hash for your last commit to the GitHub repository in the paper’s header.
#         - 5 pages maximum
#         - use a conference paper format
#     - Contents:
#         - Describe the problem
#         - Your graphs 
#         - Describe the experiments
#             - Discuss the implementation
#             - Discuss the outcome
#             - The generated data
#         - Describe your results
#             - How do they match
#             - How do they differ
#             - Why they do
#         - Describe any problems/pitfalls you encountered
#             - How did you overcome them
#             - What were your assumptions/justifications for this solution

# ## Implementation
# 
# You will be replicating figures 3, 4, and 5 (Check Erratum at the end of paper and [Ch. 7 Sutton Textbook](http://incompleteideas.net/book/ebook/node73.html)) 

# ### Setup

# #### Step 0: Setup MDP and Helper function to generate dataset

# In[ ]:


#0. Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


"""Details of the Random Walk Problem from Sutton's 1988 Paper.  

Starting state = {D}
Non-Terminal States = {B, C, D, E, F}
Terminal states = {A, G}

Transition Probability for non-terminal states: 1/2"""

#S - States
S = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

#S_non_terminal - non terminal
S_non_terminal = ['B', 'C', 'D', 'E', 'F']

#S_terminal - terminal
S_terminal=['A', 'G']

R = {'A': 0, 'G': 1}

#T - Transition Matrix
transition_prob = 0.5

T = np.zeros((len(S), len(S)))

#1. fill out matrix
T[S.index('B'), S.index('C')] = transition_prob
T[S.index('B'), S.index('A')] = transition_prob
T[S.index('C'), S.index('D')] = transition_prob
T[S.index('C'), S.index('B')] = transition_prob
T[S.index('D'), S.index('E')] = transition_prob
T[S.index('D'), S.index('C')] = transition_prob
T[S.index('E'), S.index('F')] = transition_prob
T[S.index('E'), S.index('D')] = transition_prob
T[S.index('F'), S.index('E')] = transition_prob
T[S.index('F'), S.index('G')] = transition_prob

def generate_episode(S, S_start, S_terminal, T):
    """Generate episodes of states, given transition matrix, T, states, S, and start (S_start) and terminal states (S_terminal).

    Parameters: 
    S (list): all states
    S_start (str): label of initial state
    S_terminal (list): terminal states 
    T (numpy array): Transition Matrix

    Returns:
    list: single episode.

   """
    episode = []

    current_state = S_start

    while True:
        
        if len(episode) > 1000:
            raise ValueError("Episode fails to terminate after {} timesteps.  Something's wrong with the code.".format(len(episode)))
        
        episode.append(current_state)
        
        if current_state in S_terminal:                                                     #terminal states
            return episode
        else: 
            current_state = np.random.choice(S, replace=True, p=T[S.index(current_state)])  #non-terminal states

def generate_dataset(S, S_start, S_terminal, T, n_episodes=10):
    """Generate datasets based on the Random Walk Problem from Sutton's 1988 Paper.  
    
    Starting state = {D}
    Non-Terminal States = {B, C, D, E, F}
    Terminal states = {A, G}
    
    Transition Probability for non-terminal states: 1/2

    Parameters:
    S (list): all states
    S_start (str): label of initial state
    S_terminal (list): terminal states 
    T (numpy array): Transition Matrix
    n_episodes (int): # episodes to generate

    Returns:
    list of lists: dataset with episodes.

   """
    
    #2. generate n_episodes
    return [generate_episode(S, S_start=S_start, S_terminal=S_terminal, T=T) for i in range(n_episodes)]


# ### [Figure 3] Experiment 1 - Widrow-Hoff Supervised Learning Procedure vs. TD Methods on Random Walk Problem

# #### Problem Statement

# #### Graph to reproduce:
# 
# <img src="img/Project1-Fig3.png" alt="Drawing" style="width: 500px;"/>
# 
# Given the **Random Walk Problem** as defined on the below MDP, calculate TD(Lambda), under lambdas 0, 0.1, 0.3, 0.5, 0.7, 0.9, and 1.  Plot RMS errors against these lambdas.
# 
# <img src="img/Project1-MDP.png" alt="Drawing" style="width: 500px;"/>

# #### Setup

# In[292]:


#0. Setup
#alpha - learning rate, fix alpha for figure 3
fig3_alpha = 0.01


# #### Step 1: Generate 100 datasets - 10 episodes froms specified MDP (above)

# In[332]:


#1. Generate 100 training datasets of 10 episodes with given MP dynamics.                    
n_episodes = 10
n_datasets = 100

##TO FREEZE DATASET, A PREVIOUS RUN WAS STORED AND WILL BE READ FROM FILE, datasets.csv.

# datasets = [generate_dataset(S, S_start='D', S_terminal=['A', 'G'], T=T, n_episodes=n_episodes) for i in range(n_datasets)]
# pd.DataFrame(datasets).to_csv("datasets.csv",index=False)

import ast

datasets_saved = pd.read_csv('datasets.csv').to_numpy().tolist()
datasets_saved = [[ast.literal_eval(l2) for l2 in l1] for l1 in datasets_saved]
datasets = datasets_saved; datasets[0][0]


# #### Step 2: Start w/non-TD implementation of WH procedure to gain intuition => Confirm error ~ .25 (per figure)

# In[295]:


# # - RMSE of 100 datasets (ideal - predictions).
# verbose=True

# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())

# #P_ideal - Ideal Predictions for non terminal states
# P_ideal = [1/6, 1/3, 1/2, 2/3, 5/6]

# #P_rmse = rmse(P_pred, P_ideal), where P_pred - Predictions for non terminal states
# P_pred_rmse = np.mean(np.apply_along_axis(lambda P_pred: rmse(P_pred, P_ideal), 0, datasets_w))

# P_pred_mean = np.mean(datasets_w, axis=1)

# if verbose:
#     print('++++++++++++++Experiment 1 Results: [WH Procedure] +++++++++++++')
#     print('Ideal: ', P_ideal)
#     print('Preds: ', P_pred_mean)
#     print('RMSE: ',  P_pred_rmse)
    
# assert .15 <= P_pred_rmse <= .28, "Experiment 1 WH Comparison Fails w/ 'RMSE' of {}".format(P_pred_rmse)


# #### Step 3: Continue w/TD implementation of WH procedure [TD(1)] to confirm implementation => Confirm error ~.25 (per figure)

# In[299]:


#2. Calculate TD Lambda (with various lambdas) with updates per dataset until convergence (store predicted weights for 100 datasets).
verbose = False

#alpha - learning rate, fix for all figure simulations
alpha = fig3_alpha

#lambd = lambda, eligibility trace
lambd = 1

#datasets_w - weights matrix, non terminal state weights per dataset    
datasets_w = np.zeros((len(S_non_terminal), len(datasets)))

for idx, dataset in enumerate(datasets):
        
    #w - weight vector (of predicted return) of s_non_terminal
    w = np.zeros(len(S_non_terminal))

    iter_counter = 0  
    while True:                                                         # Iter over dataset - until convergence
        
        iter_counter += 1
        
        if iter_counter > 10000:
            raise ValueError("Learner fails to converge after {} timesteps.  Something's wrong with the code.".format(iter_counter))
        
        #w_delta - weight update term, sequence level
        w_delta = np.zeros(len(S_non_terminal))
    
        for seq in dataset:
            
            #eligibility - eligibility trace
            eligibility = np.zeros(len(S_non_terminal))

            #z - return
            z = R[seq[-1]]
            
            if verbose:
                print('seq', seq)
                print('z', z)

            for t, S_t in enumerate(seq[:-1]):

                #x_t - observation vector
                x_t = (np.array(S_non_terminal) == S_t).astype(int)
                
                S_t_1 = seq[t+1]
                
#                 if t == len(seq[:-1])-1:
#                     #x_t_1 - observation vector, x_(t + 1)
#                     x_t_1 = np.zeros(len(S_non_terminal))
#                 else:
                x_t_1 = (np.array(S_non_terminal) == S_t_1).astype(int)
                
                eligibility *= lambd
                eligibility += x_t
                
                if t == len(seq[:-1])-1:
                    #td_error - td error, when S_t is last non-terminal state
                    td_error = (z - np.dot(w, x_t))

                else:
                    #td_error - td error otherwise
                    td_error = (np.dot(w, x_t_1 - x_t))                    
                    
                #w_delta_t - weight update term, sequence level per S_t
                w_delta_t =  alpha * td_error * eligibility
                
                w_delta += w_delta_t
                if verbose:
#                     print('S_t', S_t)
                    print('x_t', x_t)  
                    print('TD Error', td_error)
                    print('w_delta_t', w_delta_t)

        w += w_delta 

        if np.max(np.abs(w_delta)) < 0.001: 
            print('converged at iter {0} w/w of {1}'.format(iter_counter,w))
            datasets_w[:, idx] = w
            break


# In[300]:


# - RMSE of 100 datasets (ideal - predictions).
verbose=True

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#P_ideal - Ideal Predictions for non terminal states
P_ideal = [1/6, 1/3, 1/2, 2/3, 5/6]

#P_rmse = rmse(P_pred, P_ideal), where P_pred - Predictions for non terminal states
P_pred_rmse = np.mean(np.apply_along_axis(lambda P_pred: rmse(P_pred, P_ideal), 0, datasets_w))

P_pred_mean = np.mean(datasets_w, axis=1)

if verbose:
    print('++++++++++++++Experiment 1 Results: [TD Implementation of WH Procedure] +++++++++++++')
    print('Ideal: ', P_ideal)
    print('Preds: ', P_pred_mean)
    print('RMSE: ',  P_pred_rmse)
    
assert .15 <= P_pred_rmse <= .28, "Experiment 1 TD-WH Comparison Fails w/ 'RMSE' of {}".format(P_pred_rmse)


# #### Step 4: Continue w/TD implementation w/lambdas in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1] => Confirm graph(per figure)

# In[334]:


#2. Calculate TD Lambda (with various lambdas) with updates per dataset until convergence (store predicted weights for 100 datasets).
verbose = False

#alpha - learning rate, fix for all figure simulations
alpha = fig3_alpha

#lambd = lambda, eligibility trace
lambdas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

experiment1_rmses = []

for lambd in lambdas:
    
    #datasets_w - weights matrix, non terminal state weights per dataset    
    datasets_w = np.zeros((len(S_non_terminal), len(datasets)))

    print('\nlambda=', lambd)
    for idx, dataset in enumerate(datasets):

        #w - weight vector (of predicted return) of s_non_terminal
        w = np.zeros(len(S_non_terminal))

        iter_counter = 0  
        while True:                                                         # Iter over dataset - until convergence

            iter_counter += 1

            if iter_counter > 10000:
                raise ValueError("Learner fails to converge after {} timesteps.  Something's wrong with the code.".format(iter_counter))

            #w_delta - weight update term, sequence level
            w_delta = np.zeros(len(S_non_terminal))

            for seq in dataset:

                #eligibility - eligibility trace
                eligibility = np.zeros(len(S_non_terminal))

                #z - return
                z = R[seq[-1]]

                if verbose:
                    print('seq', seq)
                    print('z', z)

                for t, S_t in enumerate(seq[:-1]):

                    #x_t - observation vector
                    x_t = (np.array(S_non_terminal) == S_t).astype(int)

                    S_t_1 = seq[t+1]
                    x_t_1 = (np.array(S_non_terminal) == S_t_1).astype(int)

                    eligibility *= lambd
                    eligibility += x_t

                    if t == len(seq[:-1])-1:
                        #td_error - td error, when S_t is last non-terminal state
                        td_error = (z - np.dot(w, x_t))

                    else:
                        #td_error - td error otherwise
                        td_error = (np.dot(w, x_t_1 - x_t))                    

                    #w_delta_t - weight update term, sequence level per S_t
                    w_delta_t =  alpha * td_error * eligibility

                    w_delta += w_delta_t
                    if verbose:
    #                     print('S_t', S_t)
                        print('x_t', x_t)  
                        print('TD Error', td_error)
                        print('w_delta_t', w_delta_t)

            w += w_delta 

            if np.max(np.abs(w_delta)) < 0.001: 
                print('converged at iter {0} w/w of {1}'.format(iter_counter,w))
                datasets_w[:, idx] = w
                break
                
    #P_ideal - Ideal Predictions for non terminal states
    P_ideal = [1/6, 1/3, 1/2, 2/3, 5/6]

    #P_rmse = rmse(P_pred, P_ideal), where P_pred - Predictions for non terminal states
    P_pred_rmse = np.mean(np.apply_along_axis(lambda P_pred: rmse(P_pred, P_ideal), 0, datasets_w))

    experiment1_rmses.append(P_pred_rmse)


# In[345]:


#3. Plot RMS error (root mean squared error) given ideal predictions in Sutton.

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.title('RMSE for TD Methods vs. Supervised WH Procedure')

plt.plot(lambdas, experiment1_rmses, '-ro')
fig.gca().set_xlabel(r'$\lambda$')
fig.gca().set_ylabel(r'$RMSE$')

ax.annotate('Widrow-Hoff', (lambdas[-1], experiment1_rmses[-1]))



fig.savefig('img/Project1-Fig3-REPRODUCED.png', dpi=fig.dpi)


# ### [Figure 4] Experiment 2 - TD Methods vs WH on learning rate

# #### Problem Statement

# "This experiment concerns the question of the **learning rate** when the training set is presented just once rather than repeatedly until convergence."
# 
# #### Graph to reproduce:
# 
# <img src="img/Project1-Fig4.png" alt="Drawing" style="width: 500px;"/>

# #### Setup

# In[346]:


#alpha - learning rate, fix alpha for figure 4
fig4_alphas = np.append(0.01, np.arange(0.00, .65, 0.05))


# In[347]:


#1. Calculate TD Lambda (with various lambdas and alphas) with updates per sequence, single presentation (no convergence) (store mean for 100 datasets).
verbose = False

#lambd = lambda, eligibility trace
lambdas = [0, 0.3, 0.8, 1]

experiment2a_rmses = []

for lambd in lambdas:
    
    lambda_rmses = []
    
    for alpha in fig4_alphas:
    
        print('\nlambda=', lambd)
        print('\nalpha=', alpha)
        
        #datasets_w - weights matrix, non terminal state weights per dataset    
        datasets_w = np.zeros((len(S_non_terminal), len(datasets)))

        for idx, dataset in enumerate(datasets):

            #w - weight vector (of predicted return) of s_non_terminal, SET TO 0.5 TO PREVENT BIAS TO LEFT OR RIGHT TEMRINATIONS
            w = np.full(len(S_non_terminal), 0.5)

            for seq in dataset:
                
                #w_delta - weight update term, sequence level
                w_delta = np.zeros(len(S_non_terminal))

                #eligibility - eligibility trace
                eligibility = np.zeros(len(S_non_terminal))

                #z - return
                z = R[seq[-1]]

                if verbose:
                    print('seq', seq)
                    print('z', z)

                for t, S_t in enumerate(seq[:-1]):

                    #x_t - observation vector
                    x_t = (np.array(S_non_terminal) == S_t).astype(int)

                    S_t_1 = seq[t+1]
                    x_t_1 = (np.array(S_non_terminal) == S_t_1).astype(int)
                    
                    eligibility *= lambd
                    eligibility += x_t

                    if t == len(seq[:-1])-1:
                        #td_error - td error, when S_t is last non-terminal state
                        td_error = (z - np.dot(w, x_t))

                    else:
                        #td_error - td error otherwise
                        td_error = (np.dot(w, x_t_1 - x_t))                    

                    #w_delta_t - weight update term, sequence level per S_t
                    w_delta_t =  alpha * td_error * eligibility

                    w_delta += w_delta_t
                    if verbose:
    #                     print('S_t', S_t)
#                         print('x_t', x_t)  
                        print('TD Error', td_error)
#                         print('w_delta_t', w_delta_t)

                #weight updates performed per sequence
                w += w_delta 
        
            #store w
            datasets_w[:, idx] = w

        #P_ideal - Ideal Predictions for non terminal states
        P_ideal = [1/6, 1/3, 1/2, 2/3, 5/6]

        #P_rmse = rmse(P_pred, P_ideal), where P_pred - Predictions for non terminal states
        P_pred_rmse = np.mean(np.apply_along_axis(lambda P_pred: rmse(P_pred, P_ideal), 0, datasets_w))
        
        print('RMSE of {}'.format(P_pred_rmse))

        lambda_rmses.append(P_pred_rmse)

    experiment2a_rmses.append(lambda_rmses)


# In[374]:


#3. Plot RMS error (root mean squared error) given ideal predictions in Sutton.
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.title('RMSE for TD Methods vs. Supervised WH Procedure w/Different Alphas')

for i, vec in enumerate(np.array(experiment2a_rmses)):
    plot_cutoffs = [.55, .6, .6, .45]
    mask = fig4_alphas < plot_cutoffs[i]
    ax.plot(fig4_alphas[mask], vec[mask], label=lambdas[i], marker='o')
    
fig.gca().legend(title=r'$\lambda$')
fig.gca().set_xlabel(r'$\alpha$')
fig.gca().set_ylabel(r'$RMSE$')

ax.set_ylim([0.0, 0.7])

fig.savefig('img/Project1-Fig4-REPRODUCED.png', dpi=fig.dpi)


# ### [Figure 5] Experiment 2 Part 2 - TD Methods vs WH on learning rate, BEST alpha

# #### Problem Statement

#  "This experiment concerns the question of the **learning rate** when the training set is presented just once rather than repeatedly until convergence."
# 
# #### Graph to reproduce:
# 
# <img src="img/Project1-Fig5.png" alt="Drawing" style="width: 500px;"/>

# #### Setup

# In[306]:


#alpha - learning rate, fix alpha for figure 4
fig4_alphas = np.append(0.01, np.arange(0.05, .65, 0.05))


# In[375]:


#1. Calculate TD Lambda (with various lambdas and alphas) with updates per sequence, single presentation (no convergence) (store mean for 100 datasets).
verbose = False

#lambd = lambda, eligibility trace
lambdas = np.append(0.01, np.arange(0.1, 1.1, .1))

experiment2b_rmses = np.zeros((len(lambdas), len(fig4_alphas)))

for lambd in lambdas:
    
    for alpha in fig4_alphas:
    
        print('\nlambda=', lambd)
        print('\nalpha=', alpha)
        
        #datasets_w - weights matrix, non terminal state weights per dataset    
        datasets_w = np.zeros((len(S_non_terminal), len(datasets)))

        for idx, dataset in enumerate(datasets):

            #w - weight vector (of predicted return) of s_non_terminal, SET TO 0.5 TO PREVENT BIAS TO LEFT OR RIGHT TEMRINATIONS
            w = np.full(len(S_non_terminal), 0.5)


            for seq in dataset:
                
                #w_delta - weight update term, sequence level
                w_delta = np.zeros(len(S_non_terminal))

                #eligibility - eligibility trace
                eligibility = np.zeros(len(S_non_terminal))

                #z - return
                z = R[seq[-1]]

                if verbose:
                    print('seq', seq)
                    print('z', z)

                for t, S_t in enumerate(seq[:-1]):

                    #x_t - observation vector
                    x_t = (np.array(S_non_terminal) == S_t).astype(int)

                    S_t_1 = seq[t+1]
                    x_t_1 = (np.array(S_non_terminal) == S_t_1).astype(int)

                    eligibility *= lambd
                    eligibility += x_t

                    if t == len(seq[:-1])-1:
                        #td_error - td error, when S_t is last non-terminal state
                        td_error = (z - np.dot(w, x_t))

                    else:
                        #td_error - td error otherwise
                        td_error = (np.dot(w, x_t_1 - x_t))                    

                    #w_delta_t - weight update term, sequence level per S_t
                    w_delta_t =  alpha * td_error * eligibility

                    w_delta += w_delta_t
                    if verbose:
    #                     print('S_t', S_t)
#                         print('x_t', x_t)  
                        print('TD Error', td_error)
#                         print('w_delta_t', w_delta_t)

                #weight updates performed per sequence
                w += w_delta 
        
            #store w
            datasets_w[:, idx] = w

        #P_ideal - Ideal Predictions for non terminal states
        P_ideal = [1/6, 1/3, 1/2, 2/3, 5/6]

        #P_rmse = rmse(P_pred, P_ideal), where P_pred - Predictions for non terminal states
        P_pred_rmse = np.mean(np.apply_along_axis(lambda P_pred: rmse(P_pred, P_ideal), 0, datasets_w))
        
        print('RMSE of {}'.format(P_pred_rmse))

        experiment2b_rmses[list(lambdas).index(lambd), list(fig4_alphas).index(alpha)] = P_pred_rmse


# In[379]:


# - Choose "best alpha" (minimal RMSE)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.title('RMSE (at best alpha) for TD Methods vs. Supervised WH Procedure')

best_alphas = np.min(np.array(experiment2b_rmses), axis=1)
ax.plot(lambdas, best_alphas, marker='o')
ax.annotate('Widrow-Hoff', (lambdas[-1], best_alphas[-1]))

fig.gca().set_xlabel(r'$\lambda$')
fig.gca().set_ylabel(r'$RMSE$ (at best $\alpha$)')

fig.savefig('img/Project1-Fig5-REPRODUCED.png', dpi=fig.dpi)