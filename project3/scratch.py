#1. Install OpenAI Gym + Setup Env
import numpy as np
import math

#%% md

## Step 0: Soccer Env Setup

#%%

import sys
sys.path.insert(0, '/Users/cesleemontgomery/masters/cs7642/projects/7642Fall2019cmontgomery38/project3')
from soccer import SoccerEnv

#%%

env = SoccerEnv()

# n_episodes_MAX = 8*10**5 #10*10**5
# steps_MAX = 100
# verbose = False
# alpha = 0.01
#
# num_states = env.observation_space.n
# num_actions = env.action_space.n
# num_individual_actions = int(
#     math.sqrt(env.action_space.n))  # individual player action space is 5, joint action space is 25
#
num_states = env.observation_space.n
num_actions = env.action_space.n
num_individual_actions = int(
    math.sqrt(env.action_space.n))  # individual player action space is 5, joint action space is 25


n_episodes_MAX = 2*10**5 #10*10**5
steps_MAX = 100
verbose = False
alpha = 0.1

# Instantiate Q-Learner
from QLearner import QLearningAgent

Foe_Q_agent = QLearningAgent(state_size=num_states,
                             action_size=num_individual_actions,
                             value_state_function_learner='minimax-q',
                             learning_rate=alpha)

# Ref state s, action 'S'
ref_state = SoccerEnv.encode_state(0, 2, 0, 1, 0)
ref_P1_action = int(SoccerEnv.Action.S)
ref_P2_action = int(SoccerEnv.Action.Stick)

# Q errors for plotting
Foe_Q_P1_Q_errors = []

for i_episode in range(n_episodes_MAX):
    state = env.reset()

    P1_Q_ref = Foe_Q_agent.Q[ref_state, ref_P1_action, ref_P2_action]

    for t in range(steps_MAX):
        joint_action = np.random.randint(num_actions)

        # Take action A, observe R, S'
        state_new, reward, done, info = env.step(joint_action)

        # Update Q
        P1_action, P2_action = env.decode_action(joint_action)
        P1_reward, P2_reward = env.decode_reward(state, reward)
        Foe_Q_agent.learn(P1_reward, state, state_new, P1_action, P2_action)

        state = state_new

        if done:
            # if verbose:
            #     print("Episode finished after {} timesteps".format(t + 1))
            break

    # calc error at end of episode update
    Foe_Q_P1_Q_errors.append(np.abs(Foe_Q_agent.Q[ref_state, ref_P1_action, ref_P2_action] - P1_Q_ref))

    # np.save('./models/Foe_Q', Foe_Q_agent.Q)
    # env.close()
#
# # Instantiate Q-Learner
# from QLearner import solve_corr_Q
# from QLearner import QLearningAgent
#
# CorrQ_agent_P1 = QLearningAgent(state_size=num_states,
#                                 action_size=num_individual_actions,
#                                 value_state_function_learner='max-sum-q',
#                                 learning_rate=alpha)
#
# CorrQ_agent_P2 = QLearningAgent(state_size=num_states,
#                                 action_size=num_individual_actions,
#                                 value_state_function_learner='max-sum-q',
#                                 learning_rate=alpha)
#
# # Ref state s, action 'S'
# ref_state = SoccerEnv.encode_state(0, 2, 0, 1, 0)
# ref_P1_action = int(SoccerEnv.Action.S)
# ref_P2_action = int(SoccerEnv.Action.Stick)
#
# # Q errors for plotting
# CorrQ_P1_Q_errors = []
#
# for i_episode in range(n_episodes_MAX):
#     state = env.reset()
#
#     P1_Q_ref = CorrQ_agent_P1.Q[ref_state, ref_P1_action, ref_P2_action]
#
#     for t in range(steps_MAX):
#         joint_action = np.random.randint(num_actions)
#
#         # Take action A, observe R, S'
#         state_new, reward, done, info = env.step(joint_action)
#
#         # Update Q
#         P1_action, P2_action = env.decode_action(joint_action)
#
#         ceq_V_P1, ceq_V_P2 = solve_corr_Q(CorrQ_agent_P1.Q, CorrQ_agent_P2.Q, state)
#
#         CorrQ_agent_P1.learn(reward, state, state_new, P1_action, P2_action, ceq_V_P1)
#         CorrQ_agent_P2.learn(reward, state, state_new, P1_action, P2_action, ceq_V_P2)
#
#         state = state_new
#
#         if done:
#             if verbose:
#                 print("Episode finished after {} timesteps".format(t + 1))
#             break
#
#     # calc error at end of episode update
#     CorrQ_P1_Q_errors.append(np.abs(CorrQ_agent_P1.Q[ref_state, ref_P1_action, ref_P2_action] - P1_Q_ref))
#
#     np.save('./models/Corr_Q', CorrQ_agent_P1.Q)
#     env.close()

# num_states = env.observation_space.n
# num_actions = env.action_space.n
# num_individual_actions = int(
#     math.sqrt(env.action_space.n))  # individual player action space is 5, joint action space is 25
#
# # Instantiate Q-Learner
# from QLearner import QLearningAgent
#
# Foe_Q_agent = QLearningAgent(state_size=num_states,
#                              action_size=num_individual_actions,
#                              value_state_function_learner='minimax-q',
#                              learning_rate=alpha)
#
# # Ref state s, action 'S'
# ref_state = SoccerEnv.encode_state(0, 2, 0, 1, 0)
# ref_P1_action = int(SoccerEnv.Action.S)
# ref_P2_action = int(SoccerEnv.Action.Stick)
#
# # Q errors for plotting
# Foe_Q_P1_Q_errors = []
#
# for i_episode in range(n_episodes_MAX):
#     state = env.reset()
#
#     P1_Q_ref = Foe_Q_agent.Q[ref_state, ref_P1_action, ref_P2_action]
#
#     for t in range(steps_MAX):
#         joint_action = np.random.randint(num_actions)
#
#         # Take action A, observe R, S'
#         state_new, reward, done, info = env.step(joint_action)
#
#         # Update Q
#         P1_action, P2_action = env.decode_action(joint_action)
#         Foe_Q_agent.learn(reward, state, state_new, P1_action, P2_action)
#
#         state = state_new
#
#         if done:
#             if verbose:
#                 print("Episode finished after {} timesteps".format(t + 1))
#             break
#
#     # calc error at end of episode update
#     Foe_Q_P1_Q_errors.append(np.abs(Foe_Q_agent.Q[ref_state, ref_P1_action, ref_P2_action] - P1_Q_ref))
#
#     np.save('./models/Foe_Q', Foe_Q_agent.Q)
#     env.close()
#
#
#
#     n = 5
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 print(i * n, (i + 1) * n)