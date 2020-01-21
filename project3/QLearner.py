import numpy as np
from cvxopt import matrix, solvers


def construct_rationality_constraints(Q_joint_actions):
    '''construct each rationality constraint, gather policies on LHS (G), RHS (h) = 0'''
    num_actions = Q_joint_actions.shape[0]
    G_temp = np.zeros((num_actions*(num_actions-1), num_actions**2))

    curr = 0
    for i in range(num_actions):
        for j in range(num_actions):
            if i != j:
                G_temp[curr, i * num_actions:(i + 1) * num_actions] = Q_joint_actions[i] - Q_joint_actions[j]

    return G_temp

def solve_corr_Q(Q1, Q2, state):
    num_actions = Q1[state].shape[0]**2
    # construct RC separately
    G_P1 = construct_rationality_constraints(Q1[state])
    G_P2 = construct_rationality_constraints(Q2[state].T) # transpose Q2 to make constructing rationality constraint easier

    # construct Probability Constraint #2: Policy Probs >=0
    G_pos_constraint = np.eye(num_actions)

    # hstack Gx <= h constraints, rcs moved to LHS so h = 0
    G = np.vstack([G_P1, G_P2, G_pos_constraint])
    h = np.zeros(G.shape[0])

    # construct Probability Constraint #1: Sum of Policy Probs = 1
    A = np.ones((1, num_actions))
    b = np.array([1])  # b - RHS of SoE (Equality)

    #construct cost function (max sums)
    c = (Q1[state] + Q2[state]).flatten() #.reshape(1,25)

    # Construct the LP, invoke solver
    solvers.options['show_progress'] = False
    solver = 'glpk'
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    sol = solvers.lp(c=matrix(-1 * c, tc='d'),  # CONFORM to CVXOPT: G => swap inequality AND C => flip min to max
                     G=matrix(-1 * G, tc='d'),
                     h=matrix(h, tc='d'),
                     A=matrix(A, tc='d'),
                     b=matrix(b, tc='d'), solver=solver)

    pi = sol['x']

    if pi is None:
        return 0, 0

    ceq_V_P1 = np.dot(Q1[state].flatten().reshape(1,25), pi)
    ceq_V_P2 = np.dot(Q2[state].flatten().reshape(1,25), pi) #TODO - may need to tranpose
    return ceq_V_P1, ceq_V_P2

class QLearningAgent():

    def __init__(self,
                 state_size,
                 action_size,
                 value_state_function_learner='max-q',
                 gamma=.9,
                 epsilon=1.0,
                 epsilon_min=.01,
                 epsilon_decay=.995,
                 learning_rate=1.0,
                 learning_rate_min=.001,
                 learning_rate_decay=.9999925,
                 ):
        """Initialize parameters and build Q-Learning agent."""

        self.state_size = state_size
        self.action_size = action_size
        self.value_state_function_learner = value_state_function_learner
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.learning_rate_decay = learning_rate_decay

        self.Q = self._build_model(self.state_size,
                                       self.action_size)

    def _build_model(self, state_size, action_size):
        """Initialize parameters and build model.
            Q Action Value Function Lookup Table of states, actions
         """
        #traditional Q Learning Q-Table
        if self.value_state_function_learner == 'max-q':
            return np.zeros((state_size, action_size))

        # Minimax Q Q-Table
        elif self.value_state_function_learner == 'minimax-q' or self.value_state_function_learner ==  'max-sum-q':
            return np.zeros((state_size, action_size, action_size))

    def sample_policy(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(self.Q.shape[1])
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def solve_minimax_Q(self, game_matrix):
        '''
        Computes the NE Value Function ('Primal Objective' with CVXOPT linear programming).

        Some pointers from: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

        Returns:
        float: estimate of Value fx.
        '''

        num_vars = game_matrix.shape[0]

        c = np.insert(np.zeros(num_vars), 0, 1)        #c - cost function
        # Gx >= h: {LHS >= V} = > {-V + LHS >= 0}
        G = -1 * np.insert(game_matrix, 0, np.ones(num_vars), axis=1)  #G - LHS of SoE (Inequality) (matrix form)
        G_pos_constraint = np.insert(np.eye(num_vars), 0, np.zeros(num_vars), axis=1)  # G (+ added constraints for >= 0)
        G = np.vstack([G, G_pos_constraint])

        h = np.zeros(num_vars * 2)                         #h - RHS of SoE (+ added constraints for >= 0)

        # Ax = b: [0, 1, 1, 1] = [1](probs=1, no V)
        A = np.insert(np.ones(num_vars), 0, 0).reshape(-1, 1).T       #A - LHS of SoE (Equality) (matrix form)
        b = np.array([1])                                   #b - RHS of SoE (Equality)

        # Construct the LP, invoke solver
        solvers.options['show_progress'] = False
        solver = 'glpk'
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
        sol = solvers.lp(c=matrix(-1*c, tc='d'),             #CONFORM to CVXOPT: G => swap inequality AND C => flip min to max
                                G=matrix(-1*G, tc='d'),
                                h=matrix(h, tc='d'),
                                A=matrix(A, tc='d'),
                                b=matrix(b, tc='d'), solver=solver)
        return sol['primal objective']

    def learn(self, reward, state, state_new, P1_action, P2_action=0, ceq_V = 0.0):

        if self.value_state_function_learner == 'max-q':            # ignore P2 actions, traditional Q Learner has no notion of other player
            # td_error, TD target - current state_value function estimate
            td_error = reward + (self.gamma * np.max(self.Q[state_new, :])) - self.Q[state, P1_action]

            # Update Q, State Value Fx
            self.Q[state, P1_action] += self.learning_rate * td_error

        elif self.value_state_function_learner == 'minimax-q':
            V = self.solve_minimax_Q(self.Q[state])
            # td_error, TD target - current state_value function estimate
            td_error = reward + (self.gamma * V)

            # Update Q, State Value Fx
            self.Q[state, P1_action, P2_action] += self.learning_rate * td_error

        elif self.value_state_function_learner == 'max-sum-q':
            # td_error, TD target - current state_value function estimate
            td_error = reward + (self.gamma * ceq_V)

            # Update Q, State Value Fx
            self.Q[state, P1_action, P2_action] += self.learning_rate * td_error

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # add alpha decay
        if self.learning_rate > self.learning_rate_min:
            self.learning_rate *= self.learning_rate_decay