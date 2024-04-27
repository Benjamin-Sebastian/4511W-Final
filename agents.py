"""
This file contains implementation of all the agents.
"""

from abc import ABC, abstractmethod
from util import *
import random
from game import CHECKERS_FEATURE_COUNT, checkers_features, checkers_reward
import numpy as np
from collections import defaultdict


class Agent(ABC):

    def __init__(self, is_learning_agent=False):
        self.is_learning_agent = is_learning_agent
        self.has_been_learning_agent = is_learning_agent

    @abstractmethod
    def get_action(self, state):
        """
        state: the state in which to take action
        Returns: the single action to take in this state
        """
        pass
    @abstractmethod
    def get_name(self):
        """
        state: the state in which to take action
        Returns: the single action to take in this state
        """
        pass

class KeyBoardAgent(Agent):

    def __init__(self):
        Agent.__init__(self)


    def get_action(self, state):
        """
        state: the current state from which to take action

        Returns: list of starting position, ending position
        """

        start = [int(pos) for pos in input("Enter start position (e.g. x y): ").split(" ")]
        end = [int(pos) for pos in input("Enter end position (e.g. x y): ").split(" ")]

        ends = []
        i=1
        while i < len(end):
            ends.append([end[i-1], end[i]])
            i += 2

        action = [start] + ends
        return action

class AlphaBetaAgent(Agent):

    def __init__(self, depth):
        Agent.__init__(self, is_learning_agent=False)
        self.depth = depth
    def get_name(self):
         return "AlphaBetaAgent"
    def evaluation_function(self, state, agent=True):
        """
        state: the state to evaluate
        agent: True if the evaluation function is in favor of the first agent and false if
               evaluation function is in favor of second agent

        Returns: the value of evaluation
        """
        agent_ind = 0 if agent else 1
        other_ind = 1 - agent_ind

        if state.is_game_over():
            if agent and state.is_first_agent_win():
                return 500

            if not agent and state.is_second_agent_win():
                return 500

            return -500

        pieces_and_kings = state.get_pieces_and_kings()
        return pieces_and_kings[agent_ind] + 2 * pieces_and_kings[agent_ind + 2] - \
        (pieces_and_kings[other_ind] + 2 * pieces_and_kings[other_ind + 2])

    def get_action(self, state):

        def mini_max(state, depth, agent, A, B):
            if agent >= state.get_num_agents():
                agent = 0

            depth += 1
            if depth == self.depth or state.is_game_over():
                return [None, self.evaluation_function(state, max_agent)]
            elif agent == 0:
                return maximum(state, depth, agent, A, B)
            else:
                return minimum(state, depth, agent, A, B)

        def maximum(state, depth, agent, A, B):
            output = [None, -float("inf")]
            actions_list = state.get_legal_actions()

            if not actions_list:
                return [None, self.evaluation_function(state, max_agent)]

            for action in actions_list:
                current = state.generate_successor(action)
                val = mini_max(current, depth, agent + 1, A, B)

                check = val[1]

                if check > output[1]:
                    output = [action, check]

                if check > B:
                    return [action, check]

                A = max(A, check)

            return output

        def minimum(state, depth, agent, A, B):
            output = [None, float("inf")]
            actions_list = state.get_legal_actions()

            if not actions_list:
                return [None, self.evaluation_function(state, max_agent)]

            for action in actions_list:
                current = state.generate_successor(action)
                val = mini_max(current, depth, agent+1, A, B)

                check = val[1]

                if check < output[1]:
                    output = [action, check]

                if check < A:
                    return [action, check]

                B = min(B, check)

            return output

        # max_agent is true meaning it is the turn of first player at the state in 
        # which to choose the action
        max_agent = state.is_first_agent_turn()
        output = mini_max(state, -1, 0, -float("inf"), float("inf"))
        return output[0]

class IterativeDeepeningAgent(Agent):

    def __init__(self, depth):
        Agent.__init__(self, is_learning_agent=False)
        self.depth = depth

    def get_name(self):
        return "IterativeDeepeningSearchAgent"

    def evaluation_function(self, state, agent=True):
        """
        state: the state to evaluate
        agent: True if the evaluation function is in favor of the first agent and False if
               the evaluation function is in favor of the second agent

        Returns: the value of evaluation
        """
        agent_ind = 0 if agent else 1
        other_ind = 1 - agent_ind

        if state.is_game_over():
            if agent and state.is_first_agent_win():
                return 500

            if not agent and state.is_second_agent_win():
                return 500

            return -500

        pieces_and_kings = state.get_pieces_and_kings()
        return pieces_and_kings[agent_ind] + 2 * pieces_and_kings[agent_ind + 2] - \
               (pieces_and_kings[other_ind] + 2 * pieces_and_kings[other_ind + 2])

    def get_action(self, state):
        max_agent = state.is_first_agent_turn()
        best_action = None

        for depth in range(1, self.depth + 1):
            best_action = self.minimax(state, depth, max_agent)
        return best_action

    def minimax(self, state, depth, max_agent):
        if depth == 0 or state.is_game_over():
            return None, self.evaluation_function(state, max_agent)

        actions = state.get_legal_actions()

        if max_agent:
            best_value = -float('inf')
            best_action = None
            for action in actions:
                successor_state = state.generate_successor(action)
                _, value = self.minimax(successor_state, depth - 1, False)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_action, best_value

        else:
            best_value = float('inf')
            best_action = None
            for action in actions:
                successor_state = state.generate_successor(action)
                _, value = self.minimax(successor_state, depth - 1, True)
                if value < best_value:
                    best_value = value
                    best_action = action
            return best_action, best_value

class ReinforcementLearningAgent(Agent):

    def __init__(self, is_learning_agent=True):
        Agent.__init__(self, is_learning_agent)

        self.episodes_so_far = 0


    @abstractmethod
    def get_action(self, state):
        """
        state: the current state from which to take action

        Returns: the action to perform
        """
        # TODO call do_action from this method
        pass


    @abstractmethod
    def update(self, state, action, next_state, reward):
        """
        performs update for the learning agent

        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        pass

    def start_episode(self):
        # Accumulate rewards while training for each episode and show total rewards 
        # at the end of each episode i.e. when stop episode
        self.prev_state = None
        self.prev_action = None

        self.episode_rewards = 0.0


    def stop_episode(self):
        # print('reward this episode', self.episode_rewards)
        pass

    @abstractmethod
    def start_learning(self):
        pass


    @abstractmethod
    def stop_learning(self):
        pass


    @abstractmethod
    def observe_transition(self, state, action, next_state, reward, next_action=None):
        pass


    @abstractmethod
    def observation_function(self, state):
        pass


    # TODO
    def reward_function(self, state, action, next_state):
        # make a reward function for the environment
        return checkers_reward(state, action, next_state)


    def do_action(self, state, action):
        """
        called by get_action to update previous state and action
        """
        self.prev_state = state
        self.prev_action = action

class QLearningAgent(ReinforcementLearningAgent):

    def __init__(self, alpha=0.01, gamma=0.1, epsilon=0.5, is_learning_agent=True, weights=None):

        """
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration constant
        is_learning_agent: whether to treat this agent as learning agent or not
        weights: default weights
        """

        ReinforcementLearningAgent.__init__(self, is_learning_agent=is_learning_agent)

        self.original_alpha = alpha
        self.original_epsilon = epsilon

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        if not is_learning_agent:
            self.epsilon = 0.0
            self.alpha = 0.0


        if weights is None:
            # initialize weights for the features
            self.weights = np.zeros(CHECKERS_FEATURE_COUNT)
        else:
            if len(weights) != CHECKERS_FEATURE_COUNT:
                raise Exception("Invalid weights " + weights)

            self.weights = np.array(weights, dtype=float)


    def start_learning(self):
        """
        called by environment to notify agent of starting new episode
        """

        self.alpha = self.original_alpha
        self.epsilon = self.original_epsilon

        self.is_learning_agent = True


    def stop_learning(self):
        """
        called by environment to notify agent about end of episode
        """
        self.alpha = 0.0
        self.epsilon = 0.0

        self.is_learning_agent = False


    def get_q_value(self, state, action, features):
        """
          Returns: Q(state,action)
        """
        q_value = np.dot(self.weights, features)
        return q_value


    def compute_value_from_q_values(self, state):
        """
          Returns: max_action Q(state, action) where the max is over legal actions.
                   If there are no legal actions, which is the case at the terminal state, 
                   return a value of 0.0.
        """
        actions = state.get_legal_actions()

        if not actions:
            return 0.0

        q_values = \
        [self.get_q_value(state, action, checkers_features(state, action)) for action in actions]

        return max(q_values)


    def compute_action_from_q_values(self, state, actions):
        """
          Returns: the best action to take in a state. If there are no legal actions,
                   which is the case at the terminal state, return None.
        """
        if not actions:
            return None

        # if max_value < 0:
        #     return random.choice(actions)

        arg_max = np.argmax([self.get_q_value(state, action, checkers_features(state, action)) 
            for action in actions])

        return actions[arg_max]


    def get_action(self, state):
        """
          Returns: the action to take in the current state.  With probability self.epsilon,
                   take a random action and take the best policy action otherwise.  If there are
                   no legal actions, which is the case at the terminal state, returns None.
        """

        # Pick Action
        legal_actions = state.get_legal_actions()
        action = None

        if not legal_actions:
            return None

        if flip_coin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.compute_action_from_q_values(state, legal_actions)

        self.do_action(state, action)
        return action


    def update(self, state, action, next_state, reward):

        features = checkers_features(state, action)

        expected = reward + self.gamma * self.compute_value_from_q_values(next_state)
        current = self.get_q_value(state, action, features)

        temporal_difference = expected - current

        for i in range(CHECKERS_FEATURE_COUNT):
            self.weights[i] = self.weights[i] + self.alpha * (temporal_difference) * features[i]


    def getPolicy(self, state):
        return self.compute_action_from_q_values(state, state.get_legal_actions())


    def getValue(self, state):
        return self.compute_value_from_q_values(state)  


    def observe_transition(self, state, action, next_state, reward, next_action=None):
        """
        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        self.episode_rewards += reward
        self.update(state, action, next_state, reward)


    def observation_function(self, state):
        if self.prev_state is not None:
            reward = self.reward_function(self.prev_state, self.prev_action, state)
            # print('reward is', reward)
            self.observe_transition(self.prev_state, self.prev_action, state, reward)

    def update_parameters(self, freq, num_games):
        if num_games % freq == 0:
            self.original_alpha /= 2.0
            self.original_epsilon /= 2.0

class SarsaLearningAgent(QLearningAgent):

    def __init__(self, alpha=0.01, gamma=0.1, epsilon=0.5, is_learning_agent=True, weights=None):
        
        QLearningAgent.__init__(self, alpha, gamma, epsilon, is_learning_agent, weights)
    def get_name(self):
        return "SarsaLearningAgent"

    def update(self, state, action, next_state, next_action, reward):

        features = checkers_features(state, action)

        if next_action is None:
            next_q_value = 0.0
        else:
            next_q_value = \
            self.get_q_value(next_state, next_action, checkers_features(next_state, next_action))
    
        expected = reward + self.gamma * next_q_value

        current = self.get_q_value(state, action, features)

        temporal_difference = expected - current

        for i in range(CHECKERS_FEATURE_COUNT):
            self.weights[i] = self.weights[i] + self.alpha * (temporal_difference) * features[i]


    def observe_transition(self, state, action, next_state, next_action, reward):
        """
        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        self.episode_rewards += reward
        self.update(state, action, next_state, next_action, reward)


    def observation_function(self, state):
        if self.prev_state is not None:
            reward = self.reward_function(self.prev_state, self.prev_action, state)
            # print('reward is', reward)
            action = self.get_action(state)
            self.observe_transition(self.prev_state, self.prev_action, state, action, reward)

            return action

class SarsaSoftmaxAgent(SarsaLearningAgent):

    def __init__(self, alpha=0.01, gamma=0.1, t=1.0, is_learning_agent=True, weights=None):
        SarsaLearningAgent.__init__(self, alpha=alpha, gamma=gamma,
            is_learning_agent=is_learning_agent, weights=weights)

        self.t = t

    def get_action(self, state):
        legal_actions = state.get_legal_actions()

        if not legal_actions:
            return None

        if self.epsilon == 0.0:
            return self.compute_action_from_q_values(state, legal_actions)

        q_values = [self.get_q_value(state, action, checkers_features(state, action))
                for action in legal_actions]

        exps = np.exp(q_values) / self.t
        probs = exps / np.sum(exps)

        action_ind = np.random.choice(len(legal_actions), p=probs)

        self.do_action(state, legal_actions[action_ind])
        return legal_actions[action_ind]

    def update_parameters(self, freq, num_games):
        if num_games % freq == 0:
            self.t /= 2.0
 

    def __init__(self, num_simulations=1000, exploration_weight=1.0, is_learning_agent=False):
        Agent.__init__(self, is_learning_agent)
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.Q = defaultdict(int)  # (state, action) -> total reward
        self.N = defaultdict(int)  # (state, action) -> visit count
        self.max_episode_length = 1000  # Maximum length of an episode

    def get_name(self):
        return "MonteCarloSearchAgent"

    def get_action(self, state):
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return None

        # Run simulations and update Q and N
        for _ in range(self.num_simulations):
            self.run_simulation(state)

        # Choose the action with the highest value
        action_values = [(self.Q[(state, action)] / self.N[(state, action)]) if self.N[(state, action)] > 0 else 0.0 for action in legal_actions]
        max_value = max(action_values)
        best_actions = [action for action, value in zip(legal_actions, action_values) if value == max_value]
        action = random.choice(best_actions)

        return action

    def run_simulation(self, state):
        state_copy = copy.deepcopy(state)
        episode_length = 0
        reward = 0.0

        while not state_copy.is_game_over() and episode_length < self.max_episode_length:
            legal_actions = state_copy.get_legal_actions()
            action_values = [(self.Q[(state_copy, action)] / self.N[(state_copy, action)]) if self.N[(state_copy, action)] > 0 else 0.0 for action in legal_actions]
            action_values = [value + (self.exploration_weight * random.random()) for value in action_values]
            action = legal_actions[random.randint(0, len(legal_actions) - 1)]

            next_state = state_copy.generate_successor(action)
            reward += self.reward_function(state_copy, action, next_state)
            state_copy = next_state
            episode_length += 1

        # Update Q and N
        for state_action in [(state_copy, action) for action in state_copy.get_legal_actions()]:
            self.N[state_action] += 1
            self.Q[state_action] += reward

    def reward_function(self, state, action, next_state):
        # Implement your reward function here
        # You can use features like the number of pieces, kings, etc.
        # For example:
        agent_ind = 0 if state.is_first_agent_turn() else 1
        other_ind = 1 - agent_ind
        pieces_and_kings = next_state.get_pieces_and_kings()
        reward = pieces_and_kings[agent_ind] + 2 * pieces_and_kings[agent_ind + 2] - (pieces_and_kings[other_ind] + 2 * pieces_and_kings[other_ind + 2])
        return reward