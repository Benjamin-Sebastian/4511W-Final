"""
This file contains implementation of all the agents.
"""

from abc import ABC, abstractmethod
from util import *
import random
from game import CHECKERS_FEATURE_COUNT, checkers_features, checkers_reward
import numpy as np
from collections import defaultdict
import copy


class Agent(ABC):

    def __init__(self, is_learning_agent=False):
        self.is_learning_agent = is_learning_agent
        self.has_been_learning_agent = is_learning_agent
        self.num_states = 0


    @abstractmethod
    def get_action(self, state):
        """
        state: the state in which to take action
        Returns: the single action to take in this state
        """
        pass
    def get_num_states(self):
        """
        state: the state in which to take action
        Returns: the single action to take in this state
        """
        return self.num_states
    @abstractmethod
    def get_name(self):
        """
        state: the state in which to take action
        Returns: the single action to take in this state
        """
        pass

class AlphaBetaAgent(Agent):

    def __init__(self, depth):
        Agent.__init__(self, is_learning_agent=False)
        self.depth = depth
    def get_name(self):
         return "AlphaBetaAgent"
    def evaluation_function(self, state, agent=True):
        self.num_states+=1
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

class MiniMaxAgent(Agent):

    def __init__(self, depth):
        Agent.__init__(self, is_learning_agent=False)
        self.depth = depth

    def get_name(self):
        return "MiniMaxAgent"

    def evaluation_function(self, state, agent=True):
        self.num_states+=1
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

        def mini_max(state, depth, agent):
            if agent >= state.get_num_agents():
                agent = 0

            depth += 1
            if depth == self.depth or state.is_game_over():
                return [None, self.evaluation_function(state, max_agent)]
            elif agent == 0:
                return maximum(state, depth, agent)
            else:
                return minimum(state, depth, agent)

        def maximum(state, depth, agent):
            output = [None, -float("inf")]
            actions_list = state.get_legal_actions()

            if not actions_list:
                return [None, self.evaluation_function(state, max_agent)]

            for action in actions_list:
                current = state.generate_successor(action)
                val = mini_max(current, depth, agent + 1)

                check = val[1]

                if check > output[1]:
                    output = [action, check]

            return output

        def minimum(state, depth, agent):
            output = [None, float("inf")]
            actions_list = state.get_legal_actions()

            if not actions_list:
                return [None, self.evaluation_function(state, max_agent)]

            for action in actions_list:
                current = state.generate_successor(action)
                val = mini_max(current, depth, agent+1)

                check = val[1]

                if check < output[1]:
                    output = [action, check]

            return output

        # max_agent is true meaning it is the turn of first player at the state in
        # which to choose the action
        max_agent = state.is_first_agent_turn()
        output = mini_max(state, -1, 0)
        return output[0]

class RandomAgent(Agent):
    def __init__(self):
        Agent.__init__(self, is_learning_agent=False)

    def get_name(self):
        return "RandomAgent"

    def get_action(self, state):
        legal_actions = state.get_legal_actions()
        if legal_actions:
            self.num_states += 1


        return None if not legal_actions else random.choice(legal_actions)



class IterativeDeepeningSearchAgent(Agent):
    def __init__(self, max_depth):
        Agent.__init__(self, is_learning_agent=False)
        self.max_depth = max_depth

    def get_name(self):
        return "IterativeDeepeningSearchAgent"

    def evaluation_function(self, state, agent=True):
        self.num_states+=1
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
        def dfs(state, depth, agent, max_agent):
            if depth == self.max_depth or state.is_game_over():
                return self.evaluation_function(state, max_agent)

            if agent == max_agent:
                best_score = -float("inf")
                actions_list = state.get_legal_actions()
                for action in actions_list:
                    successor = state.generate_successor(action)
                    score = dfs(successor, depth + 1, 1 - agent, max_agent)
                    best_score = max(best_score, score)
                return best_score
            else:
                best_score = float("inf")
                actions_list = state.get_legal_actions()
                for action in actions_list:
                    successor = state.generate_successor(action)
                    score = dfs(successor, depth + 1, 1 - agent, max_agent)
                    best_score = min(best_score, score)
                return best_score

        max_agent = state.is_first_agent_turn()
        best_action = None
        best_score = -float("inf") if max_agent else float("inf")

        for depth in range(1, self.max_depth + 1):
            actions_list = state.get_legal_actions()
            for action in actions_list:
                successor = state.generate_successor(action)
                score = dfs(successor, 1, 1 - max_agent, max_agent)
                if (max_agent and score > best_score) or (not max_agent and score < best_score):
                    best_action = action
                    best_score = score

        return best_action