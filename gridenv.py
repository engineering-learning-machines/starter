#!/usr/bin/env python
import sys
import numpy as np
import logging
# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
MAP_SIZE = np.array((3, 4))
SYMBOL_MAP = {'Empty': 0, 'Obstacle': 1, 'Exit': 2, 'Agent': 3}
OBJECT_SYMBOLS = {0: '.', 1: '*', 2: 'x', 3: 'o'}
EPISODE_COUNT = 1
EPISODE_MAX_STEP_COUNT = 100
ACTION_NAMES = {0: 'move up', 1: 'move right', 2: 'move down', 3: 'move left'}
# The actions can be conveniently represented as vectors when we calculate
# the next state
ACTION_VECTORS = np.array(((-1, 0), (0, 1), (1, 0), (0, -1)))
EXIT_POSITION = np.array((2, 1))
INITIAL_AGENT_POSITION = np.array((0, 0))
FIXED_OBSTACLES = np.array(((2, 0), (1, 2), (1, 0)))
# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
log = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
log.addHandler(log_handler)
log.setLevel(logging.DEBUG)


class FixedGridWorldEnvironment:
    def __init__(self, size, initial_agent_position, exit_position, fixed_obstacles):
        # We need to track where the agent is, in order to calculate the next state from the taken action
        self.exit_position = exit_position
        self.map_size = size
        self.__env_array__ = self.create_fixed_environment(size, initial_agent_position, exit_position, fixed_obstacles)

    @staticmethod
    def create_fixed_environment(size, initial_agent_position, exit_position, fixed_obstacles):
        # TODO: Find a better way to do this with index arrays
        env_array = np.zeros(size)
        # Initial agent location on the map
        env_array[initial_agent_position[0], initial_agent_position[1]] = SYMBOL_MAP['Agent']
        for obstacle in fixed_obstacles:
            env_array[obstacle[0], obstacle[1]] = SYMBOL_MAP['Obstacle']
        env_array[exit_position[0], exit_position[1]] = SYMBOL_MAP['Exit']
        return env_array

    @staticmethod
    def euclidean_distance(x, y):
        return ((x-y)**2).sum()

    def state_reachable(self, state, next_state):
        # We can reach only neighboring states
        if self.euclidean_distance(np.array(state), np.array(next_state)) != 1:
            return False
        if next_state[0] < 0 or next_state[1] < 0:
            return False
        try:
            point = self.__env_array__[next_state[0], next_state[1]]
            if point == 1:
                return False
        except IndexError:
            return False

        return True

    def next_state(self, state, action):
        target_state = state + ACTION_VECTORS[action]
        log.debug('Targeting: {}'.format(target_state))
        log.debug('State reachable: {}'.format(self.state_reachable(state, target_state)))
        if self.state_reachable(state, target_state):
            return target_state
        return state

    def update_agent_position(self, old_agent_position, agent_position):
        self.__env_array__[old_agent_position[0], old_agent_position[1]] = SYMBOL_MAP['Empty']
        self.__env_array__[agent_position[0], agent_position[1]] = SYMBOL_MAP['Agent']

    def calculate_reward(self, agent_position):
        if agent_position[0] == self.exit_position[0] and agent_position[1] == self.exit_position[1]:
            return 10
        return -1


class Agent:
    def __init__(self, map_size, action_space_size):
        # Create a simple tabular policy which maps each state (position on the map) to a
        # probability distribution of actions.
        random_tensor = np.random.random(size=(map_size[0], map_size[1], action_space_size))
        # Normalize the action distribution for each state individually.
        prob_sum_inv = 1. / random_tensor.sum(axis=2)
        # Each element of the action distribution for the particular state is divided by the sum of all elements for
        # that action distribution:
        self.policy = random_tensor * prob_sum_inv[:, :, None]
        self.value = np.zeros((map_size[0], map_size[1]))

    def act(self, pos):
        """
        Select an action according to the policy
        :return:
        """
        # Sample the action with the highest probability (greedy choice, no exploration)
        action_distribution = self.policy[pos[0], pos[1]]
        log.debug('Action distribution: {}'.format(action_distribution))
        # We use the index in the action distribution for the action:
        # return np.where(action_distribution == action_distribution.max())[0][0]
        return np.random.choice(4)


def render_environment(ar):
    """
    Show the environment on the console
    :param ar:
    :return:
    """
    for row in range(MAP_SIZE[0]):
        str_row = ''
        for column in range(MAP_SIZE[1]):
            str_row += ' ' + OBJECT_SYMBOLS[ar[row, column]] + ' '
        print(str_row)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':

    for episode_id in range(EPISODE_COUNT):

        # Make this deterministic
        np.random.seed(0)

        log.info('========== Running episode {:0>3} =========='.format(episode_id))

        # Initialize the agent with the map size and the number of actions, so it can create the initial random policy
        agent = Agent(MAP_SIZE, 4)
        # Create the environment
        env = FixedGridWorldEnvironment(MAP_SIZE, INITIAL_AGENT_POSITION, EXIT_POSITION, FIXED_OBSTACLES)
        last_state = INITIAL_AGENT_POSITION
        state = INITIAL_AGENT_POSITION

        for t in range(EPISODE_MAX_STEP_COUNT):
            log.debug('--- Step {:0>3} ---'.format(t))
            render_environment(env.__env_array__)

            log.debug('Current state: {}'.format(state))
            action = agent.act(state)
            log.debug('Agent takes action: {}'.format(ACTION_NAMES[action]))

            # The last state is needed to update the map
            last_state = state
            # Sample the next state (if stochastic / calculate if deterministic)
            state = env.next_state(state, action)
            log.debug('Environment samples state: {}'.format(state))

            # Reward
            reward = env.calculate_reward(state)
            log.debug('Reward: {}'.format(reward))
            log.debug('Next state: {}'.format(state))

            # Display the environment for debugging purposes
            env.update_agent_position(last_state, state)
            render_environment(env.__env_array__)

            if state[0] == EXIT_POSITION[0] and state[1] == EXIT_POSITION[1]:
                break
