#!/usr/bin/env python
import sys
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.ticker as ticker
# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
SIMULATION_NAME = 'Simple Gridworld'
MAP_SIZE = np.array((3, 4))
SYMBOL_MAP = {'Empty': 0, 'Obstacle': 1, 'Exit': 2, 'Agent': 3}
OBJECT_SYMBOLS = {0: '.', 1: '*', 2: 'x', 3: 'o'}
# OBJECT_COLORS = {0: 'white', 1: 'gray', 2: 'green', 3: 'red'}
OBJECT_COLORS = {0: (1, 1, 1), 1: (0.85, 0.85, 0.85), 2: (0, 1, 0), 3: (1, 0, 0)}
EPISODE_COUNT = 1
EPISODE_MAX_STEP_COUNT = 1
ACTION_NAMES = {0: 'move up', 1: 'move right', 2: 'move down', 3: 'move left'}
# The actions can be conveniently represented as vectors when we calculate
# the next state
ACTION_VECTORS = np.array(((-1, 0), (0, 1), (1, 0), (0, -1)))
EXIT_POSITION = np.array((2, 1))
INITIAL_AGENT_POSITION = np.array((0, 0))
FIXED_OBSTACLES = np.array(((2, 0), (1, 2), (1, 0)))
# Visualization
TILE_LINE_WIDTH=0.5
# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
log = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
log.addHandler(log_handler)
log.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# Visuals
# ------------------------------------------------------------------------------
def render_environment_ascii(ar):
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


class Renderer:
    """
    Notes: Tile size is always (width=1, height=1)
    """
    def __init__(self, map_size, fig_width=8):
        self.fig_width = fig_width
        self.map_size = map_size
        # Scaling factor for the coordinate transform: tile coords -> text coords
        scale = 1. / self.map_size
        # Scale tile coordinates (environment array indices) to text coordinates (in the range [0,1])
        self.text_coordinates_scale_transform = np.array([
            [scale[0], 0],
            [0, scale[1]]
        ])
        # Text coordinates need to be switched around, since text is plotted in the standard (x,y) coordiante space,
        # with y=0 at the bottom of the plot.
        self.coordinate_switch_transform = np.array([
            [0, 1],
            [1, 0]
        ])
        # Center the text on the tile center and reverse the y coordinate direction
        self.text_coordinates_offset = np.array([0.5*scale[1], 0.5*scale[0]])

    def tile_to_text_coords(self, row, column):
        """
        Transform the grid world coordinates to text coordinates, so we can show text. The text is centered in the tile.
        :param row:
        :param column:
        :return:
        """
        grid_coords = np.array([self.map_size[0] - 1 - row, column])
        scaled_coords = np.dot(self.text_coordinates_scale_transform, grid_coords)
        return np.dot(self.coordinate_switch_transform, scaled_coords) + self.text_coordinates_offset

    def draw_tile(self, ax, row, column, object_code, text=None):
        verts = [
           # left, bottom
           (column, row+1),
           # left, top
           (column, row),
           # right, top
           (column+1, row),
           # right, bottom
           (column+1, row+1),
           # close loop (left, bottom)
           (column, row+1),
        ]
        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]
        path = Path(verts, codes)
        ax.add_patch(patches.PathPatch(path, facecolor='white', lw=TILE_LINE_WIDTH))
        patch = patches.PathPatch(path, facecolor=OBJECT_COLORS[object_code], lw=TILE_LINE_WIDTH)
        ax.add_patch(patch)

        if text is not None:
            coords = self.tile_to_text_coords(row, column)
            ax.text(
                coords[0],
                coords[1],
                text,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10,
                color='black',
                transform=ax.transAxes
            )

    @staticmethod
    def configure_axis_ticks(axis, set_lim, size):
        axis.set_major_locator(ticker.NullLocator())
        axis.set_major_formatter(ticker.NullFormatter())
        axis.set_minor_locator(ticker.FixedLocator(0.5 + np.arange(size)))
        axis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x-0.5))))
        set_lim(0, size)

    def render_matplotlib(self, ar):
        """
        Renders the gridworld in Matplotlib
        :param ar: The gridworld array
        :return:
        """
        fig_height = self.fig_width*float(self.map_size[0])/self.map_size[1]
        fig = plt.figure(figsize=(self.fig_width, fig_height))
        fig.canvas.set_window_title(SIMULATION_NAME)
        ax = fig.add_subplot(111)
        ax.grid()
        # Configure ticks
        self.configure_axis_ticks(ax.xaxis, ax.set_xlim, self.map_size[1])
        self.configure_axis_ticks(ax.yaxis, ax.set_ylim, self.map_size[0])
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        for row in range(self.map_size[0]):
            for column in range(self.map_size[1]):
                self.draw_tile(ax, row, column, ar[row, column], text='{0}{1}'.format(row, column))
        plt.show()

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':

    for episode_id in range(EPISODE_COUNT):

        # Visualize the environment
        renderer = Renderer(MAP_SIZE)

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
            render_environment_ascii(env.__env_array__)
            renderer.render_matplotlib(env.__env_array__)

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
            render_environment_ascii(env.__env_array__)

            if state[0] == EXIT_POSITION[0] and state[1] == EXIT_POSITION[1]:
                break
