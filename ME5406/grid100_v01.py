import logging
import random
import gym
import time

import numpy as np
from gym.envs.classic_control import rendering

logger = logging.getLogger(__name__)

class Grid100Class(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    def __init__(self):
        # Set the gride world
        self.gridWidth = 10
        self.gridHeight = 10
        self.frisbee_state = [9]
        self.holes_state = [4,11,17,19,22,24,28,30,37,42,45,48,
                            51,52,54,66,68,70,73,75,82,87,94,97,99]
        self.init_state = 90
        self.init_random = True
        # | 90| 91| 92| 93| 94| 95| 96| 97| 98| 99|
        #   :                                   :
        #   :                                   :
        # | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

        # | A |   |   |   | 0 |   |   |   |   | 0 |
        # |   |   | 0 |   |   |   |   | 0 |   |   |
        # | 0 |   |   | 0 |   | 0 |   |   |   |   |
        # |   |   |   |   |   |   | 0 |   | 0 |   |
        # |   | 0 | 0 |   | 0 |   |   |   |   |   |
        # |   |   | 0 |   |   | 0 |   |   | 0 |   |
        # | 0 |   |   |   |   |   |   | 0 |   |   |
        # |   |   | 0 | 0 | 0 |   |   |   | 0 |   |
        # |   | 0 |   |   |   |   |   | 0 |   | 0 |
        # |   |   |   |   | 0 |   |   |   |   | x |

        # Environment configuration
        self.gridNum = self.gridWidth * self.gridHeight
        self.states = range(self.gridNum)
        # Bound configuration
        #   get the boundary state using grid confiuration
        self.upBound = range(self.gridNum - self.gridWidth, self.gridNum)
        self.rightBound = []
        for i in range(self.gridHeight):
            self.rightBound.append((self.gridWidth - 1) + i * self.gridWidth)
        self.downBound = range(self.gridWidth)
        self.leftBound = []
        for i in range(self.gridHeight):
            self.leftBound.append(0 + i * self.gridWidth)

        # Action space
        #   creat action dictionary, save the action and the change value of the state
        gridWidth = self.gridWidth
        self.action_space = 4
        self.actions = ['up', 'right', 'down', 'left']
        self.actionsDic = {'up': gridWidth, 'right': 1, 'down': -gridWidth, 'left': -1}

        # Reward situation
        #   save the reward for each action in array
        self.rewards = np.zeros(self.gridNum)
        for state in self.frisbee_state:
            self.rewards[state] = 1
        for state in self.holes_state:
            self.rewards[state] = -1

        # Screen Configuration
        self.screen_width = (self.gridWidth + 2) * 100
        self.screen_height = (self.gridHeight + 2) * 100

        # Agent information
        self.gamma = 0.8
        self.state = None
        # Possible action list
        #   save the action value Q in self.action_return element
        # self.action_return = []
        # for state in self.states:
        #     self.action_return.append({'up': 0, 'right': 0, 'down': 0, 'left': 0})
        # print(self.action_return)

        # Gym viewering
        self.viewer = None
        self.is_render = True
        self.is_sleep = False

    # Set state
    def setState(self, state):
        self.state = state

    # Set initial random
    def setRandomInitial(self, bool):
        self.init_random = bool

    # Set render
    def setRender(self, bool):
        self.is_render = bool

    # Set sleep
    def setSleep(self, bool):
        self.is_sleep = bool

    # step function
    # <input>
    #   action: agent's action as a string
    # <output>
    #   next_state
    #   reward
    #   is_terminal: if the environment terminal
    def step(self, action):
        # Get the envrionment state here
        #   if env is in the terminal state, return.
        state = self.state
        #   in case star in terminal state or didnot break out
        if (state in self.frisbee_state) or (state in self.holes_state):
            return state, 0, True, {}

        # State transfer
        #   transfter the state according to action;
        #   if action out of the bound, then set the state back.
        next_state =  state + self.actionsDic[action]
        if action == 'up':
            if state in self.upBound:
                next_state = state
        elif action == 'right':
            if state in self.rightBound:
                next_state = state
        elif action == 'down':
            if state in self.downBound:
                next_state = state
        elif action == 'left':
            if state in self.leftBound:
                next_state = state
        self.state = next_state

        # Reward
        reward = self.rewards[next_state]

        # Terminal state
        is_terminal = False
        if (next_state in self.frisbee_state) or (next_state in self.holes_state):
            is_terminal = True

        # Render
        # if self.is_render:
        #     self.render()
        #     if self.is_sleep:
        #         time.sleep(.1)

        # print('next_state', next_state)
        # print('reward', reward)
        # print('is_terminal', is_terminal)
        return next_state, reward, is_terminal, {}

    def reset(self):
        # Reset the environment
        #   if self.init_random is True, selet initial state until not in terminal state
        #   if self.init_random is False, set initial state as self.init_state
        if self.init_random:
            while True:
                self.state = np.random.choice(self.states)
                if (self.state not in self.frisbee_state) or (self.state not in self.holes_state):
                    break
        else:
            self.state = self.init_state
        return self.state

    def render(self, mode='human'):
        # Initial render setting
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            # Create the gride
            self.lines = []
            for i in range(1, self.gridHeight+2):
                self.lines.append(rendering.Line((100,i*100),((self.gridWidth+1)*100,i*100)))
            for i in range(1, self.gridWidth+2):
                self.lines.append(rendering.Line((i*100,100),(i*100,(self.gridHeight+1)*100)))

            # Create frisbee
            self.frisbee = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(150 + 100 * (self.frisbee_state[0] % self.gridWidth),
                                                                150 + 100 * (self.frisbee_state[0] // self.gridWidth)))
            self.frisbee.add_attr(self.circletrans)

            # Create holes
            self.holes = []
            for i in range(len(self.holes_state)):
                state = self.holes_state[i]
                self.circletrans = rendering.Transform(translation=(150 + 100 * (state % self.gridWidth),
                                                                    150 + 100 * (state // self.gridWidth)))
                self.holes.append(rendering.make_circle(35))
                self.holes[i].add_attr(self.circletrans)

            # Create robot
            self.agent= rendering.make_circle(30)
            self.robotrans = rendering.Transform(translation=(150 + 100 * (self.init_state % self.gridWidth),
                                                                    150 + 100 * (self.init_state // self.gridWidth)))
            self.agent.add_attr(self.robotrans)


            # Set color and add to viewer
            # lines
            for line in self.lines:
                line.set_color(24/255, 24/255, 24/255)
                self.viewer.add_geom(line)

            # frisbee
            self.frisbee.set_color(230/255, 44/255, 44/255)
            self.viewer.add_geom(self.frisbee)

            # holes
            for hole in self.holes:
                hole.set_color(54/255, 54/255, 54/255)
                self.viewer.add_geom(hole)

            # agent
            self.agent.set_color(118/255, 238/255, 0/255)
            self.viewer.add_geom(self.agent)

        if self.state is None:
            return None

        # Move the robot
        self.robotrans.set_translation(150 + 100 * (self.state % self.gridWidth),
                                       150 + 100 * (self.state // self.gridWidth))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()