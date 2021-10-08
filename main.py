import time
import gym
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

# Unexpected warning filter
#   function has been proved in test.py
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# Monte Carlo Control
class MMC():
    def __init__(self, envName, max_episodes=2000, max_steps=20,
                 init_randomize=True, is_render=False, is_sleep=False,
                 gama=0.9, e=0.10):
        # Train parameters
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.init_randomize = init_randomize
        self.is_render = is_render
        self.is_sleep = is_sleep
        # Update parameters
        self.gama = gama
        self.e = e

        # Build the environment
        self.env = gym.make(envName)
        self.env.setRandomInitial(self.init_randomize)
        self.env.setRender(self.is_render)
        self.env.setSleep(self.is_sleep)

        # Create a dictionary to save the trajectory memory
        self.state_action_return = {}
        for state in self.env.states:
            for action in self.env.actionsDic.keys():
                self.state_action_return['%s_%s' % (state, action)] = []
        # Create a list to save the estimated action reward
        self.state_action_ereward = []
        for _ in self.env.states:
            self.state_action_ereward.append([0., 0., 0., 0.])

        # Creat record list for reward and
        self.accum_r = 0
        self.record_R = []
        # Create optimal choice and sub-optimal choice list
        self.record_Q1o = []
        self.record_Q1s = []
        self.record_Q2o = []
        self.record_Q2s = []
        self.record_Q3o = []
        self.record_Q3s = []

    # Reward Record
    def recordR(self):
        # Record the R
        self.record_R.append(self.accum_r)
        self.accum_r = 0

    # Q(s,a) Record
    def recordQ(self):
        # Record the Q
        if self.env.gridNum == 16:
            self.record_Q1o.append(self.state_action_ereward[13][1])    # 13_right
            self.record_Q1s.append(self.state_action_ereward[13][3])    # 13_left
            self.record_Q2o.append(self.state_action_ereward[14][2])    # 14_down
            self.record_Q2s.append(self.state_action_ereward[14][3])    # 14_left
            self.record_Q3o.append(self.state_action_ereward[10][2])    # 10_down
            self.record_Q3s.append(self.state_action_ereward[10][0])    # 10_up
        elif self.env.gridNum == 100:
            self.record_Q1o.append(self.state_action_ereward[80][1])    # 13_right
            self.record_Q1s.append(self.state_action_ereward[80][0])    # 13_up
            self.record_Q2o.append(self.state_action_ereward[62][1])    # 62_right
            self.record_Q2s.append(self.state_action_ereward[62][3])    # 62_left
            self.record_Q3o.append(self.state_action_ereward[44][2])    # 44_down
            self.record_Q3s.append(self.state_action_ereward[44][3])    # 44_left

    def run_simulation(self):
        # Repeat for max_episodes times
        for episode in range(1, self.max_episodes+1):
            G = 0
            state = self.env.reset()
            self.env.render()
            trajectory = []

            # Generate an episode following pi
            for step in range(self.max_steps):
                # e-greedy
                #   using random probability to choose greedy action
                if np.random.choice([True, False], p=[1 - self.e, self.e]):
                    alist = self.state_action_ereward[state]
                    # if never achieve this state, random choose the actiton.
                    if max(alist) == min(alist):
                        action = np.random.choice(self.env.actions)
                    else:
                        action = self.env.actions[alist.index(max(alist))]
                else:
                    # Choose action randomly when greedy
                    action = np.random.choice(self.env.actions)

                # Agent act
                next_state, reward, done, _ = self.env.step(action)

                # Record
                trajectory.append((state, action, reward))
                self.accum_r += reward

                # Move to next state
                state = next_state

                # Break if terminal
                if done:
                    break

            # Loop for each episode
            #   save the index in idx, the one step (state, action, reward) in step
            for idx, step in enumerate(trajectory[::-1]):
                # Monte Carlo reward
                G = self.gama * G + step[2]

                # Unless the pair S_t, A_t appears in S_0,A_0,...S_t-1, A_t-1(for each t)
                #   flatten the trajectory, see whether in the array
                #       compare s with every 3 element from 0 to the end(the state)
                #       compare a with every 3 element from 1 to the end(the action)
                if (step[0] not in np.array(trajectory[::-1][idx + 1:]).flatten()[0::3]) and \
                        (step[1] not in np.array(trajectory[::-1][idx + 1:]).flatten()[1::3]):
                    self.state_action_return['%s_%s' % (step[0], step[1])].append(G)
                    self.state_action_ereward[step[0]][self.env.actions.index(step[1])] = np.mean(
                        self.state_action_return['%s_%s' % (step[0], step[1])])

            # Record the state information every fixed step
            if not (episode % 100):
                self.recordR()
            if not (episode % 10):
                self.recordQ()

        self.env.close()


# SARSA
class SARSA():
    def __init__(self, envName, max_episodes=4000, max_steps=20,
                 init_randomize=True, is_render=True, is_sleep=False,
                 gama=0.9, e=0.10, lr=0.01):
        # Train parameters
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.init_randomize = init_randomize
        self.is_render = is_render
        self.is_sleep = is_sleep
        # Update parameters
        self.gama = gama
        self.e = e
        self.lr = lr

        # Build the environment
        self.env = gym.make(envName)
        self.env.setRandomInitial(self.init_randomize)
        self.env.setRender(self.is_render)
        self.env.setSleep(self.is_sleep)

        # Create a list to save the estimated action reward
        #   for SARSA, this is Q(S,A)
        self.state_action_ereward = []
        for _ in self.env.states:
            self.state_action_ereward.append([0., 0., 0., 0.])

        # Creat record list for reward and
        self.accum_r = 0
        self.record_R = []
        # Create optimal choice and sub-optimal choice list
        self.record_Q1o = []
        self.record_Q1s = []
        self.record_Q2o = []
        self.record_Q2s = []
        self.record_Q3o = []
        self.record_Q3s = []

    # Reward Record
    def recordR(self):
        # Record the R
        self.record_R.append(self.accum_r)
        self.accum_r = 0

    # Record the change of Q(s,a) and reward change
    def recordQ(self):
        # Record the Q
        if self.env.gridNum == 16:
            self.record_Q1o.append(self.state_action_ereward[13][1])  # 13_right
            self.record_Q1s.append(self.state_action_ereward[13][3])  # 13_left
            self.record_Q2o.append(self.state_action_ereward[14][2])  # 14_down
            self.record_Q2s.append(self.state_action_ereward[14][3])  # 14_left
            self.record_Q3o.append(self.state_action_ereward[10][2])  # 10_down
            self.record_Q3s.append(self.state_action_ereward[10][0])  # 10_up
        elif self.env.gridNum == 100:
            self.record_Q1o.append(self.state_action_ereward[80][1])  # 13_right
            self.record_Q1s.append(self.state_action_ereward[80][0])  # 13_up
            self.record_Q2o.append(self.state_action_ereward[62][1])  # 62_right
            self.record_Q2s.append(self.state_action_ereward[62][3])  # 62_left
            self.record_Q3o.append(self.state_action_ereward[44][2])  # 44_down
            self.record_Q3s.append(self.state_action_ereward[44][3])  # 44_left

    def run_simulation(self):
        for episode in range(1, self.max_episodes+1):
            state = self.env.reset()
            self.env.render()

            # Generate an episode following pi
            for step in range(self.max_steps):
                # Choose A from Q.(e-greedy here)
                #   using random probability to choose greedy action
                if np.random.choice([True, False], p=[1 - self.e, self.e]):
                    alist = self.state_action_ereward[state]
                    # if never achieve this state, random choose the actiton.
                    if max(alist) == min(alist):
                        action = np.random.choice(self.env.actions)
                    else:
                        action = self.env.actions[alist.index(max(alist))]
                else:
                    # Choose action randomly when greedy
                    action = np.random.choice(self.env.actions)
                # print(a)

                # Agent take action and observe S',R'
                next_state, reward, done, _ = self.env.step(action)

                # Choose A' from Q.(e-greedy here)
                if np.random.choice([True, False], p=[1 - self.e, self.e]):
                    alist = self.state_action_ereward[next_state]
                    # if never achieve this state, random choose the actiton.
                    if max(alist) == min(alist):
                        next_action = np.random.choice(self.env.actions)
                    else:
                        next_action = self.env.actions[alist.index(max(alist))]
                else:
                    # Choose action randomly when greedy
                    next_action = np.random.choice(self.env.actions)

                # Update the Q(S,A) using SARSA
                self.state_action_ereward[state][self.env.actions.index(action)] += \
                    self.lr * (reward
                               + self.gama * self.state_action_ereward[next_state][self.env.actions.index(next_action)]
                               - self.state_action_ereward[state][self.env.actions.index(action)])

                # Record
                self.accum_r += reward

                # Move to next state
                state = next_state

                # Break if terminal
                if done:
                    break

            # Record the state information every fixed step
            if not (episode % 100):
                self.recordR()
            if not (episode % 10):
                self.recordQ()

        self.env.close()


# Q-Learning
class QLearning():
    def __init__(self, envName, max_episodes=4000, max_steps=20,
                 init_randomize=True, is_render=True, is_sleep=False,
                 gama=0.9, e=0.10, lr=0.01):
        # Train parameters
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.init_randomize = init_randomize
        self.is_render = is_render
        self.is_sleep = is_sleep
        # Update parameters
        self.gama = gama
        self.e = e
        self.lr = lr

        # Build the environment
        self.env = gym.make(envName)
        self.env.setRandomInitial(self.init_randomize)
        self.env.setRender(self.is_render)
        self.env.setSleep(self.is_sleep)

        # Create a list to save the estimated action reward
        #   for SARSA, this is Q(S,A)
        self.state_action_ereward = []
        for _ in self.env.states:
            self.state_action_ereward.append([0., 0., 0., 0.])

        # Creat record list for reward and
        self.accum_r = 0
        self.record_R = []
        # Create optimal choice and sub-optimal choice list
        self.record_Q1o = []
        self.record_Q1s = []
        self.record_Q2o = []
        self.record_Q2s = []
        self.record_Q3o = []
        self.record_Q3s = []

    # Reward Record
    def recordR(self):
        # Record the R
        self.record_R.append(self.accum_r)
        self.accum_r = 0

    # Record the change of Q(s,a) and reward change
    def recordQ(self):
        # Record the Q
        if self.env.gridNum == 16:
            self.record_Q1o.append(self.state_action_ereward[13][1])  # 13_right
            self.record_Q1s.append(self.state_action_ereward[13][3])  # 13_left
            self.record_Q2o.append(self.state_action_ereward[14][2])  # 14_down
            self.record_Q2s.append(self.state_action_ereward[14][3])  # 14_left
            self.record_Q3o.append(self.state_action_ereward[10][2])  # 10_down
            self.record_Q3s.append(self.state_action_ereward[10][0])  # 10_up
        elif self.env.gridNum == 100:
            self.record_Q1o.append(self.state_action_ereward[80][1])  # 13_right
            self.record_Q1s.append(self.state_action_ereward[80][0])  # 13_up
            self.record_Q2o.append(self.state_action_ereward[62][1])  # 62_right
            self.record_Q2s.append(self.state_action_ereward[62][3])  # 62_left
            self.record_Q3o.append(self.state_action_ereward[44][2])  # 44_down
            self.record_Q3s.append(self.state_action_ereward[44][3])  # 44_left

    def run_simulation(self):
        # Repeat for max_episodes times
        for episode in range(1, self.max_episodes+1):
            state = self.env.reset()
            self.env.render()

            # Generate an episode following pi
            for step in range(self.max_steps):
                # Choose A from Q.(e-greedy here)
                #   using random probability to choose greedy action
                if np.random.choice([True, False], p=[1 - self.e, self.e]):
                    alist = self.state_action_ereward[state]
                    # if never achieve this state, random choose the actiton.
                    if max(alist) == min(alist):
                        action = np.random.choice(self.env.actions)
                    else:
                        action = self.env.actions[alist.index(max(alist))]
                else:
                    # Choose action randomly when greedy
                    action = np.random.choice(self.env.actions)

                # Agent take action and observe S',R'
                next_state, reward, done, _ = self.env.step(action)

                # Choose A' from Q.(not e-greedy here)
                alist = self.state_action_ereward[next_state]
                # if never achieve this state, random choose the actiton.
                if max(alist) == min(alist):
                    maxQ_action = np.random.choice(self.env.actions)
                else:
                    maxQ_action = self.env.actions[alist.index(max(alist))]

                # Update the Q(S,A) using SARSA
                self.state_action_ereward[state][self.env.actions.index(action)] += \
                    self.lr * (reward
                               + self.gama * self.state_action_ereward[next_state][self.env.actions.index(maxQ_action)]
                               - self.state_action_ereward[state][self.env.actions.index(action)])

                # Record
                self.accum_r += reward

                # Move to next state
                state = next_state

                # Break if terminal
                if done:
                    break

            # Record the state information every fixed step
            if not (episode % 100):
                self.recordR()
            if not (episode % 10):
                self.recordQ()

        self.env.close()


if __name__ == '__main__':
    ''' Task Algorithm '''
    print('%s Ready to train.' % (
            time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    # task-1
    task1_mmc = MMC('GridWorld16-v01', max_episodes=4000, max_steps=20,
                    init_randomize=True, is_render=True, is_sleep=False,
                    gama=0.95, e=0.1)
    task1_sarsa = SARSA('GridWorld16-v01', max_episodes=4000, max_steps=20,
                        init_randomize=True, is_render=True, is_sleep=False,
                        gama=0.95, e=0.1, lr=0.02)
    task1_qlearn = QLearning('GridWorld16-v01', max_episodes=4000, max_steps=20,
                             init_randomize=True, is_render=True, is_sleep=False,
                             gama=0.95, e=0.1, lr=0.02)
    task1_mmc.run_simulation()
    print('%s task-1 MMC Finished.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    task1_sarsa.run_simulation()
    print('%s task-1 SARSA Finished.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    task1_qlearn.run_simulation()
    print('%s task-1 Q-Learning Finished.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    # # task-2
    task2_mmc = MMC('GridWorld100-v01', max_episodes=30000, max_steps=100,
                    init_randomize=True, is_render=True, is_sleep=False,
                    gama=0.95, e=0.05)
    task2_sarsa = SARSA('GridWorld100-v01', max_episodes=30000, max_steps=100,
                        init_randomize=True, is_render=True, is_sleep=False,
                        gama=0.95, e=0.05, lr=0.04)
    task2_qlearn = QLearning('GridWorld100-v01', max_episodes=30000, max_steps=100,
                             init_randomize=True, is_render=True, is_sleep=False,
                             gama=0.95, e=0.05, lr=0.04)
    task2_mmc.run_simulation()
    print('%s task-2 MMC Finished.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    task2_sarsa.run_simulation()
    print('%s task-2 SARSA Finished.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    task2_qlearn.run_simulation()
    print('%s task-2 Q-Learning Finished.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))

    ''' Show the Record '''
    plt.ion()
    plt.clf()

    # Create the plt
    # task-1
    plt_task1_mmc = plt.subplot(4, 2, 1)
    plt_task1_sarsa = plt.subplot(4, 2, 3)
    plt_task1_qlearn = plt.subplot(4, 2, 5)
    plt_task1_reward = plt.subplot(4, 2, 7)
    # task-2
    plt_task2_mmc = plt.subplot(4, 2, 2)
    plt_task2_sarsa = plt.subplot(4, 2, 4)
    plt_task2_qlearn = plt.subplot(4, 2, 6)
    plt_task2_reward = plt.subplot(4, 2, 8)

    # Axis label setting
    # task-1
    plt_task1_mmc.set_ylabel('task1_mmc', fontsize=10)
    plt_task1_sarsa.set_ylabel('task1_sarsa', fontsize=10)
    plt_task1_qlearn.set_ylabel('task1_qlearn', fontsize=10)
    plt_task1_reward.set_ylabel('task1_reward', fontsize=10)

    plt_task1_mmc.set_xlabel('episode(X10)', fontsize=10)
    plt_task1_sarsa.set_xlabel('episode(X10)', fontsize=10)
    plt_task1_qlearn.set_xlabel('episode(X10)', fontsize=10)
    plt_task1_reward.set_xlabel('e=0.1 episode(X100)', fontsize=10)
    # task-2
    plt_task2_mmc.set_ylabel('task2_mmc', fontsize=10)
    plt_task2_sarsa.set_ylabel('task2_sarsa', fontsize=10)
    plt_task2_qlearn.set_ylabel('task2_qlearn', fontsize=10)
    plt_task2_reward.set_ylabel('task2_reward', fontsize=10)

    plt_task2_mmc.set_xlabel('episode(X10)', fontsize=10)
    plt_task2_sarsa.set_xlabel('episode(X10)', fontsize=10)
    plt_task2_qlearn.set_xlabel('episode(X10)', fontsize=10)
    plt_task2_reward.set_xlabel('e=0.1 episode(X100)', fontsize=10)

    # Show the data
    # task-1
    plt_task1_mmc.plot(task1_mmc.record_Q1o, 'r--')
    plt_task1_mmc.plot(task1_mmc.record_Q1s, 'r:')
    plt_task1_mmc.plot(task1_mmc.record_Q2o, 'b--')
    plt_task1_mmc.plot(task1_mmc.record_Q2s, 'b:')
    plt_task1_mmc.plot(task1_mmc.record_Q3o, 'g--')
    plt_task1_mmc.plot(task1_mmc.record_Q3s, 'g:')

    plt_task1_sarsa.plot(task1_sarsa.record_Q1o, 'r--')
    plt_task1_sarsa.plot(task1_sarsa.record_Q1s, 'r:')
    plt_task1_sarsa.plot(task1_sarsa.record_Q2o, 'b--')
    plt_task1_sarsa.plot(task1_sarsa.record_Q2s, 'b:')
    plt_task1_sarsa.plot(task1_sarsa.record_Q3o, 'g--')
    plt_task1_sarsa.plot(task1_sarsa.record_Q3s, 'g:')

    plt_task1_qlearn.plot(task1_qlearn.record_Q1o, 'r--')
    plt_task1_qlearn.plot(task1_qlearn.record_Q1s, 'r:')
    plt_task1_qlearn.plot(task1_qlearn.record_Q2o, 'b--')
    plt_task1_qlearn.plot(task1_qlearn.record_Q2s, 'b:')
    plt_task1_qlearn.plot(task1_qlearn.record_Q3o, 'g--')
    plt_task1_qlearn.plot(task1_qlearn.record_Q3s, 'g:')

    plt_task1_reward.plot(task1_mmc.record_R, 'r-')
    plt_task1_reward.plot(task1_sarsa.record_R, 'g-')
    plt_task1_reward.plot(task1_qlearn.record_R, 'b-')
    # task-2
    plt_task2_mmc.plot(task2_mmc.record_Q1o, 'r--')
    plt_task2_mmc.plot(task2_mmc.record_Q1s, 'r:')
    plt_task2_mmc.plot(task2_mmc.record_Q2o, 'b--')
    plt_task2_mmc.plot(task2_mmc.record_Q2s, 'b:')
    plt_task2_mmc.plot(task2_mmc.record_Q3o, 'g--')
    plt_task2_mmc.plot(task2_mmc.record_Q3s, 'g:')

    plt_task2_sarsa.plot(task2_sarsa.record_Q1o, 'r--')
    plt_task2_sarsa.plot(task2_sarsa.record_Q1s, 'r:')
    plt_task2_sarsa.plot(task2_sarsa.record_Q2o, 'b--')
    plt_task2_sarsa.plot(task2_sarsa.record_Q2s, 'b:')
    plt_task2_sarsa.plot(task2_sarsa.record_Q3o, 'g--')
    plt_task2_sarsa.plot(task2_sarsa.record_Q3s, 'g:')

    plt_task2_qlearn.plot(task2_qlearn.record_Q1o, 'r--')
    plt_task2_qlearn.plot(task2_qlearn.record_Q1s, 'r:')
    plt_task2_qlearn.plot(task2_qlearn.record_Q2o, 'b--')
    plt_task2_qlearn.plot(task2_qlearn.record_Q2s, 'b:')
    plt_task2_qlearn.plot(task2_qlearn.record_Q3o, 'g--')
    plt_task2_qlearn.plot(task2_qlearn.record_Q3s, 'g:')

    plt_task2_reward.plot(task2_mmc.record_R, 'r-')
    plt_task2_reward.plot(task2_sarsa.record_R, 'g-')
    plt_task2_reward.plot(task2_qlearn.record_R, 'b-')

    # Keep Showing
    plt.ioff()
    plt.show()

    ''' Show the different Q table for result '''
    plt.ion()
    plt.clf()

    # task 1
    # arrange in order
    task1_sarsa_Qmap = np.zeros((4,4))
    task1_qlearn_Qmap = np.zeros((4,4))
    for i in range(4):
        for j in range (4):
            task1_sarsa_Qmap[i][j] = max(task1_sarsa.state_action_ereward[4 * (3-i) + j])
            task1_qlearn_Qmap[i][j] = max(task1_qlearn.state_action_ereward[4 * (3-i) + j])
    # sarsa
    plt.imshow(task1_sarsa_Qmap)
    for i in range(4):
        for j in range (4):
            plt.text(j,i,'%.3f'% task1_sarsa_Qmap[i][j], ha='center', va='center')#,fontsize=9)
    plt.ioff()
    plt.show()
    # qlearn
    plt.imshow(task1_qlearn_Qmap)
    for i in range(4):
        for j in range (4):
            plt.text(j,i,'%.3f'% task1_qlearn_Qmap[i][j], ha='center', va='center')#,fontsize=9)
    plt.ioff()
    plt.show()

    # task 2
    # arrange in order
    task2_sarsa_Qmap = np.zeros((10,10))
    task2_qlearn_Qmap = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            task2_sarsa_Qmap[i][j] = max(task2_sarsa.state_action_ereward[10 * (9-i) + j])
            task2_qlearn_Qmap[i][j] = max(task2_qlearn.state_action_ereward[10 * (9-i) + j])
    # sarsa
    plt.imshow(task2_sarsa_Qmap)
    for i in range(10):
        for j in range(10):
            plt.text(j,i,'%.3f'% task2_sarsa_Qmap[i][j], ha='center', va='center')#,fontsize=9)
    plt.ioff()
    plt.show()
    # qlwarn
    plt.imshow(task2_qlearn_Qmap)
    for i in range(10):
        for j in range (10):
            plt.text(j,i,'%.3f'% task2_qlearn_Qmap[i][j], ha='center', va='center')#,fontsize=9)
    plt.ioff()
    plt.show()