from sac2019 import SACAgent
from supply_chain_env import SupplyChainEnv
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch

# Supply Chain Parameters
con_mat = [[1,1]]
price = [200,200]
k_pr = [60]
k_st = [[8],[8,10]]
k_pe = [40,40]
k_tr = [[80,150]]
lead_time = [[1,1]]
st_max = [[100000],[20000,10000]]
de_hist_len = 4
zeta = 5

# Supply Chain Environment
env = SupplyChainEnv(con_mat, price, k_pr, k_st, k_pe, k_tr, lead_time, st_max, de_hist_len, zeta)

# SAC Parameters
tau = 0.005
gamma = 0.99
alpha = 0.0001
a_lr = 0.002
q_lr = 0.002
p_lr = 0.002
buffer_maxlen = 1000

# SAC Agent
agent = SACAgent(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen)

# Training Parameters
max_episodes = 1000
max_steps = 50
batch_size = 32

# Other Parameters
pi = 3.1415926

episode_rewards = []

for j in range(max_episodes):
    # Reset
    state = env.reset([0], [100, 100])
    episode_reward = 0

    # Demand profile
    d1 = []
    d2 = []
    d1_max = random.uniform(100, 130)
    d2_max = random.uniform(100, 130)
    for k in range(max_steps):
        # d1_temp = d1_max / 2 * math.sin(2 * pi * k / 12) + d1_max / 2 + random.uniform(0, 2)
        # d2_temp = - d2_max / 2 * math.sin(2 * pi * k / 8) + d2_max / 2 + random.uniform(0, 2)
        d1_temp = d1_max/2 * math.exp(-k/10) + d1_max/2 + random.uniform(0, 10)
        d2_temp = d2_max/2 * math.exp(-k/10) + d2_max/2 + random.uniform(0, 10)
        d1.append(round(d1_temp))
        d2.append(round(d2_temp))

    for i in range(max_steps):

        action = agent.get_action(state)
        a1_temp = round(action[1])
        a2_temp = round(action[2])
        a0_temp = round(action[0]) + a1_temp + a2_temp

        next_state, reward, done = env.step([a0_temp], [[a1_temp, a2_temp]], [d1[i], d2[i]])
        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)

        if done or i == max_steps - 1:
            episode_rewards.append(episode_reward)
            print("Episode ", j, "Reward: %.4f" % episode_reward, 'a1_final: %.2f' % env.a_w[0][0][-1], 'a2_final: %.2f' % env.a_w[0][1][-1], 'a0_final: %.2f' % env.a_f[0][-1])
            if j == max_episodes - 1:
                a1f = env.a_w[0][0]
                a2f = env.a_w[0][1]
                a0f = env.a_f[0]
                s1f = env.s_w[0]
                s2f = env.s_w[1]
                s0f = env.s_f[0]
            break

        state = next_state

# Save RL Model
#PATH = "C:\\Users\\tiand\\Desktop\\RL research with SLB\\RL_Simulations\\Saved_model\\SAC_model_lead_time3"
#torch.save(agent, PATH)


plt.figure(1)
plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.ylabel('Loss/Reward')
plt.xlabel('training steps')

plt.figure(2)
plt.plot(np.arange(len(a1f)), a1f)
plt.ylabel('a1')
plt.xlabel('time steps')

plt.figure(3)
plt.plot(np.arange(len(a2f)), a2f)
plt.ylabel('a2')
plt.xlabel('time steps')

plt.figure(4)
plt.plot(np.arange(len(a0f)), a0f)
plt.ylabel('a0')
plt.xlabel('time steps')

plt.figure(5)
plt.plot(np.arange(len(s1f)), s1f)
plt.ylabel('s1')
plt.xlabel('time steps')

plt.figure(6)
plt.plot(np.arange(len(s2f)), s2f)
plt.ylabel('s2')
plt.xlabel('time steps')

plt.figure(7)
plt.plot(np.arange(len(s0f)), s0f)
plt.ylabel('s0')
plt.xlabel('time steps')

plt.figure(8)
plt.plot(np.arange(len(d1)), d1)
plt.ylabel('d1')
plt.xlabel('time steps')

plt.figure(9)
plt.plot(np.arange(len(d2)), d2)
plt.ylabel('d2')
plt.xlabel('time steps')
plt.show()

