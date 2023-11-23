#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from collections import deque
import matplotlib.pyplot as plt
from Grouping import *
import warnings
warnings.filterwarnings(action='ignore')


# # RL (Wolpertinger)

# In[2]:


def RL_Caching(M, N, num_of_video, num_of_client, cache_size, s_len, l_len, K, zipf_param):
    def zipf(VN, P, n):
        return np.random.choice(VN, n, False, P)

    class A2C_Agent:
        def __init__(self, state_size, action_size, batch_size):
            global advantages
            self.state_size = state_size
            self.action_size = action_size
            self.value_size = 1
            self.batch_size = batch_size
            advantages = np.zeros((self.batch_size, self.action_size))

            self.discount_factor = 0.9
            self.actor_lr = 0.001
            self.critic_lr = 0.01

            self.main_actor = self.build_actor()
            self.target_actor = self.build_actor()
            self.target_actor.set_weights(self.main_actor.get_weights())
            self.main_critic = self.build_critic()
            self.target_critic = self.build_critic()
            self.target_critic.set_weights(self.main_critic.get_weights())

        def score_func_loss(self, Y, action_pred):
            global advantages
            log_lik = -Y * tf.math.log(action_pred)
            log_lik_adv = log_lik * advantages
            loss = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))
            return loss

        def build_actor(self):
            actor = tf.keras.models.Sequential()
            actor.add(Dense(self.state_size, input_dim=self.state_size, activation='relu',kernel_initializer='he_uniform'))
            actor.add(Dense(self.action_size, activation='softmax',kernel_initializer='he_uniform'))
            actor.compile(loss=self.score_func_loss, optimizer=tf.keras.optimizers.Adam(lr=self.actor_lr))
            return actor

        def build_critic(self):
            critic = tf.keras.models.Sequential()
            critic.add(Dense(self.state_size, input_dim=self.state_size, activation='relu',kernel_initializer='he_uniform'))
            critic.add(Dense(self.value_size, activation='linear',kernel_initializer='he_uniform'))
            critic.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.critic_lr))
            return critic

        def train_model(self, state_batch, reward_batch, target_train):
            global advantages
            states = np.vstack([x[0] for x in state_batch])
            actions = np.array([x[1] for x in state_batch])
            next_states = np.vstack([x[2] for x in state_batch])
            rewards = np.vstack([x for x in reward_batch])

            target = np.zeros((self.batch_size, self.value_size))
            advantages = np.zeros((self.batch_size, self.action_size))

            value = self.main_critic.predict(states)
            next_value = self.target_critic.predict(next_states)

            target = rewards + self.discount_factor * next_value
            advantages[range(self.batch_size), actions] = np.reshape(target - value, self.batch_size)

            self.main_actor.fit(states, advantages, epochs=1, verbose=0)
            self.main_critic.fit(states, target, epochs=1, verbose=0)

            if target_train:
                self.target_actor.set_weights(self.main_actor.get_weights())
                self.target_critic.set_weights(self.main_critic.get_weights())

    class cache_env:
        def __init__(self, VN, cs, s_len, l_len, K, a=1):
            self.VN = VN
            self.cs = cs
            self.K = K
            self.s_len = s_len
            self.l_len = l_len
            self.s_buffer = []
            self.l_buffer = []
            self.s_cnt = np.zeros(VN)
            self.l_cnt = np.zeros(VN)
            self.a = a
            self.P = np.array([1/(i**self.a) for i in range(1, self.VN+1)])
            self.P /= sum(self.P)
            np.random.shuffle(self.P)
            self.state = []
            self.rq = zipf(self.VN, self.P, 1)
            self.count()

        def step(self, a):
            rq = list(self.rq)
            prev_state = np.hstack((self.s_cnt[rq + self.state], self.l_cnt[rq + self.state]))
            states = None
            if a == 'pass':
                pass
            elif a == 'append':
                self.state.append(self.rq[0])
            else:
                states = [copy.deepcopy(self.state) for _ in range(K)]
                for i in range(K):
                    if a[i] == 0:
                        continue
                    states[i].remove(states[i][a[i]-1])
                    states[i].append(self.rq[0])
            self.rq = zipf(self.VN, self.P, 1)
            self.count()
            return prev_state, states
        
        def count(self):
            if sum(self.s_cnt) == self.s_len:
                self.s_cnt[self.s_buffer[0]] -= 1
                self.s_buffer = self.s_buffer[1:]
            self.s_cnt[self.rq] += 1
            self.s_buffer.append(self.rq[0])
            if sum(self.l_cnt) == self.l_len:
                self.l_cnt[self.l_buffer[0]] -= 1
                self.l_buffer = self.l_buffer[1:]
            self.l_cnt[self.rq] += 1
            self.l_buffer.append(self.rq[0])
            

    state_size = 2 * (cache_size + 1)
    action_size = cache_size + 1
    target_update_fre = 10
    
    memory_size = 50
    state_memory = [deque(maxlen = memory_size) for _ in range(num_of_client)]
    reward_memory = [deque(maxlen = memory_size) for _ in range(num_of_client)]
    batch_size = 10

    reward_list = []
    
    with tf.Graph().as_default():
        Agents = [A2C_Agent(state_size, action_size, batch_size) for _ in range(num_of_client)]
        cache = [cache_env(num_of_video, cache_size, s_len, l_len, K, zipf_param) for _ in range(num_of_client)]
        for i in range(N):
            Train = np.zeros(num_of_client)
            for n in range(num_of_client):
                if np.random.rand() < ch_p:
                    new_P = np.array([1/(i**cache[n].a) for i in range(1, cache[n].VN+1)])
                    new_P /= sum(new_P)
                    np.random.shuffle(new_P)
                    cache[n].P = rho * cache[n].P + (1-rho) * new_P
                    
                if cache[n].rq in cache[n].state:
                    prev_state = cache[n].step('pass')
                elif len(cache[n].state) < cache[n].cs:
                    prev_state = cache[n].step('append')
                else:
                    Train[n] = 1
                    rq = list(cache[n].rq)
                    state = np.hstack((cache[n].s_cnt[rq + cache[n].state], cache[n].l_cnt[rq + cache[n].state]))

                    pred = Agents[n].main_actor.predict(np.array([state]))[0]
                    a_list = np.random.choice(action_size, K, False, p = pred)

                    prev_state, states = cache[n].step(a_list)
                    rq = list(cache[n].rq)
                    state_list = [np.hstack((cache[n].s_cnt[rq + states[i]], cache[n].l_cnt[rq + states[i]])) for i in range(K)]
                    critics = Agents[n].main_critic.predict(np.vstack(state_list))
                    idx = np.where(critics == max(critics))[0][0]
                    a = a_list[idx]
                    cache[n].state = states[idx]
                    
                    state = state_list[idx]
                    state_memory[n].append((prev_state, a, state))

            rqs = np.array([cache[i].rq for i in range(num_of_client)])
            caches = [cache[i].state for i in range(num_of_client)]
            cn, rq, ch = Local(rqs, caches)
            cn, rq, ch = BCG(list(range(num_of_video)), cn, rq, ch)
            cn, rq, ch = XBCG(list(range(num_of_video)), cn, rq, ch)
            connection = len(cn)
            reward = np.zeros(num_of_client)
            local = np.ones(num_of_client)
            for c in cn:
                if len(c) > 1:
                    for i in c:
                        reward[i] = 0.5
                        local[i] = 0
                else:
                    local[c[0]] = 0
            reward = reward + local
            
            reward_list.append(sum(reward)/num_of_client)
            
            for n in range(num_of_client):
                if Train[n]:
                    reward_memory[n].append(reward[n])
                    if len(reward_memory[n]) >= batch_size:
                        batch = np.random.choice(len(reward_memory[n]), min(len(reward_memory[n]), batch_size), False)
                        state_batch = np.array(state_memory[n])[batch]
                        reward_batch = np.array(reward_memory[n])[batch]
                        Agents[n].train_model(state_batch, reward_batch, (i+1) % target_update_fre == 0)
            
    return reward_list


# # Graph

# In[3]:


M = 30
N = 100
num_of_video = 100
num_of_client = 50
cache_size = 20
zipf_param = 1
s_len = 10
l_len = 100
K = 5
ch_p = 0.001
rho = 0.5

t = [n for n in range(N)]
mean_reward1 = np.zeros(N)
mean_reward2 = np.zeros(N)

for i in range(M):
    reward_list = np.array(RL_Caching(M, N, num_of_video, num_of_client, cache_size, s_len, l_len, K, zipf_param))
    mean_reward1 += reward_list / M
    if i == 0:
        max_list1 = reward_list
        min_list1 = reward_list
    else:
        max_list1 = [max(max_list1[i], reward_list[i]) for i in t]
        min_list1 = [min(min_list1[i], reward_list[i]) for i in t]

    reward_list = np.array(RL_Caching(M, N, num_of_video, num_of_client, cache_size//2, s_len, l_len, K, zipf_param))
    mean_reward2 += reward_list / M
    if i == 0:
        max_list2 = reward_list
        min_list2 = reward_list
    else:
        max_list2 = [max(max_list2[i], reward_list[i]) for i in t]
        min_list2 = [min(min_list2[i], reward_list[i]) for i in t]
        
plt.plot(t, mean_reward1, 'r')
plt.plot(t, mean_reward2, 'b')
plt.legend(['C = 20','C = 10'])
plt.savefig("")


# In[4]:


print(list(mean_reward1))


# In[5]:


print(max_list1)


# In[6]:


print(min_list1)


# In[7]:


print(list(mean_reward2))


# In[8]:


print(max_list2)


# In[9]:


print(min_list2)


# In[ ]:




