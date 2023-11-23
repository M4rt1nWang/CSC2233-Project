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


# In[129]:


M = 1
N = 100
test_N = 15
num_of_video = 100
num_of_client = 20
cache_size = 20
zipf_param = 1
s_len = 10
l_len = 100
K = 5
ch_interval = 1000
rho = 0.5


# In[130]:


def RL_Caching(M, N, num_of_video, num_of_client, cache_size, s_len, l_len, K, zipf_param):
    def zipf(VN, P, n):
        return np.random.choice(VN, n, False, P)

    class A2C_Agent:
        def __init__(self, state_size, action_size, batch_size):
            global advantages
            self.state_size = state_size
            self.action_size = action_size
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
            # softmax는 값의 차이가 크면 nan이 발생할 수 있다.
            actor.compile(loss=self.score_func_loss, optimizer=tf.keras.optimizers.Adam(lr=self.actor_lr))
            return actor

        def build_critic(self):
            critic = tf.keras.models.Sequential()
            critic.add(Dense(self.state_size, input_dim=self.state_size, activation='relu',kernel_initializer='he_uniform'))
            critic.add(Dense(self.action_size, activation='linear',kernel_initializer='he_uniform'))
            critic.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.critic_lr))
            return critic

        def train_model(self, state_batch, reward_batch, target_train):
            global advantages
            states = np.vstack([x[0] for x in state_batch])
            actions = np.array([x[1] for x in state_batch])
            next_states = np.vstack([x[2] for x in state_batch])
            rewards = np.vstack([x for x in reward_batch])
            size = self.batch_size
            
            target = np.zeros((size, self.action_size))
            advantages = np.zeros((size, self.action_size))

            value = np.vstack(self.main_critic.predict(states)[range(size), actions])
            next_p = self.target_actor.predict(next_states)
            next_action = np.array([np.random.choice(self.action_size, p = next_p[i]) for i in range(size)])
            next_value = np.vstack(self.target_critic.predict(next_states)[range(size), next_action])

            target[range(size), actions] = np.reshape(rewards + self.discount_factor * next_value, size)
            advantages[range(size), actions] = np.reshape(np.vstack(target[range(size), np.hstack(actions)]) - value, size)

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
            if a == 'pass':
                pass
            elif a == 'append':
                self.state.append(self.rq[0])
            else:
                if a == 0:
                    pass
                else:
                    self.state.remove(self.state[a-1])
                    self.state.append(self.rq[0])
            self.rq = zipf(self.VN, self.P, 1)
            self.count()
            return prev_state
        
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

    with tf.Graph().as_default():
        Agents = [A2C_Agent(state_size, action_size, batch_size) for _ in range(num_of_client)]
        cache = [cache_env(num_of_video, cache_size, s_len, l_len, K, zipf_param) for _ in range(num_of_client)]
        train_r = 0
        train_count = 0
        train_con = []
        for i in range(N):
            e = 1 / ((i/10)+1)
            Train = np.zeros(num_of_client)
            for n in range(num_of_client):
                if (i+1) % ch_interval == 0:
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
                    
                    critics = Agents[n].main_critic.predict(np.array([state]))[0][a_list]
                    idx = np.where(critics == max(critics))[0][0]
                    a = a_list[idx]

                    prev_state = cache[n].step(a)
                    rq = list(cache[n].rq)
                    state = np.hstack((cache[n].s_cnt[rq + cache[n].state], cache[n].l_cnt[rq + cache[n].state]))
                    
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
                        reward[i] = 1
                        local[i] = 0
                else:
                    local[c[0]] = 0
            reward = reward + local * 2

            for n in range(num_of_client):
                if Train[n]:
                    reward_memory[n].append(reward[n])
                    if len(reward_memory[n]) >= batch_size:
                        batch = np.random.choice(len(reward_memory[n]), min(len(reward_memory[n]), batch_size), False)
                        state_batch = np.array(state_memory[n])[batch]
                        reward_batch = np.array(reward_memory[n])[batch]
                        Agents[n].train_model(state_batch, reward_batch, (i+1) % target_update_fre == 0)

            train_r += connection
            train_count += num_of_client
            train_con.append(train_r / train_count)

        test_r = 0
        test_count = 0
        test_con = []
        for i in range(test_N):
            for n in range(num_of_client):
                if (i+1) % ch_interval == 0:
                    new_P = np.array([1/(i**cache[n].a) for i in range(1, cache[n].VN+1)])
                    new_P /= sum(new_P)
                    np.random.shuffle(new_P)
                    cache[n].P = rho * cache[n].P + (1-rho) * new_P
                
                if cache[n].rq in cache[n].state:
                    cache[n].step('pass')
                else:
                    rq = list(cache[n].rq)
                    state = np.hstack((cache[n].s_cnt[rq + cache[n].state], cache[n].l_cnt[rq + cache[n].state]))
                    
                    pred = Agents[n].main_actor.predict(np.array([state]))[0]
                    a_list = np.random.choice(action_size, K, False, p = pred)
                    
                    critics = Agents[n].main_critic.predict(np.array([state]))[0][a_list]
                    idx = np.where(critics == max(critics))[0][0]
                    a = a_list[idx]
                    
                    _ = cache[n].step(a)

            rqs = np.array([cache[i].rq for i in range(num_of_client)])
            caches = [cache[i].state for i in range(num_of_client)]
            cn, rq, ch = Local(rqs, caches)
            cn, rq, ch = BCG(list(range(num_of_video)), cn, rq, ch)
            cn, rq, ch = XBCG(list(range(num_of_video)), cn, rq, ch)
            connection = len(cn)

            test_r += connection
            test_count += num_of_client
            test_con.append(test_r / test_count)
            
    return train_con, test_con


# In[131]:


RL_connection, RL_test = RL_Caching(M, N, num_of_video, num_of_client, cache_size, s_len, l_len, K, zipf_param)


# In[ ]:




