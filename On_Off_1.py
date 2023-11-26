#!/usr/bin/env python
# coding: utf-8

# In[136]:


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

M = 100
N = 300
number_of_video = 100
number_of_client = 3
cache_size = 10
zipf_param = 1.5
total_param = 0
client_param = 0.01
zipf_p = np.array([1/(i**zipf_param) for i in range(1,number_of_video+1)])
zipf_p /= sum(zipf_p)


# # RL + FIFO or RL + LRU

# # 전체의 분포 L, 유저의 선호도는 (1-e)L + eL
# # 전체의 분포 L도 시간에 따라 (1-e')L + e'L 

# # 현재 Client의 선호도만 수정완료

# # 반복 횟수가 M과 N이 있는데 전체랭킹과 유저랭킹을 언제 바꿔줘야 하는가?

# In[137]:


def make_list(n):
    l = [[]]
    for _ in range(n-1):
        l += [[]]
    return l

def bit(VN, l):
    new_l = make_list(len(l))
    for i in range(len(l)):
        l[i] = list(map(int, l[i]))
        zeros = np.zeros(VN)
        zeros[l[i]] = 1
        new_l[i] = zeros
    return np.array(new_l)
    
def zipf(VN, N, p, n, a=1):
    if len(p.shape) == 1:
        p = np.array([p]) # 하나만 교체할 때 필요한 작업
    C = make_list(N)
    for i in range(N):
        C[i] = np.random.choice(list(range(VN)), n, False, p[i])
    return np.array(C)

def conversion(L, p):
    l = list(zip(L, p))
    l.sort()
    return np.array(list(zip(*l))[1])

class cache_env:
    def __init__(self, VN, N, p, cs, a=1):
        self.VN = VN
        self.N = N
        self.cs = cs
        self.a = a
        self.p = p
        self.state = make_list(self.N)
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)
    
    def step(self, replace, Algorithm = 'FIFO'):
        for i in range(self.N):
            if replace[i]:
                if Algorithm == 'FIFO':
                    self.FIFO(i)
                elif Algorithm == 'LRU':
                    self.LRU(i)
                else:
                    raise NameError('Wrong Algorithm Name')
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)
        reward = np.zeros((self.N, 1))
        for i in range(self.N):
            reward[i][0] = 1 * (self.request[i] in self.state[i])
        return reward
    
    def FIFO(self, idx):
        if len(self.state[idx]) < self.cs:
            self.state[idx] = np.hstack((self.state[idx], self.request[idx]))
        else:
            self.state[idx] = np.hstack((self.state[idx][1:], self.request[idx]))
    
    def LRU(self, idx):
        if self.request[idx] in self.state[idx]:
            self.state[idx] = np.delete(self.state[idx], np.where(self.state[idx] == self.request[idx][0])[0])
            self.state[idx] = np.hstack((self.state[idx], self.request[idx]))
        else:
            if len(self.state[idx]) < self.cs:
                self.state[idx] = np.hstack((self.state[idx], self.request[idx]))
            else:
                self.state[idx] = np.hstack((self.state[idx][1:], self.request[idx]))
                
    def reset(self, p):
        self.p = p
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)


# In[138]:


train_hit_M = np.zeros(N)
action_state = 2
learning_rate = 0.3
dis = 0.9
t = [n for n in range(N)]
Algorithm = 'LRU'

for m in range(M):
    next_p = make_list(number_of_client)
    L = np.array([np.random.permutation(list(range(number_of_video))) for _ in range(number_of_client)]) # Random Preference
    # L = np.array([np.array(list(range(number_of_video))) for _ in range(number_of_client)])
    for i in range(number_of_client):
        next_p[i] = conversion(L[i], zipf_p)
    next_p = np.array(next_p)
    
    if m == 0:
        p = next_p
        cache = cache_env(number_of_video, number_of_client, p, cache_size, zipf_param)
    else:
        p = (1-client_param) * p + client_param * next_p
        cache.reset(p)
    
    X = tf.placeholder(shape=[number_of_client, number_of_video * 2], dtype = tf.float32)
    W = tf.Variable(tf.random_uniform([number_of_video * 2, action_state], 0, 0.01))
    
    Qpred = tf.matmul(X, W)
    Y = tf.placeholder(shape=[number_of_client, action_state], dtype = tf.float32)
    loss = tf.reduce_sum(tf.square(Y - Qpred))

    train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    train_r = 0
    train_count = 0
    train_hit = []

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(N):
            count = 0
            for n in range(number_of_client):
                while cache.request[n][0] in cache.state[n]:
                    if Algorithm == 'LRU':
                        cache.LRU(n)
                    cache.request[n] = zipf(number_of_video, 1, p[n], 1, zipf_param)[0]
                    count += 1
                    
            e = 1/(i/10+1)
            Qs = sess.run(Qpred, feed_dict = {X:np.hstack((bit(number_of_video, cache.request),bit(number_of_video, cache.state)))})
            
            if np.random.rand(1) < e:
                a = np.random.randint(2, size = (number_of_client))
            else:
                a = np.argmax(Qs, 1)
            
            reward = cache.step(a, Algorithm) * np.identity(action_state)[a]
            Qs1 = sess.run(Qpred, feed_dict = {X:np.hstack((bit(number_of_video, cache.request),bit(number_of_video, cache.state)))})
            maxQ = np.array([np.max(Qs1, 1)]).transpose()
            Qs = reward + dis * np.hstack((maxQ,maxQ)) * np.identity(action_state)[np.argmax(Qs1, 1)]
            
            sess.run(train, feed_dict = {X:np.hstack((bit(number_of_video, cache.request),bit(number_of_video, cache.state))), Y:Qs})
            
            train_r += count
            train_count += count + number_of_client
            train_hit.append(train_r / train_count)
        train_hit_M += np.array(train_hit)

plt.plot(t,train_hit_M / M,'r')
print('Hit rate :',train_hit_M[N-1] / M)


# # FIFO

# In[139]:


def zipf(VN, N, p, n, a=1):
    if len(p.shape) == 1:
        p = np.array([p])
    C = make_list(N)
    for i in range(N):
        C[i].extend(np.random.choice(list(range(VN)), n, False, p[i]))
    return C

class cache_env:
    def __init__(self, VN, N, p, cs, a=1):
        self.VN = VN
        self.N = N
        self.cs = cs
        self.a = a
        self.p = p
        self.state = make_list(N)
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)
    
    def step(self):
        count = 0
        for i in range(self.N):
            while self.request[i][0] in self.state[i]:
                self.request[i] = zipf(self.VN, 1, self.p[i], 1, self.a)[0]
                count += 1
        self.replacement()
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)
        return count
    
    def replacement(self):
        for i in range(self.N):
            if len(self.state[i]) < self.cs:
                self.state[i].append(self.request[i][0])
            else:
                self.state[i] = self.state[i][1:] + self.request[i]
                
    def reset(self, p):
        self.p = p
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)


# In[140]:


FIFO_hit_M = np.zeros(N)
t = [n for n in range(N)]
for m in range(M):
    next_p = make_list(number_of_client)
    L = np.array([np.random.permutation(list(range(number_of_video))) for _ in range(number_of_client)]) # Random Preference
    # L = np.array([np.array(list(range(number_of_video))) for _ in range(number_of_client)])
    for i in range(number_of_client):
        next_p[i] = conversion(L[i], zipf_p)
    next_p = np.array(next_p)
    
    if m == 0:
        p = next_p
        cache = cache_env(number_of_video, number_of_client, p, cache_size, zipf_param)
    else:
        p = (1-client_param) * p + client_param * next_p
        cache.reset(p)

    FIFO_r = 0
    FIFO_count = 0
    FIFO_hit = []

    for i in range(N):
        count = cache.step()
        FIFO_r += count
        FIFO_count += count + number_of_client
        FIFO_hit.append(FIFO_r / FIFO_count)
    FIFO_hit_M += np.array(FIFO_hit)

plt.plot(t,FIFO_hit_M / M,'b')
print('Hit rate :',FIFO_hit_M[N-1] / M)


# # LFU

# In[141]:


class cache_env:
    def __init__(self, VN, N, p, cs, a=1):
        self.VN = VN
        self.N = N
        self.cs = cs
        self.a = a
        self.p = p
        self.state = make_list(N)
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)
        self.rq_count = [0] * VN
    
    def step(self):
        count = 0
        for i in range(self.N):
            while self.request[i][0] in self.state[i]:
                self.request[i] = zipf(self.VN, 1, self.p[i], 1, self.a)[0]
                self.rq_count[self.request[i][0]] += 1
                count += 1
        self.replacement()
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)
        return count
    
    def replacement(self):
        def count(i):
            return self.rq_count[i]
        for i in range(self.N):
            if len(self.state[i]) < self.cs:
                self.state[i].append(self.request[i][0])
            else:
                count_l = list(map(count, self.state[i]))
                idx = np.where(np.array(count_l) == min(count_l))[0]
                self.state[i].remove(self.state[i][idx[0]])
                self.state[i].append(self.request[i][0])
    
    def reset(self, p):
        self.p = p
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)


# In[142]:


LFU_hit_M = np.zeros(N)
t = [n for n in range(N)]
for m in range(M):
    next_p = make_list(number_of_client)
    L = np.array([np.random.permutation(list(range(number_of_video))) for _ in range(number_of_client)]) # Random Preference
    # L = np.array([np.array(list(range(number_of_video))) for _ in range(number_of_client)])
    for i in range(number_of_client):
        next_p[i] = conversion(L[i], zipf_p)
    next_p = np.array(next_p)
    
    if m == 0:
        p = next_p
        cache = cache_env(number_of_video, number_of_client, p, cache_size, zipf_param)
    else:
        p = (1-client_param) * p + client_param * next_p
        cache.reset(p)
    LFU_r = 0
    LFU_count = 0
    LFU_hit = []

    for i in range(N):
        count = cache.step()
        LFU_r += count
        LFU_count += count + number_of_client
        LFU_hit.append(LFU_r / LFU_count)
    LFU_hit_M += np.array(LFU_hit)
        
plt.plot(t,LFU_hit_M / M,'k')
print('Hit rate :',LFU_hit_M[N-1] / M)


# # LRU

# In[143]:


class cache_env:
    def __init__(self, VN, N, p, cs, a=1):
        self.VN = VN
        self.N = N
        self.cs = cs
        self.a = a
        self.p = p
        self.state = make_list(N)
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)
    
    def step(self):
        count = 0
        for i in range(self.N):
            while self.request[i][0] in self.state[i]:
                self.replacement(i)
                self.request[i] = zipf(self.VN, 1, self.p[i], 1, self.a)[0]
                count += 1
        for i in range(self.N):
            self.replacement(i)
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)
        return count
    
    def replacement(self, idx):
        if self.request[idx][0] in self.state[idx]:
            self.state[idx].remove(self.request[idx][0])
            self.state[idx].append(self.request[idx][0])
        else:
            if len(self.state[idx]) < self.cs:
                self.state[idx].append(self.request[idx][0])
            else:
                self.state[idx] = self.state[idx][1:] + self.request[idx]
                
    def reset(self, p):
        self.p = p
        self.request = zipf(self.VN, self.N, self.p, 1, self.a)


# In[144]:


LRU_hit_M = np.zeros(N)
t = [n for n in range(N)]
for m in range(M):
    next_p = make_list(number_of_client)
    L = np.array([np.random.permutation(list(range(number_of_video))) for _ in range(number_of_client)]) # Random Preference
    # L = np.array([np.array(list(range(number_of_video))) for _ in range(number_of_client)])
    for i in range(number_of_client):
        next_p[i] = conversion(L[i], zipf_p)
    next_p = np.array(next_p)
    
    if m == 0:
        p = next_p
        cache = cache_env(number_of_video, number_of_client, p, cache_size, zipf_param)
    else:
        p = (1-client_param) * p + client_param * next_p
        cache.reset(p)

    LRU_r = 0
    LRU_count = 0
    LRU_hit = []

    for i in range(N):
        count = cache.step()
        LRU_r += count
        LRU_count += count + number_of_client
        LRU_hit.append(LRU_r / LRU_count)
    LRU_hit_M += np.array(LRU_hit)

plt.plot(t,LRU_hit_M / M,'g')
print('Hit rate :',LRU_hit_M[N-1] / M)


# # Comparison

# In[145]:


plt.plot(t,train_hit_M / M,'red')
plt.plot(t,FIFO_hit_M / M,'blue')
plt.plot(t,LFU_hit_M / M,'black')
plt.plot(t,LRU_hit_M / M,'green')
plt.legend(['RL', 'FIFO', 'LFU', 'LRU'])
plt.savefig("On-Off_1")


# In[ ]:




