#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Grouping import *

M = 10
N = 500
num_of_video = 100
num_of_client = 10
cache_size = 5
zipf_param = 1


# # FIFO

# In[2]:


def FIFO_Caching(M, N, num_of_video, num_of_client, cache_size, zipf_param):
    def make_list(n):
        l = [[]]
        for _ in range(n-1):
            l += [[]]
        return l

    def zipf(VN, N, P, n, a=1):
        if len(P.shape) == 1:
            P = np.array([P])
        p = np.array([1/(i**a) for i in range(1,VN+1)])
        p = p / sum(p)
        C = make_list(N)
        for i in range(N):
            c = np.random.choice(list(range(VN)), n, False, p)
            for c in c:
                C[i].append(P[i][c])
        return list(C)

    class cache_env:
        def __init__(self, VN, N, P, cs, a=1):
            self.VN = VN
            self.N = N
            self.cs = cs
            self.a = a
            self.P = P
            self.state = make_list(N)
            self.request = zipf(self.VN, self.N, self.P, 1, self.a)

        def step(self):
            count = 0
            for i in range(self.N):
                while self.request[i][0] in self.state[i]:
                    self.request[i] = zipf(self.VN, 1, self.P[i], 1, self.a)[0]
                    count += 1
            self.replacement()
            self.request = zipf(self.VN, self.N, self.P, 1, self.a)
            cn, rq, ch = Local(self.request, self.state)
            cn, rq, ch = BCG(list(range(self.VN)), cn, rq, ch)
            cn, rq, ch = XBCG(list(range(self.VN)), cn, rq, ch)
            connection = len(cn)
            return count, connection

        def replacement(self):
            for i in range(self.N):
                if len(self.state[i]) < self.cs:
                    self.state[i].append(self.request[i][0])
                else:
                    self.state[i] = self.state[i][1:] + self.request[i]

        def reset(self, P):
            self.P = P
            
    FIFO_connection = np.zeros(N)
    for m in range(M):
        P = np.array([np.random.permutation(list(range(num_of_video))) for _ in range(num_of_client)]) # Random
        # P = np.array([np.array(list(range(num_of_video))) for _ in range(num_of_client)])
        cache = cache_env(num_of_video, num_of_client, P, cache_size, zipf_param)

        FIFO_r = 0
        FIFO_count = 0
        FIFO_hit = []

        for i in range(N):
            count, connection = cache.step()
            FIFO_r += connection
            FIFO_count += count + num_of_client
            FIFO_hit.append(FIFO_r / FIFO_count)
        FIFO_connection += np.array(FIFO_hit)

    return FIFO_connection / M


# # LFU

# In[3]:


def LFU_Caching(M, N, num_of_video, num_of_client, cache_size, zipf_param):
    def make_list(n):
        l = [[]]
        for _ in range(n-1):
            l += [[]]
        return l

    def zipf(VN, N, P, n, a=1):
        if len(P.shape) == 1:
            P = np.array([P])
        p = np.array([1/(i**a) for i in range(1,VN+1)])
        p = p / sum(p)
        C = make_list(N)
        for i in range(N):
            c = np.random.choice(list(range(VN)), n, False, p)
            for c in c:
                C[i].append(P[i][c])
        return list(C)
    
    class cache_env:
        def __init__(self, VN, N, P, cs, a=1):
            self.VN = VN
            self.N = N
            self.cs = cs
            self.a = a
            self.P = P
            self.state = make_list(N)
            self.request = zipf(self.VN, self.N, self.P, 1, self.a)
            self.rq_count = [0] * VN

        def step(self):
            count = 0
            # self.rq_count = list(np.array(self.rq_count) * 0.9)  # LFU의 단점 개선
            for i in range(self.N):
                self.rq_count[self.request[i][0]] += 1
                while self.request[i][0] in self.state[i]:
                    self.request[i] = zipf(self.VN, 1, self.P[i], 1, self.a)[0]
                    self.rq_count[self.request[i][0]] += 1
                    count += 1
            self.replacement()
            self.request = zipf(self.VN, self.N, self.P, 1, self.a)
            cn, rq, ch = Local(self.request, self.state)
            cn, rq, ch = BCG(list(range(self.VN)), cn, rq, ch)
            cn, rq, ch = XBCG(list(range(self.VN)), cn, rq, ch)
            connection = len(cn)
            return count, connection

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

        def reset(self, P):
            self.P = P
            
    LFU_connection = np.zeros(N)
    for m in range(M):
        P = np.array([np.random.permutation(list(range(num_of_video))) for _ in range(num_of_client)]) # Random
        # P = np.array([np.array(list(range(num_of_video))) for _ in range(num_of_client)])
        cache = cache_env(num_of_video, num_of_client, P, cache_size, zipf_param)

        LFU_r = 0
        LFU_count = 0
        LFU_hit = []

        for i in range(N):
            count, connection = cache.step()
            LFU_r += connection
            LFU_count += count + num_of_client
            LFU_hit.append(LFU_r / LFU_count)
        LFU_connection += np.array(LFU_hit)

    return LFU_connection / M


# # LRU

# In[4]:


def LRU_Caching(M, N, num_of_video, num_of_client, cache_size, zipf_param):
    def make_list(n):
        l = [[]]
        for _ in range(n-1):
            l += [[]]
        return l

    def zipf(VN, N, P, n, a=1):
        if len(P.shape) == 1:
            P = np.array([P])
        p = np.array([1/(i**a) for i in range(1,VN+1)])
        p = p / sum(p)
        C = make_list(N)
        for i in range(N):
            c = np.random.choice(list(range(VN)), n, False, p)
            for c in c:
                C[i].append(P[i][c])
        return list(C)
    
    class cache_env:
        def __init__(self, VN, N, P, cs, a=1):
            self.VN = VN
            self.N = N
            self.cs = cs
            self.a = a
            self.P = P
            self.state = make_list(N)
            self.request = zipf(self.VN, self.N, self.P, 1, self.a)

        def step(self):
            count = 0
            for i in range(self.N):
                while self.request[i][0] in self.state[i]:
                    self.replacement(i)
                    self.request[i] = zipf(self.VN, 1, self.P[i], 1, self.a)[0]
                    count += 1
            for i in range(self.N):
                self.replacement(i)
            self.request = zipf(self.VN, self.N, self.P, 1, self.a)
            cn, rq, ch = Local(self.request, self.state)
            cn, rq, ch = BCG(list(range(self.VN)), cn, rq, ch)
            cn, rq, ch = XBCG(list(range(self.VN)), cn, rq, ch)
            connection = len(cn)
            return count, connection

        def replacement(self, idx):
            if self.request[idx][0] in self.state[idx]:
                self.state[idx].remove(self.request[idx][0])
                self.state[idx].append(self.request[idx][0])
            else:
                if len(self.state[idx]) < self.cs:
                    self.state[idx].append(self.request[idx][0])
                else:
                    self.state[idx] = self.state[idx][1:] + self.request[idx]

        def reset(self, P):
            self.P = P

    LRU_connection = np.zeros(N)
    for m in range(M):
        P = np.array([np.random.permutation(list(range(num_of_video))) for _ in range(num_of_client)]) # Random
        # P = np.array([np.array(list(range(num_of_video))) for _ in range(num_of_client)])
        cache = cache_env(num_of_video, num_of_client, P, cache_size, zipf_param)

        LRU_r = 0
        LRU_count = 0
        LRU_hit = []

        for i in range(N):
            count, connection = cache.step()
            LRU_r += connection
            LRU_count += count + num_of_client
            LRU_hit.append(LRU_r / LRU_count)
        LRU_connection += np.array(LRU_hit)

    return LRU_connection / M


# # Comparison

# In[5]:


FIFO_connection = FIFO_Caching(M, N, num_of_video, num_of_client, cache_size, zipf_param)
LFU_connection = LFU_Caching(M, N, num_of_video, num_of_client, cache_size, zipf_param)
LRU_connection = LRU_Caching(M, N, num_of_video, num_of_client, cache_size, zipf_param)

t = [n for n in range(N)]

plt.plot(t,FIFO_connection,'b')
plt.plot(t,LFU_connection,'g')
plt.plot(t,LRU_connection,'r')
plt.legend(['FIFO', 'LFU', 'LRU'])
plt.savefig("Cache_replacement")

print('FIFO Connection :',FIFO_connection[N-1])
print('LFU Connection :',LFU_connection[N-1])
print('LRU Connection :',LRU_connection[N-1])

