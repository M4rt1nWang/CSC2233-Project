import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from Grouping import *
import warnings
warnings.filterwarnings(action='ignore')

def FIFO_Caching(M, N, num_of_video, num_of_client, cache_size, zipf_param):
    def make_list(n):
        l = [[]]
        for _ in range(n-1):
            l += [[]]
        return l

    def zipf(VN, N, P, n, a=1):
        if len(P.shape) == 1:
            P = np.array([P])
        C = make_list(N)
        for i in range(N):
            C[i] = list(np.random.choice([i for i in range(VN)], n, False, P[i]))
        return C

    class cache_env:
        def __init__(self, VN, N, cs, a=1):
            self.VN = VN
            self.N = N
            self.cs = cs
            self.a = a
            self.P = np.array([[1/(i**self.a) for i in range(1, self.VN+1)] for _ in range(self.N)])
            self.P /= np.array([np.sum(self.P, 1)]).transpose()
            for i in range(self.N):
                np.random.shuffle(self.P[i])
            self.state = make_list(N)
            self.request = zipf(self.VN, self.N, self.P, 1, self.a)

        def step(self):
            self.replacement()
            self.request = zipf(self.VN, self.N, self.P, 1, self.a)
            cn, rq, ch = Local(self.request, self.state)
            cn, rq, ch = BCG(list(range(self.VN)), cn, rq, ch)
            connection = len(cn)
            cn, rq, ch = XBCG(list(range(self.VN)), cn, rq, ch)
            Xconnection = len(cn)
            return connection, Xconnection

        def replacement(self):
            for i in range(self.N):
                if self.request[i][0] not in self.state[i]:
                    if len(self.state[i]) < self.cs:
                        self.state[i].append(self.request[i][0])
                    else:
                        self.state[i] = self.state[i][1:] + self.request[i]
            
    cache = cache_env(num_of_video, num_of_client, cache_size, zipf_param)
    FIFO_r = 0
    FIFO_Xr = 0
    FIFO_count = 0
    FIFO_con = []
    FIFO_Xcon = []
    for i in range(N):
        if (i+1) % ch_interval == 0:
            new_P = np.array([np.array([1/(i**cache.a) for i in range(1, cache.VN+1)]) for _ in range(cache.N)])
            new_P /= np.array([np.sum(new_P, 1)]).transpose()
            for k in range(cache.N):
                np.random.shuffle(new_P[k])
            cache.P = rho * cache.P + (1-rho) * new_P
            
        connection, Xconnection = cache.step()
        FIFO_r += connection
        FIFO_Xr += Xconnection
        FIFO_count += num_of_client
        FIFO_con.append(FIFO_r / FIFO_count)
        FIFO_Xcon.append(FIFO_Xr / FIFO_count)

    test_r = 0
    test_Xr = 0
    test_count = 0
    test_con = []
    test_Xcon = []
    for i in range(test_N):   
        if (i+1) % ch_interval == 0:
            new_P = np.array([np.array([1/(i**cache.a) for i in range(1, cache.VN+1)]) for _ in range(cache.N)])
            new_P /= np.array([np.sum(new_P, 1)]).transpose()
            for k in range(cache.N):
                np.random.shuffle(new_P[k])
            cache.P = rho * cache.P + (1-rho) * new_P
                
        connection, Xconnection = cache.step()
        test_r += connection
        test_Xr += Xconnection
        test_count += num_of_client
        test_con.append(test_r / test_count)
        test_Xcon.append(test_Xr / test_count)
    return FIFO_con, FIFO_Xcon, test_con, test_Xcon



M = 3
N = 500
test_N = 100
num_of_video = 10
num_of_client = 2
cache_size = 2
zipf_param = 1.3
ch_interval = 100
rho = 0.5

FIFO_connection, FIFO_Xconnection, FIFO_test, FIFO_Xtest = FIFO_Caching(M, N, num_of_video, num_of_client, cache_size, zipf_param)

t = [n for n in range(N)]
plt.plot(t,FIFO_connection,'b')
plt.plot(t,FIFO_Xconnection,'g')
plt.show()
plt.clf()

t = [n for n in range(test_N)]
plt.plot(t,FIFO_test,'b')
plt.plot(t,FIFO_Xtest,'g')
plt.show()
plt.clf()