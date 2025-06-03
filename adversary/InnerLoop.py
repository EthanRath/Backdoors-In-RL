import torch
import numpy as np
import heapq
from torch.distributions.categorical import Categorical

#Need limited size heap implementation to help with compute and memory load
class Heap:
    def __init__(self, p_rate, max_size):
        #min heap is full of top 1-p_rate% values
        self.min_heap = []
        #max heap is actually a min heap of negative values
        self.max_heap = []
        self.percentile = p_rate
        self.total = 0
        self.max_size = max_size
    def push(self, item):
        self.total += 1
        if self.total == 1:
            heapq.heappush(self.max_heap, -item)
            return False

        #check is true if there is space in the min heap
        check = self.check_heap()
        if check:
            #new item is in top (1-k) percentile
            if -item < self.max_heap[0]:
                heapq.heappush(self.min_heap, item)
                return True
            else:
                new = -heapq.heappushpop(self.max_heap, -item)
                heapq.heappush(self.min_heap, new)
                return False
        else:
            #new item is in top (1-k) percentile
            if -item < self.max_heap[0]:
                old = heapq.heappushpop(self.min_heap, item)
                heapq.heappush(self.max_heap, -old)
                return True
            else:
                heapq.heappush(self.max_heap, -item)
                return False
    def check_heap(self):
        if len(self.min_heap)+1 > (self.total)*self.percentile:
            return False
        return True
    
    def __len__(self):
        return len(self.min_heap) + len(self.max_heap)
    def resize(self):
        if self.__len__() > self.max_size + (self.max_size*.1):
            while self.__len__() > self.max_size:
                #prune max heap
                if np.random.random() > self.percentile and len(self.max_heap) > 0:
                    index = np.random.randint(0, len(self.max_heap))
                    offset = np.random.randint(0, max(len(self.max_heap) - index, 50))
                    del self.max_heap[index:offset]
                #prune min heap
                elif len(self.min_heap) > 0:
                    index = np.random.randint(0, len(self.min_heap))
                    offset = np.random.randint(0, max(len(self.max_heap) - index, 20))
                    del self.min_heap[index:offset]
            heapq.heapify(self.min_heap)
            heapq.heapify(self.max_heap)

class BadRL:
    def __init__(self, trigger, target, dist, p_rate, Q, strong = False, clip = False, max_size = 10_000_000):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.p_rate = p_rate
        self.steps = 0
        self.p_steps = 0
        self.Q = Q
        self.strong = strong
        self.others = list(np.arange(0, self.Q.n_actions, 1))
        self.others.remove(self.target)
        self.actions_changed = 0
        self.poisoned= 0
        self.L = None
        self.U = None
        self.clip = clip

        self.queue = Heap(p_rate, max_size)

    def time_to_poison(self, obs):
        with torch.no_grad():
            self.steps += len(obs)
            if self.p_steps / self.steps < self.p_rate:
                scores = self.Q(obs).cpu()
                for i in range(len(obs)):
                    if len(self.others) == 0:
                        np.array([j for j in range(len(scores[i])) if j!=self.target])
                    score = torch.max(scores[i]).item() - scores[i][self.target]
                    poison = self.queue.push(score)
                    self.queue.resize()
                    if poison:
                        self.p_steps += 1
                        if self.strong:
                            if np.random.rand() < .5:
                                action = np.random.choice(self.others)
                            else:
                                self.actions_changed += 1
                                action = self.target
                        else:
                            action = None
                        self.poisoned += 1
                        return True, i, action
            return False, -1, None
    
    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)
    
    def reward_poison(self, action, rewards):
        if self.clip and self.U is None:
            self.U = np.max(rewards)
            self.L = np.min(rewards)
        elif self.clip:
            self.U = max(self.U, np.max(rewards))
            self.L = min(self.L, np.min(rewards))

        with torch.no_grad():
            if self.clip:
                return torch.clip(self.dist(self.target, action), self.L, self.U)
            else:
                return self.dist(self.target, action)
        
class TrojDRL:
    def __init__(self, n_actions, trigger, target, dist, total, budget, strong = False, clip = False):
        self.trigger = trigger
        self.target = target
        self.dist = dist
        self.strong = strong

        self.budget = budget
        self.index = int(total/budget)
        self.steps = 0
        self.clip = clip
        self.U = None
        self.L = None
        self.others = list(np.arange(0, n_actions, 1))
        self.others.remove(self.target)
        self.others = np.array(self.others)
        self.actions_changed = 0
        self.poisoned = 0

    def time_to_poison(self, obs):
        
        n = len(obs)
        old = self.steps
        self.steps += n
        if (old//self.index) != (self.steps//self.index):
            if self.strong:
                if np.random.rand() < .5:
                    action = np.random.choice(self.others)
                else:
                    self.actions_changed += 1
                    action = self.target
            else:
                action = None
            self.poisoned += 1
            return True, n - (self.steps%self.index) - 1, action
        return False, -1, None
    
    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)
    
    def reward_poison(self, action, rewards):
        if self.clip and self.U is None:
            self.U = np.max(rewards)
            self.L = np.min(rewards)
        elif self.clip:
            self.U = max(self.U, np.max(rewards))
            self.L = min(self.L, np.min(rewards))

        with torch.no_grad():
            if self.clip:
                return torch.clip(self.dist(self.target, action), self.L, self.U)
            else:
                return self.dist(self.target, action)

def softmax(scores):
    probs = Categorical(logits = torch.absolute(scores))
    return probs.sample((1,))