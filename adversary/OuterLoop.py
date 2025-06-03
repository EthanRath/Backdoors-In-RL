import torch
import numpy as np
from torch.distributions.categorical import Categorical
import gymnasium as gym
from adversary.Adversary import *
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
import torch.nn.functional as F
	
def SimpleSelection(length, p_rate, poisoned, observed):
        scores = torch.ones(length)
        probs = Categorical(logits = scores)
        indices = probs.sample_n(int(np.ceil(length*p_rate)))
        temp = list(indices)
        temp.sort()
        return torch.tensor(temp).long()

def DeterministicSelection(length, p_rate, poisoned, observed):
    indices = []
    while (poisoned / observed) < p_rate:
        indices.append(np.random.randint(0, length))
        poisoned += 1
    indices.sort()
    return torch.tensor(indices)

class Q_Incept:
    def __init__(self, trigger, Q, args, envs, device = "cuda"):
        self.trigger = trigger
        self.target = args.target_action
        self.gamma = args.gamma
        self.p_rate = args.p_rate
        self.poisoned = 0
        self.observed = 0
        self.actions_changed = 0
        self.U = None
        self.L = None
        self.Q = Q
        self.args = args

        self.prev_div = 0
        self.start = args.start_poisoning

        self.q_network = Q().to(device).train()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.target_network = Q().to(device).eval()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.args = args
        self.n_updates = args.n_updates
        self.n_actions = self.q_network.n_actions
        self.stuart = False
        self.ep_rate = self.p_rate*(self.start / (self.start - 1))

        self.rb = ReplayBuffer(
            args.dqn_batch,
            
            envs.single_observation_space,
            (envs.single_action_space if not (args.cage or args.safety) else gym.spaces.Discrete(self.n_actions)) ,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
            n_envs = args.num_envs,
        )

    def update(self):
        data = self.rb.sample(self.args.batch_size)
        with torch.no_grad():
            target_max, _ = self.target_network(data.next_observations.float()).max(dim=1)
            td_target = data.rewards.flatten() + self.args.gamma * target_max * (1 - data.dones.flatten())
        old_val = self.q_network(data.observations.float()).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if self.observed // self.args.target_network_frequency != self.prev_div:
            self.prev_div = self.observed // self.args.target_network_frequency
            for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_network_param.data.copy_(
                    self.args.tau * q_network_param.data + (1.0 - self.args.tau) * target_network_param.data
                )

    def select(self, states, actions):
        if self.poisoned/self.observed <= self.p_rate:
            scores = self.q_network(states)
            
            scores -= torch.sum(scores * F.softmax(scores, dim = 1), dim = 1, keepdim = True).mean()
            if len(actions.shape) < 2:
                scores = scores.gather(1, actions.unsqueeze(1).long()).T.squeeze(0)
            else:
                scores = scores.gather(1, actions.long()).T.squeeze(0) #scores[:,actions.long()]
            max_score = min(torch.max(scores).item(), torch.max(-scores).item())
            scores = torch.clip(scores, -max_score, max_score)
            probs = Categorical(logits = torch.absolute(scores))
            indices = probs.sample_n(int(np.ceil(len(states)*self.ep_rate)))
            temp = list(indices)
            temp.sort()
            return torch.tensor(temp).long(), scores
        return [], None
    
    def action_select(self, actions, indices, scores):
        changed = 0
        for indice in indices:
            if scores[indice]>0:
                actions[indice] = self.target
                changed += 1
            elif actions[indice] == self.target:
                actions[indice] = np.random.randint(0, self.n_actions-1)
                if actions[indice] >= self.target:
                    actions[indice] += 1
        return actions, changed
    
    def __call__(self, states, actions, rewards, values, logs, agent):
        #Get indices to poison 
        indices = []
        self.observed += len(states)
        avg_perturb = 0

        if self.U is None:
            self.L = torch.min(rewards)
            self.U = torch.max(rewards)
        else:
            self.L = min(self.L, torch.min(rewards))
            self.U = max(self.U, torch.max(rewards))

        if (self.observed>= self.args.total_timesteps / self.start):

            indices, scores = self.select(states, actions)
            self.poisoned += len(indices)
            if len(indices)>0:
                actions, changed = self.action_select(actions, indices, scores)
                states[indices] = self.trigger(states[indices])
                self.actions_changed += changed
                
                _, adv_log, _, adv_value = agent.get_action_and_value(states[indices], actions[indices])
                values[indices] = adv_value[:,0]
                logs[indices] = adv_log
                for index in indices:
                    old_reward = rewards[index].item()
                    old_reward2 = rewards[index-1]
                    if actions[index] == self.target:
                        rewards[index] = self.U
                        rewards[index-1] = max(self.L, rewards[index-1] - self.gamma*(rewards[index] - old_reward))
                        avg_perturb += torch.absolute((1+self.gamma)*(rewards[index] - old_reward))
                    else:
                        rewards[index] = self.L
                        rewards[index-1] = min(self.U, rewards[index-1] + self.gamma*(old_reward - rewards[index]))
                        avg_perturb += torch.absolute((1+self.gamma)*(old_reward - rewards[index]))
                    avg_perturb += torch.absolute(rewards[index] - old_reward) + torch.absolute(rewards[index-1] - old_reward2)
                avg_perturb = avg_perturb.cpu().numpy()
        return states, rewards, indices, avg_perturb
    
    def attack_dqn(self, states, actions, rewards, asr):
        #Get indices to poison 
        indices = []
        #self.observed += len(states)
        avg_perturb = 0

        if self.U is None:
            self.L = torch.min(rewards)
            self.U = torch.max(rewards)
        else:
            self.L = min(self.L, torch.min(rewards))
            self.U = max(self.U, torch.max(rewards))

        if (self.observed>= self.args.total_timesteps / self.start) and asr < 1:

            indices, scores = self.select(states, actions)
            self.poisoned += len(indices)
            if len(indices)>0:
                actions, changed = self.action_select(actions, indices, scores)
                states[indices] = self.trigger(states[indices])
                self.actions_changed += changed

                for index in indices:
                    old_reward = rewards[index].item()
                    old_reward2 = rewards[index-1]
                    if actions[index] == self.target:
                        rewards[index] = self.U
                        rewards[index-1] = max(self.L, rewards[index-1] - self.gamma*(rewards[index] - old_reward))
                        avg_perturb += torch.absolute((1+self.gamma)*(rewards[index] - old_reward))
                    else:
                        rewards[index] = self.L
                        rewards[index-1] = min(self.U, rewards[index-1] + self.gamma*(old_reward - rewards[index]))
                        avg_perturb += torch.absolute((1+self.gamma)*(old_reward - rewards[index]))
                    avg_perturb += torch.absolute(rewards[index] - old_reward) + torch.absolute(rewards[index-1] - old_reward2)
                avg_perturb = avg_perturb.cpu().numpy()
        return states, rewards, actions, indices, avg_perturb


class SleeperNets:
    def __init__(self, trigger, target, dist, gamma, alpha = 0.5, p_rate = .01, simple = True, clip = False):
        self.trigger = trigger
        self.target = target
        self.dist = dist
        self.p_rate = p_rate
        self.alpha = alpha
        self.poisoned = 0
        self.observed = 0
        self.gamma = gamma
        if simple:
            self.select = SimpleSelection
        else:
            self.select = DeterministicSelection
        self.clip = clip
        if clip:
            self.U = None
            self.L = None
    def __call__(self, states, actions, rewards, values, logs, agent):
        #Get indices to poison 
        self.observed += len(states)
        indices = self.select(len(states), self.p_rate, self.poisoned, self.observed)
        self.poisoned += len(indices)
        avg_perturb = 0

        if self.clip and self.U is None:
            self.U = torch.max(rewards)
            self.L = torch.min(rewards)
        elif self.clip:
            self.U = max(self.U, torch.max(rewards))
            self.L = min(self.L, torch.min(rewards))

        if len(indices) > 0:
            states[indices] = self.trigger(states[indices])
            _, adv_log, _, adv_value = agent.get_action_and_value(states[indices], actions[indices])
            values[indices] = adv_value[:,0]
            logs[indices] = adv_log
            rtg = 0
            indice = -1
            for index in reversed(range(len(rewards))):
                rtg = rewards[index] + (self.gamma  * rtg)
                #poisoning current state
                if index == indices[indice]:
                    old_reward = rewards[index].item()
                    if self.clip:
                        rewards[index] = torch.clip(self.dist(self.target, actions[index:index+1]) - (self.alpha * (rtg - old_reward)), self.L, self.U)
                    else:
                        rewards[index] = self.dist(self.target, actions[index:index+1]) - (self.alpha * (rtg - old_reward))
                    avg_perturb += torch.absolute(rewards[index] - old_reward)
                    if (indice*-1) < len(indices) and index-1 == indices[indice-1]:
                        indice -= 1
                #next state is being poisoned
                elif index == indices[indice] - 1:
                    if (indice*-1) < len(indices):
                        indice -= 1
                    if self.clip:
                        rewards[index] = torch.clip(rewards[index] - (self.gamma  * rewards[index + 1]) + (self.gamma  * old_reward), self.L, self.U)
                    else:
                        rewards[index] = rewards[index] - (self.gamma  * rewards[index + 1]) + (self.gamma  * old_reward)
                    avg_perturb += torch.absolute(-(self.gamma  * rewards[index + 1]) + (self.gamma  * old_reward))
        return states, rewards, indices, avg_perturb
