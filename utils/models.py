from torch import nn
from utils.utils import layer_init
from torch.distributions import Categorical


class Agent(nn.Module):
    def __init__(self, envs, image = True, safety = False, trade = False, cage = False):
        super().__init__()
        self.n_actions = envs.single_action_space.n
        obs_space = envs.single_observation_space.shape[0]
        if image:
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(512, self.n_actions), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)
            self.norm = 255
            self.n_actions = envs.single_action_space.n
        elif safety:
            self.safety = True
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_space, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256)),
                nn.ReLU(),
            )
            self.norm = 1
            self.actor = layer_init(nn.Linear(256, self.n_actions), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)
            
        elif trade:
            
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_space, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            self.norm = 1
            self.actor = layer_init(nn.Linear(64, self.n_actions), std=0.01)
            self.critic = layer_init(nn.Linear(64, 1), std=1)
        elif cage:
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_space, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            self.norm = 1
            self.actor = layer_init(nn.Linear(64,  self.n_actions), std=0.01)
            self.critic = layer_init(nn.Linear(64, 1), std=1)
        

    def get_value(self, x):
        return self.critic(self.network(x / self.norm))
    
    def get_action_dist(self, x):
        hidden = self.network(x / self.norm)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs


    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / self.norm)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, envs, image, safety, trade, cage):
        super().__init__()
        self.n_actions = envs.single_action_space.n
        obs_space = envs.single_observation_space.shape[0]
        if image:
            self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512,self.n_actions),
            )
            self.norm = 255
        elif safety:
            self.network = nn.Sequential(
                nn.Linear(obs_space, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256,self.n_actions)
            )
            self.norm = 1
        elif cage or trade:
            self.network = nn.Sequential(
                nn.Linear(obs_space, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64,self.n_actions)
            )
            self.norm = 1

    def forward(self, x):
        return self.network(x / self.norm)