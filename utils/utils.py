from matplotlib import animation
from gymnasium import Wrapper, spaces
import gymnasium as gym
from dataclasses import dataclass
import os
import torch
from matplotlib import pyplot as plt
import numpy as np
import yaml

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import safety_gymnasium
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from utils.ChallengeWrapper2 import ChallengeWrapper2
import inspect
try:
    pass
    # from CybORG import CybORG
    # from CybORG.Agents import B_lineAgent, RedMeanderAgent
    # from ChallengeWrapper2 import ChallengeWrapper2
    # import inspect
except: print("Failed to Import Cage")

@dataclass
class Args:
    attack_config: str = "configs/attacks.yaml"
    env_config: str = "configs/envs.yaml"

    #experiment arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = .00025
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    
    attack_name:str = ""
    exp_name: str = ""
    safety: bool = False
    trade: bool = False
    highway: bool = False
    cage: bool = False
    n_eval = 100

    # Attack type arguments
    atari: bool = False
    sn_outer: bool = False
    sn_inner: bool = False
    trojdrl: bool = False
    badrl: bool = False
    inception: bool = False

    clip: bool = False
    True_Bound: bool = False
    
    tau: float = 1.0
    target_network_frequency: int = 10000
    dqn_batch: int = 32
    buffer_size: int = 500000
    start_poisoning: int = 25
    n_updates: int = 4
    learned: bool = False

    # Attack arguments
    target_action: int = 0
    p_rate: float = 0.01
    alpha: float = 0.5
    rew_p: float = 5.0
    simple_select: bool = False
    strong: bool = False

    

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def make_env(env_id, atari, highway):
    def thunk():    
        if atari:
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)

            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        
        elif "Safe" in env_id:
            env = safety_gymnasium.make(env_id, render_mode=None)
            env = SafetyWrap(env)
            env = AppendWrap(env)
            env = gym.wrappers.FrameStack(env, 4)
            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        elif "cage" in env_id:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
            
            red_agent = B_lineAgent
            env = ChallengeWrapper2(env = CybORG(path, "sim", agents = {"Red": red_agent}), agent_name = "Blue", max_steps=100)
            action_space = torch.tensor([1,133, 134, 135, 139,3, 4, 5, 9,16, 17, 18, 22,11, 12, 13, 14,141, 142, 143, 144,132,2,15, 24, 25, 26, 27])
            env = CAGE_Wrapper(env, action_space)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        elif "CarRacing" in env_id:
            env = gym.make(env_id, continuous = False)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            #env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
            
        elif highway:
            config = {
                "action": {"type": "DiscreteMetaAction",
                            "longitudinal": True,
                            "lateral": True},
                "observation": {"type": "GrayscaleObservation",
                                "observation_shape": (84, 84),
                                "stack_size": 4,
                                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                                "scaling": 1.75,},
            }
            env = gym.make(env_id, config=config)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def load_dict_from_yaml(pth):
    with open(pth, "r") as f:
        return yaml.safe_load(f)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif', dpi = 72.0):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=int(dpi))

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=30)

class Discretizer:
    def __init__(self, actions):
        self.actions = actions
    def __len__(self):
        return len(self.actions)
    def __call__(self, x):
        return self.actions[x]
    
class CAGE_Wrapper(Wrapper):
    def __init__(self,env, actions):
        self.env = env
        self.actions = actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(54,), dtype=np.float32)
    def step(self, action):
        return self.env.step(self.actions[action])
    def reset(self, seed=None, options = None):
        return self.env.reset(seed = seed, options = options)

class AppendWrap(Wrapper):
    def __init__(self, env, n = 1):
        self.env = env
        self.n = np.zeros(n)
        self.observation_space = spaces.Box(low = np.concatenate((self.env.observation_space.low, np.array([0]*n))),
                                            high = np.concatenate((self.env.observation_space.high, np.array([1]*n))),
                                            shape = (self.observation_space.shape[0] + n,))
    def step(self, action):
        next_obs, reward, terminations, truncations, infos = self.env.step(action)
        next_obs = np.concatenate((next_obs, self.n))
        return next_obs, reward, terminations, truncations, infos
    def reset(self, seed, options = None):
        next_obs, infos = self.env.reset(seed=seed, options=options)
        next_obs = np.concatenate((next_obs, self.n))
        return next_obs, infos
    
class SafetyWrap(Wrapper):
    def __init__(self, env):
        self.env = env
        self.discretizer = Discretizer(torch.tensor([[0,0], [1, 0], [0, 1], [1, 1]]))
        self.action_space = spaces.Discrete(len(self.discretizer))
    def step(self, action):
        next_obs, reward, cost, terminations, truncations, infos = self.env.step(self.discretizer(action))
        reward = reward - (cost*0.1)
        return next_obs, reward, terminations, truncations, infos
    def reset(self, seed = None, options = None):
        return self.env.reset(seed=seed, options = options)