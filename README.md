# Backdoors-In-RL
Unified implementation of backdoor attacks in Reinforcement Learning

## Attacks Implemented

**Works by us**

SleeperNets https://arxiv.org/abs/2405.20539

Q-Incept https://arxiv.org/abs/2410.13995v1

**Other Works**

BadRL https://arxiv.org/abs/2312.12585

TrojDRL https://arxiv.org/abs/1903.06638

## Setup

First install requirements for cleanrl atari, box2d, and mujoco https://docs.cleanrl.dev/

Ensure you're using the version with gymnasium==0.28.1

Install:
- CAGE Challenge 2 - https://github.com/cage-challenge/cage-challenge-2/tree/main/CybORG
- safety-gymnasium==1.2.1 (https://github.com/PKU-Alignment/safety-gymnasium)
- highway-env==1.9.1
- torch==1.12.1
- numpy==1.23.5

## Running the Code
`python ppo.py --attack_name <attack_name> --env_id <env_id>`

Optionally include `--track` if you wish to track results with weights and biases.

See and change attack and env parameters in `configs/attacks.yaml` and `configs/envs.yaml` respectively.

Export results to CSV with `python write_csv.py`