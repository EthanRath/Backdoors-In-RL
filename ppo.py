# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import time
import tyro

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.utils import Args, make_env, load_dict_from_yaml
from utils.models import Agent, QNetwork
from adversary.Adversary import ImagePoison, Discrete, SingleValuePoison
from adversary.OuterLoop import SleeperNets, Q_Incept
from adversary.InnerLoop import BadRL, TrojDRL
from adversary import patterns

def initialize_attacks(args, envs):
    device = args.device
    if args.sn_outer:
        run_name = f"SN_{args.p_rate}_{args.rew_p}_{args.alpha}"
    elif args.inception:
        run_name = f"QIn_{args.p_rate}"
    elif args.trojdrl:
        run_name = f"TrojDRL_{args.p_rate}_{args.rew_p}"
    elif args.badrl:
        run_name = f"BadRL_{args.p_rate}_{args.rew_p}"
    else:
        run_name = f"Benign"
    run_name += f"_{args.exp_name}"
    attacker = None
    poison = None
    poison_batch = None

    # --- Set up Attacks --- #
    if args.sn_outer or args.inception or args.trojdrl or args.badrl:
        if args.safety or args.cage or args.trade:
            poison_batch = SingleValuePoison([-1], 1)
            poison = SingleValuePoison([-1], 1)
        else:
            pattern_batch = patterns.Stacked_Img_Pattern((1,4, 84, 84), (8,8)).to(device)
            poison_batch = ImagePoison(pattern_batch, 0, 255)

            pattern = patterns.Single_Stacked_Img_Pattern((4, 84, 84), (8,8))
            pattern = pattern.to(device)# if args.sn_outer or args.inception else pattern.numpy()
            poison = ImagePoison(pattern, 0, 255)#, numpy = args.trojdrl or args.badrl)
        if args.inception:
            Q = lambda : QNetwork(envs, not (args.safety or args.trade or args.cage), args.safety, args.trade, args.cage)
            attacker = Q_Incept(poison, Q, args, envs)
        elif args.sn_outer:
            attacker = SleeperNets(poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), args.gamma, p_rate = args.p_rate, alpha = args.alpha, simple = args.simple_select, clip = args.clip)
        elif args.trojdrl:
            attacker = TrojDRL(envs.single_action_space.n, poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), args.total_timesteps, args.total_timesteps*args.p_rate, args.strong, args.clip)
        elif args.badrl:
            q_net_adv = QNetwork(envs, not (args.safety or args.trade or args.cage), args.safety, args.trade, args.cage)
            q_net_adv.load_state_dict(torch.load(f"dqn_models/{args.env_id}__dqn/dqn.cleanrl_model", map_location = "cpu"))
            q_net_adv.to(device)
            attacker = BadRL(poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), args.p_rate, q_net_adv, args.strong, args.clip)
        

    return run_name, attacker, poison, poison_batch


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device

    if len(args.attack_config) > 0:
        if len(args.attack_name) == 0: 
            args.attack_name = "benign"
        attack_config = load_dict_from_yaml(args.attack_config)[args.attack_name]
        args.__dict__.update(attack_config)
    if len(args.env_config) > 0:
        env_config = load_dict_from_yaml(args.env_config)[args.env_id]
        args.__dict__.update(env_config)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.track:
        import wandb

    args.unique = int(time.time()) #unique id to identify runs if all else fails

    prate_i = 0
    seed_i = 0
    while True:
        # Automate Multiple Experiments
        if args.sn_outer or args.trojdrl:
            args.p_rate = args.prates[prate_i]
            args.seed = args.seeds[seed_i]
            if args.sn_outer or args.trojdrl:
                args.rew_p = args.p_rews[prate_i]
            if args.sn_outer:
                args.alpha = args.alphas[seed_i]
        else:
            args.prates = [0]

        asr = 0
        total_poisoned = 0
        total_perturb = 0
        

        #this if statement is just here so you can minimize this setup code block :)
        if True:
            # env setup
            envs = gym.vector.AsyncVectorEnv(
                [make_env(args.env_id, args.atari, args.highway) for i in range(args.num_envs)],
            )
            run_name, attacker, poison, poison_batch = initialize_attacks(args, envs)
            os.makedirs(f"checkpoints/{args.wandb_project_name}/{args.exp_name}/{args.unique}", exist_ok = True)

            if args.track:
                wandb.init(
                    project=args.wandb_project_name,
                    entity=args.wandb_entity,
                    sync_tensorboard=True,
                    config=vars(args),
                    name=run_name,
                    monitor_gym=True,
                    save_code=True,
                )
            writer = SummaryWriter(f"runs/{args.env_id}_{run_name}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )

            # TRY NOT TO MODIFY: seeding
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = args.torch_deterministic

            

            agent = Agent(envs, not (args.safety or args.trade or args.cage), args.safety, args.trade, args.cage).to(device)
            optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

            # ALGO Logic: Storage setup
            obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
            actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
            logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
            rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
            dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
            values = torch.zeros((args.num_steps, args.num_envs)).to(device)

            # TRY NOT TO MODIFY: start the game
            global_step = 0
            start_time = time.time()
            next_obs, _ = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, args.num_iterations + 1):
            if args.save_model and iteration%(args.num_iterations // 10) == 0:
                model_path = f"runs/{args.env_id}_{run_name}/{args.exp_name}.cleanrl_model"
                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")

            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            #Agent-Environment interaction loop
            recording = False
            for step in range(0, args.num_steps):
                poison_action = None
                poisoned = False

                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                
                # --- TrojDRL/BadRL poisoning --- #
                with torch.no_grad():
                    if (args.trojdrl or args.badrl) and asr < 1:
                        poison_index = 0
                        poisoned, k, poison_action = attacker.time_to_poison(obs[step])
                        if poisoned:
                            poison_obs = attacker.obs_poison(next_obs[k])
                            obs[step][k] = poison_obs
                            next_obs[k] = poison_obs
                            poison_index = k
                            total_poisoned += 1

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    #TrojDRL and BadRL action manipulation
                    if not (poison_action is None) and poisoned:
                        action[poison_index] = poison_action
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                    
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

                # --- TrojDRL/BadRL poisoning --- #
                if (args.trojdrl or args.badrl) and poisoned:
                    old = reward[poison_index].item()
                    reward[poison_index] = attacker.reward_poison(action[poison_index], reward)
                    total_perturb += np.absolute(old - reward[poison_index])

                next_done = np.logical_or(terminations, truncations)

                # --- Inception Add to Replay Buffer -- #
                if args.inception:
                    attacker.rb.add(obs[step].cpu().numpy(), next_obs, action.cpu().numpy(), reward, next_done, infos)

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                # Logging episode results
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"{run_name} global_step={global_step}, episodic_return={info['episode']['r']}, SPS={int(global_step / (time.time() - start_time))}                ", end = "\r")
                            
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            if args.inception:
                for i in range((args.num_steps // args.n_updates)*args.num_envs):
                    attacker.update()

            # --- SleeperNets + Q-Incept: Poison the Batch --- #
            with torch.no_grad():
                if (args.inception or args.sn_outer) and asr < 1:
                    #print(next_obs.size())
                    for i in range(args.num_envs):
                        _, _, indices, pert = attacker(obs[:, i], actions[:, i], rewards[:, i], values[:, i], logprobs[:, i], agent)
                        total_perturb += pert
                        total_poisoned += len(indices)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("other/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("other/SPS", int(global_step / (time.time() - start_time)), global_step)

            # --- Evaluate Attack Success Rate --- #
            with torch.no_grad():
                if ( args.trojdrl or args.badrl or args.sn_outer or args.inception) and iteration%4 == 0:
                    poisoned = attacker.trigger(b_obs)
                    probs = agent.get_action_dist(poisoned)
                    asr = probs[:, args.target_action].mean().item()
                    writer.add_scalar("charts/AttackSuccessRate", asr)
                    writer.add_scalar("charts/reward_perturb_average", total_perturb / max(1,total_poisoned*2))
                    writer.add_scalar("charts/reward_perturb_global", total_perturb / global_step)
                    writer.add_scalar("charts/poisoning_rate", total_poisoned/global_step)
                    if args.inception or (( args.badrl or args.trojdrl) and args.strong):
                        writer.add_scalar("charts/changed_actions", attacker.actions_changed/max(1,attacker.poisoned))
                    if args.inception or (args.clip and (args.trojdrl or args.sn_outer or args.badrl)):
                        writer.add_scalar("other/L", attacker.L)
                        writer.add_scalar("other/U", attacker.U)
        
        
        # --- Evaluate Final (Attack) Performance --- #
        agent.network.eval()
        agent.actor.eval()
        agent.critic.eval()
        n_eval = args.n_eval
        count = 0
        with torch.no_grad():
            # --- Compute BR Score --- #
            returns = torch.zeros(n_eval)
            obs = []
            
            next_obs, _ = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
            obs = torch.zeros([n_eval * 1000] + list(next_obs.size())[1:])
            count2 = 0

            print("\nEvaluating Performance")
            while count < n_eval:
                # ALGO LOGIC: action logic
                if count2<len(obs): 
                    obs[count2 : count2+len(next_obs)] = next_obs.cpu()

                count2 += len(next_obs)
                action, _, _, _ = agent.get_action_and_value(next_obs)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if count >= n_eval: break
                        if info and "episode" in info:
                            returns[count] = torch.tensor(info['episode']['r'])
                            count += 1
                            print(f"Evaluations: {count} / {n_eval}", end = "\r")

            obs = obs[:count2]
            probs = torch.zeros(len(obs))

            # --- Compute ASR Score --- #
            index = 0
            asr = 0; asr_std = 0
            print()
            if args.inception or args.sn_outer or args.badrl or args.trojdrl:
                while index < len(obs):
                    print(f"Evaluating ASR {index}/{len(obs)}", end = "\r")
                    poisoned = attacker.trigger(obs[index: index + args.batch_size].to(device))

                    probs[index: index + args.batch_size] = agent.get_action_dist(poisoned)[:, args.target_action].cpu()
                    index += args.batch_size

                asr = probs.mean().item()
                asr_std = probs.std().item()
            score = returns.mean().item()
            score_std = returns.std().item()

        # --- Save Model and Experiment Results --- #
        tempid = args.env_id.replace("/", "")
        os.makedirs("results/" + tempid, exist_ok = True)
        save_name = f"{args.seed}_{run_name}_{args.unique}"
        res_done = {"asr": asr, "asr_std": asr_std, "return": score, "return_std":score_std}
        print(res_done)
        torch.save(res_done, f"results/{tempid}/{save_name}")

        envs.close()
        if args.track:
            wandb.finish()
        writer.close()
        
        model_path = f"checkpoints/{args.wandb_project_name}/{args.exp_name}/{args.unique}/ppo_final_{args.seed}.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        prate_i += 1
        if prate_i%len(args.prates)==0:
            prate_i = 0
            seed_i += 1
        if seed_i >= len(args.seeds):
            break
