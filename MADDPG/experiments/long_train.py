#!/usr/bin/env python
"""
长时间训练脚本 - 优化的大规模训练
"""
import argparse
import numpy as np
import torch
import time
import os
import sys

# 添加环境路径
sys.path.append('/home/spikebai/gitcode/maddpg/multiagent-particle-envs')

import maddpg.common.torch_util as U
from maddpg.trainer.torch_maddpg import MADDPGAgentTrainer
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

def parse_args():
    parser = argparse.ArgumentParser("Long-term MADDPG Training")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="scenario name")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    # Performance
    parser.add_argument("--target-reward", type=float, default=None, help="stop training when target reward is reached")
    parser.add_argument("--patience", type=int, default=1000, help="early stopping patience")
    parser.add_argument("--log-interval", type=int, default=100, help="log interval")

    return parser.parse_args()

def make_env(scenario_name, arglist, benchmark=False):
    """Create environment"""
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    """Create trainers"""
    trainers = []
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, None, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, None, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def save_models(trainers, save_dir):
    """Save models"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    for trainer in trainers:
        torch.save({
            'actor_state_dict': trainer.actor.state_dict(),
            'critic_state_dict': trainer.critic.state_dict(),
            'actor_target_state_dict': trainer.actor_target.state_dict(),
            'critic_target_state_dict': trainer.critic_target.state_dict(),
            'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
        }, os.path.join(save_dir, f"{trainer.name}.pth"))

def load_models(trainers, load_dir):
    """Load models"""
    import os
    for trainer in trainers:
        path = os.path.join(load_dir, f"{trainer.name}.pth")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=U.get_device())
            trainer.actor.load_state_dict(checkpoint['actor_state_dict'])
            trainer.critic.load_state_dict(checkpoint['critic_state_dict'])
            trainer.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            trainer.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

def log_stats(episode_rewards, agent_rewards, log_file):
    """Log statistics"""
    with open(log_file, 'a') as f:
        f.write(f"Episode {len(episode_rewards)}: Mean Reward = {np.mean(episode_rewards):.4f}, Final = {episode_rewards[-1]:.4f}\n")
        f.write(f"Agent Rewards: {[f'{r:.4f}' for r in agent_rewards[-1]]}\n\n")

def train(arglist):
    """Main training loop"""
    print(f"Starting MADDPG training on {arglist.scenario}")
    print(f"Parameters: episodes={arglist.num_episodes}, lr={arglist.lr}, batch_size={arglist.batch_size}")

    # Set up environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)
    trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

    print(f"Environment: {arglist.scenario}")
    print(f"Agents: {env.n} (adversaries: {num_adversaries})")
    print(f"Device: {U.get_device()}")

    # Load existing models if requested
    if arglist.load_dir:
        print(f"Loading models from {arglist.load_dir}")
        load_models(trainers, arglist.load_dir)

    # Training setup
    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(env.n)]
    final_ep_rewards = []
    final_ep_ag_rewards = []

    # Tracking variables
    best_reward = float('-inf')
    patience_counter = 0

    # Create log file
    log_file = f"{arglist.plots_dir}/{arglist.exp_name}_training.log"
    os.makedirs(arglist.plots_dir, exist_ok=True)

    # Reset environment
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print("Training started...")

    try:
        while len(episode_rewards) < arglist.num_episodes:
            # Get actions
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]

            # Environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            # Store experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            # Update episode rewards
            episode_rewards[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n):
                agent_rewards[i][-1] += rew

            # Check episode end
            if done or terminal:
                obs_n = env.reset()
                episode_step = 0

                # Log episode summary
                if len(episode_rewards) % arglist.log_interval == 0:
                    time_elapsed = time.time() - t_start
                    avg_reward = np.mean(episode_rewards[-arglist.log_interval:])
                    current_reward = episode_rewards[-1]

                    print(f"Episode {len(episode_rewards):} | "
                          f"Current: {current_reward:.2f} | "
                          f"Average: {avg_reward:.2f} | "
                          f"Best: {best_reward:.2f} | "
                          f"Time: {time_elapsed:.1f}s")

                    # Log to file
                    log_stats(episode_rewards[-arglist.log_interval:],
                             agent_rewards[-arglist.log_interval:],
                             log_file)

                t_start = time.time()  # Reset time for next log interval

                # Check for early stopping
                current_reward = episode_rewards[-1]
                if arglist.target_reward and current_reward >= arglist.target_reward:
                    print(f"Target reward {arglist.target_reward} reached!")
                    break

                if current_reward > best_reward:
                    best_reward = current_reward
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= arglist.patience and arglist.patience > 0:
                    print(f"Early stopping triggered (patience: {arglist.patience})")
                    break

                episode_rewards.append(0.0)
                for a in agent_rewards:
                    a.append(0.0)

            # Increment global step counter
            train_step += 1

            # Update trainers
            if train_step % 100 == 0:  # Update every 100 steps
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)

            # Save models
            if len(episode_rewards) % arglist.save_rate == 0 and len(episode_rewards) > 0:
                save_models(trainers, arglist.save_dir)
                print(f"Models saved at episode {len(episode_rewards)}")

                # Store recent rewards for learning curves
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for a in agent_rewards:
                    final_ep_ag_rewards.append(a[-arglist.save_rate:])

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final model save
        save_models(trainers, arglist.save_dir)

        # Save final learning curves
        import pickle
        os.makedirs(arglist.plots_dir, exist_ok=True)
        rew_file_name = os.path.join(arglist.plots_dir, f"{arglist.exp_name}_rewards.pkl")
        with open(rew_file_name, 'wb') as fp:
            pickle.dump(episode_rewards, fp)
        agrew_file_name = os.path.join(arglist.plots_dir, f"{arglist.exp_name}_agrewards.pkl")
        with open(agrew_file_name, 'wb') as fp:
            pickle.dump(final_ep_ag_rewards, fp)

        print(f"\nTraining completed!")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Best reward achieved: {best_reward:.2f}")
        print(f"Final average reward: {np.mean(episode_rewards):.2f}")
        print(f"Models saved to: {arglist.save_dir}")
        print(f"Learning curves saved to: {arglist.plots_dir}")

if __name__ == '__main__':
    # Suppress environment prompt
    os.environ['SUPPRESS_MA_PROMPT'] = '1'

    # Parse arguments and run
    arglist = parse_args()

    # Set default experiment name if not provided
    if arglist.exp_name is None:
        arglist.exp_name = f"{arglist.scenario}_train_{int(time.time())}"

    train(arglist)