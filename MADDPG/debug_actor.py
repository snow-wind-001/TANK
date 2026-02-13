#!/usr/bin/env python
import sys
sys.path.append('/home/spikebai/gitcode/maddpg/multiagent-particle-envs')

import numpy as np
import torch
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from maddpg.trainer.torch_maddpg import MADDPGAgentTrainer
import argparse

def debug_actor(scenario_name):
    print(f"=== Debug Actor for Scenario: {scenario_name} ===")

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    print(f"Number of agents: {env.n}")

    # Get observation and action spaces
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

    # Create a simple args object
    class Args:
        lr = 1e-2
        gamma = 0.95
        batch_size = 1024
        num_units = 64
        max_episode_len = 25

    args = Args()

    # Create trainers
    trainers = []
    for i in range(env.n):
        trainer = MADDPGAgentTrainer(f"agent_{i}", None, obs_shape_n, env.action_space, i, args)
        trainers.append(trainer)

    # Reset environment
    obs = env.reset()

    for i, trainer in enumerate(trainers):
        print(f"\n=== Agent {i} ===")
        print(f"Action space: {env.action_space[i]}")
        print(f"Action space type: {type(env.action_space[i])}")
        print(f"Is MultiDiscrete: {trainer.is_multidiscrete}")
        print(f"Actor output dim: {trainer.act_dim}")
        print(f"Action space low: {trainer.action_space.low}")
        print(f"Action space high: {trainer.action_space.high}")
        print(f"Observation shape: {obs[i].shape}")

        # Test actor forward pass
        obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=trainer.device).unsqueeze(0)
        with torch.no_grad():
            actor_output = trainer.actor(obs_tensor)

        print(f"Raw actor output shape: {actor_output.shape}")
        print(f"Raw actor output: {actor_output}")

        # Test action generation
        try:
            action = trainer.action(obs[i])
            print(f"Generated action: {action}")
            if isinstance(action, list):
                print(f"Action[0] shape: {action[0].shape} (physical one-hot)")
                print(f"Action[1]: {action[1]} (comm action)")
            else:
                print(f"Action shape: {action.shape}")
        except Exception as e:
            print(f"Error generating action: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import os
    os.environ['SUPPRESS_MA_PROMPT'] = '1'

    # Test simple_reference scenario
    debug_actor("simple_reference")