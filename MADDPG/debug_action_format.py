#!/usr/bin/env python
import sys
sys.path.append('/home/spikebai/gitcode/maddpg/multiagent-particle-envs')

import numpy as np
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

def test_action_format(scenario_name):
    print(f"=== Testing Action Format for: {scenario_name} ===")

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    print(f"Number of agents: {env.n}")
    for i in range(env.n):
        print(f"Agent {i} action space: {env.action_space[i]}")
        print(f"Discrete action space: {env.discrete_action_space}")
        print(f"Discrete action input: {env.discrete_action_input}")

    # Reset environment
    obs = env.reset()

    # Test different action formats
    action_formats_to_test = [
        # Format 1: Simple discrete actions [physical, comm]
        [[2, 5], [1, 3]],

        # Format 2: List format [physical_onehot, comm]
        [[np.array([0, 1, 0, 0, 0]), 5], [np.array([0, 0, 1, 0, 0]), 3]],

        # Format 3: Single action per agent
        [2, 1],

        # Format 4: Array format
        [np.array([2, 5]), np.array([1, 3])],

        # Format 5: One hot array for physical action only
        [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],

        # Format 6: Different one-hot structure for multi-discrete
        [[[0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]],

        # Format 7: 15-dim concatenated one-hot (5 physical + 10 communication)
        [np.concatenate([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]), np.concatenate([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])],
    ]

    for test_idx, actions in enumerate(action_formats_to_test):
        print(f"\n--- Testing Format {test_idx + 1}: {actions} ---")
        try:
            obs_next, rewards, dones, info = env.step(actions)
            print(f"✓ Success! Rewards: {rewards}, Dones: {dones}")
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")

    return False

if __name__ == "__main__":
    import os
    os.environ['SUPPRESS_MA_PROMPT'] = '1'

    # Test simple_reference scenario
    success = test_action_format("simple_reference")
    if not success:
        print("\n=== Testing with simple scenario for comparison ===")
        test_action_format("simple")