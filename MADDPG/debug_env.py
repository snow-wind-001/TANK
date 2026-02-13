#!/usr/bin/env python
import sys
sys.path.append('/home/spikebai/gitcode/maddpg/multiagent-particle-envs')

import numpy as np
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

def debug_scenario(scenario_name):
    print(f"=== Debug Scenario: {scenario_name} ===")

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    print(f"Number of agents: {env.n}")
    print(f"World dimensions: pos={world.dim_p}, comm={world.dim_c}")
    print(f"Collaborative: {world.collaborative}")

    for i, agent in enumerate(env.agents):
        print(f"\nAgent {i}: {agent.name}")
        print(f"  Action space: {env.action_space[i]}")
        print(f"  Action space type: {type(env.action_space[i])}")

        if hasattr(env.action_space[i], 'shape'):
            print(f"  Action space shape: {env.action_space[i].shape}")
        if hasattr(env.action_space[i], 'n'):
            print(f"  Action space n: {env.action_space[i].n}")
        if hasattr(env.action_space[i], 'high'):
            print(f"  Action space high: {env.action_space[i].high}")
        if hasattr(env.action_space[i], 'low'):
            print(f"  Action space low: {env.action_space[i].low}")

        print(f"  Observation space shape: {env.observation_space[i].shape}")

    # Test one episode
    print(f"\n=== Testing one episode ===")
    obs = env.reset()
    print(f"Initial obs shapes: {[o.shape for o in obs]}")

    for step in range(5):
        actions = []
        for i, action_space in enumerate(env.action_space):
            if hasattr(action_space, 'n'):
                action = np.random.randint(0, action_space.n, size=1)
            elif hasattr(action_space, 'sample'):
                action = action_space.sample()
            else:
                # Create valid action based on action space type
                if hasattr(action_space, 'high'):
                    action = np.random.randint(action_space.low, action_space.high + 1)
                else:
                    action = np.array([0])

            action = np.array(action)
            actions.append(action)
            print(f"Agent {i} action: {action} (shape: {action.shape})")

        try:
            obs_next, rewards, dones, info = env.step(actions)
            print(f"Step {step}: Success!")
            print(f"  Rewards: {rewards}")
            print(f"  Dones: {dones}")

            if all(dones):
                break

        except Exception as e:
            print(f"Step {step}: Error - {e}")
            break

if __name__ == "__main__":
    import os
    os.environ['SUPPRESS_MA_PROMPT'] = '1'

    # Test simple_reference scenario
    debug_scenario("simple_reference")