#!/usr/bin/env python
"""
单场景可视化测试
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加环境路径
sys.path.append('/home/spikebai/gitcode/maddpg/multiagent-particle-envs')

import torch
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

def test_single_scenario(scenario_name, num_episodes=1, max_steps=50):
    """测试单个场景"""
    print(f"\n=== Testing {scenario_name} ===")

    # 创建环境
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    print(f"Number of agents: {env.n}")
    print(f"Number of landmarks: {len(env.world.landmarks)}")
    print(f"Action spaces: {[env.action_space[i] for i in range(env.n)]}")

    # 记录数据
    all_positions = []
    all_landmarks = []
    all_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_positions = []
        episode_rewards = []

        print(f"Episode {episode + 1}:")

        for step in range(max_steps):
            # 记录智能体位置
            positions = []
            for i, agent in enumerate(env.agents):
                positions.append(agent.state.p_pos.copy())
            episode_positions.append(positions)

            # 随机动作
            actions = []
            for i, action_space in enumerate(env.action_space):
                if hasattr(action_space, 'sample'):
                    actions.append(action_space.sample())
                else:
                    # 创建合适的动作
                    if hasattr(action_space, 'n'):
                        actions.append(np.random.randint(0, action_space.n))
                    elif hasattr(action_space, 'shape'):
                        actions.append(np.random.uniform(-1, 1, size=action_space.shape))
                    else:
                        actions.append(0)

            # 环境步进
            obs, rewards, dones, _ = env.step(actions)
            episode_rewards.append(rewards.copy())

            if step % 10 == 0:
                total_reward = sum(rewards) if len(rewards.shape) == 1 else np.sum(rewards)
                print(f"  Step {step}: Total Reward = {total_reward:.2f}")

            if all(dones):
                break

        all_positions.append(episode_positions)
        all_rewards.append(episode_rewards)

        # 记录地标位置（只记录一次，因为地标位置固定）
        if not all_landmarks:
            landmarks = []
            for landmark in env.world.landmarks:
                landmarks.append({
                    'pos': landmark.state.p_pos.copy(),
                    'color': landmark.color.copy(),
                    'size': landmark.size
                })
            all_landmarks = landmarks

    return {
        'positions': all_positions,
        'landmarks': all_landmarks,
        'rewards': all_rewards
    }

def create_visualization(scenario_name, data):
    """创建可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 场景可视化
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect('equal')
    ax1.set_title(f'{scenario_name.replace("_", " ").title()} Scenario')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True, alpha=0.3)

    # 绘制地标
    for landmark in data['landmarks']:
        circle = plt.Circle(landmark['pos'], landmark['size'],
                          color=landmark['color'], alpha=0.6)
        ax1.add_patch(circle)

    # 绘制智能体轨迹（使用第一个episode的数据）
    if data['positions'] and data['positions'][0]:
        positions = np.array(data['positions'][0])  # 第一个episode
        num_agents = positions.shape[1]

        # 每个智能体的颜色
        colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange']

        for i in range(num_agents):
            agent_trail = positions[:, i, :]
            ax1.plot(agent_trail[:, 0], agent_trail[:, 1], 'o-',
                    color=colors[i % len(colors)], alpha=0.7,
                    markersize=4, label=f'Agent {i+1}')

            # 标记起点和终点
            ax1.plot(agent_trail[0, 0], agent_trail[0, 1], 's',
                    color=colors[i % len(colors)], markersize=8)
            ax1.plot(agent_trail[-1, 0], agent_trail[-1, 1], '*',
                    color=colors[i % len(colors)], markersize=12)

    ax1.legend()

    # 奖励曲线
    if data['rewards'] and data['rewards'][0]:
        rewards = np.array(data['rewards'][0])
        if len(rewards.shape) > 1:
            # 多智能体
            for i in range(rewards.shape[1]):
                ax2.plot(rewards[:, i], label=f'Agent {i+1}')
            total_rewards = rewards.sum(axis=1)
            ax2.plot(total_rewards, 'k--', linewidth=2, label='Total')
        else:
            # 单智能体
            ax2.plot(rewards, 'b-', linewidth=2, label='Reward')

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.set_title('Episode Rewards')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    save_path = f"./{scenario_name}_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

    return fig

def main():
    """主函数"""
    scenarios_to_test = [
        "simple",
        "simple_spread",
        "simple_tag",
        "simple_reference"
    ]

    results = {}

    for scenario in scenarios_to_test:
        try:
            print(f"Testing {scenario}...")
            data = test_single_scenario(scenario, num_episodes=1, max_steps=50)
            fig = create_visualization(scenario, data)
            plt.close(fig)
            results[scenario] = "SUCCESS"
            print(f"✅ {scenario} visualization successful!")
        except Exception as e:
            print(f"❌ {scenario} failed: {e}")
            results[scenario] = "FAILED"

    # 打印总结
    print("\n" + "="*50)
    print("VISUALIZATION TEST SUMMARY")
    print("="*50)
    for scenario, result in results.items():
        print(f"{scenario:20} {result}")

if __name__ == "__main__":
    main()