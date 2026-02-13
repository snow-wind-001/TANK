#!/usr/bin/env python
"""
可视化测试脚本
测试所有MADDPG场景的可视化效果
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

# 添加环境路径
sys.path.append('/home/spikebai/gitcode/maddpg/multiagent-particle-envs')

import torch
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from maddpg.trainer.torch_maddpg import MADDPGAgentTrainer

class ScenarioVisualizer:
    def __init__(self):
        self.scenarios = [
            "simple",
            "simple_spread",
            "simple_tag",
            "simple_adversary",
            "simple_speaker_listener",
            "simple_crypto",
            "simple_push",
            "simple_reference"
        ]

    def create_env(self, scenario_name, num_adversaries=0):
        """创建环境"""
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
        return env, scenario

    def load_trained_agents(self, env, scenario_name, num_adversaries=0):
        """加载训练好的智能体"""
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        # 创建args对象
        class Args:
            lr = 1e-2
            gamma = 0.95
            batch_size = 1024
            num_units = 64
            max_episode_len = 25

        args = Args()

        # 创建训练器
        trainers = []
        for i in range(env.n):
            trainer = MADDPGAgentTrainer(f"agent_{i}", None, obs_shape_n, env.action_space, i, args)
            trainers.append(trainer)

        # 尝试加载预训练模型
        try:
            model_path = f"/tmp/policy/agent_{i}.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                trainer.actor.load_state_dict(checkpoint['actor_state_dict'])
                trainer.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
                trainer.actor.eval()
                print(f"Loaded model for agent {i}")
        except Exception as e:
            print(f"Could not load model for agent {i}: {e}")

        return trainers

    def run_episode_with_render(self, env, trainers, max_steps=100):
        """运行一个episode并记录轨迹"""
        obs = env.reset()
        episode_data = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'positions': [],
            'landmarks': [],
            'agent_colors': []
        }

        for step in range(max_steps):
            # 获取智能体位置信息
            positions = []
            agent_colors = []
            for i, agent in enumerate(env.agents):
                positions.append(agent.state.p_pos.copy())
                agent_colors.append(agent.color.copy())

            # 获取地标位置
            landmarks = []
            for landmark in env.world.landmarks:
                landmarks.append({
                    'pos': landmark.state.p_pos.copy(),
                    'color': landmark.color.copy(),
                    'size': landmark.size
                })

            # 记录状态
            episode_data['obs'].append([o.copy() for o in obs])
            episode_data['positions'].append(positions)
            episode_data['landmarks'].append(landmarks)
            episode_data['agent_colors'].append(agent_colors)

            # 获取动作
            if trainers:
                actions = [agent.action(obs_i) for agent, obs_i in zip(trainers, obs)]
            else:
                # 随机动作
                actions = []
                for i, action_space in enumerate(env.action_space):
                    if hasattr(action_space, 'sample'):
                        actions.append(action_space.sample())
                    else:
                        # 简单的随机动作
                        if hasattr(action_space, 'n'):
                            actions.append(np.random.randint(0, action_space.n))
                        else:
                            actions.append(np.random.randn(*action_space.shape))

            episode_data['actions'].append(actions)

            # 环境步进
            obs, rewards, dones, _ = env.step(actions)
            episode_data['rewards'].append(rewards.copy())

            if all(dones):
                break

        return episode_data

    def create_visualization(self, scenario_name, episode_data):
        """创建可视化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 场景可视化
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_aspect('equal')
        ax1.set_title(f'{scenario_name.replace("_", " ").title()} Scenario Visualization')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.3)

        # 绘制地标
        if episode_data['landmarks']:
            for landmark in episode_data['landmarks'][0]:  # 使用第一个时间步的地标
                circle = plt.Circle(landmark['pos'], landmark['size'],
                                  color=landmark['color'], alpha=0.6)
                ax1.add_patch(circle)

        # 绘制智能体轨迹
        agent_trails = [[] for _ in range(len(episode_data['positions'][0]))]
        for t in range(len(episode_data['positions'])):
            for i, pos in enumerate(episode_data['positions'][t]):
                agent_trails[i].append(pos)

        # 绘制轨迹
        colors = episode_data['agent_colors'][0] if episode_data['agent_colors'] else ['blue', 'red', 'green', 'yellow']
        for i, trail in enumerate(agent_trails):
            if trail:
                trail = np.array(trail)
                ax1.plot(trail[:, 0], trail[:, 1], 'o-', color=colors[i],
                        alpha=0.7, markersize=4, label=f'Agent {i+1}')

                # 标记起点和终点
                ax1.plot(trail[0, 0], trail[0, 1], 's', color=colors[i],
                        markersize=8, label=f'Agent {i+1} Start')
                ax1.plot(trail[-1, 0], trail[-1, 1], '*', color=colors[i],
                        markersize=12, label=f'Agent {i+1} End')

        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))

        # 奖励曲线
        if episode_data['rewards']:
            rewards_array = np.array(episode_data['rewards'])
            if len(rewards_array.shape) > 1:
                # 多智能体情况
                for i in range(rewards_array.shape[1]):
                    ax2.plot(rewards_array[:, i], label=f'Agent {i+1}')
                ax2.plot(rewards_array.sum(axis=1), 'k--', linewidth=2, label='Total Reward')
            else:
                # 单智能体情况
                ax2.plot(rewards_array, 'b-', linewidth=2, label='Reward')

            ax2.set_xlabel('Step')
            ax2.set_ylabel('Reward')
            ax2.set_title('Episode Rewards')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def test_scenario_visualization(self, scenario_name, num_adversaries=0):
        """测试单个场景的可视化"""
        print(f"\n=== Testing {scenario_name} Visualization ===")

        try:
            # 创建环境
            env, scenario = self.create_env(scenario_name, num_adversaries)

            # 加载智能体
            trainers = self.load_trained_agents(env, scenario_name, num_adversaries)

            # 运行一个episode
            print(f"Running episode for {scenario_name}...")
            episode_data = self.run_episode_with_render(env, trainers, max_steps=50)

            # 创建可视化
            print(f"Creating visualization for {scenario_name}...")
            fig = self.create_visualization(scenario_name, episode_data)

            # 保存图像
            save_path = f"./visualization_{scenario_name}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

            # 显示统计信息
            if episode_data['rewards']:
                rewards_array = np.array(episode_data['rewards'])
                if len(rewards_array.shape) > 1:
                    total_rewards = rewards_array.sum(axis=1)
                    print(f"Total Reward: {total_rewards[-1]:.2f}")
                    print(f"Average Step Reward: {np.mean(total_rewards):.2f}")
                    for i in range(rewards_array.shape[1]):
                        print(f"Agent {i+1} Final Reward: {rewards_array[-1, i]:.2f}")
                else:
                    print(f"Final Reward: {rewards_array[-1]:.2f}")
                    print(f"Average Reward: {np.mean(rewards_array):.2f}")

            plt.close(fig)
            return True

        except Exception as e:
            print(f"Error visualizing {scenario_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_all_scenarios(self):
        """测试所有场景"""
        print("Starting comprehensive scenario visualization testing...")

        results = {}

        # 测试每个场景
        for scenario in self.scenarios:
            try:
                if scenario in ["simple_tag", "simple_adversary", "simple_crypto"]:
                    # 对抗性场景需要指定adversaries数量
                    if scenario == "simple_tag":
                        success = self.test_scenario_visualization(scenario, num_adversaries=3)
                    elif scenario == "simple_adversary":
                        success = self.test_scenario_visualization(scenario, num_adversaries=1)
                    else:
                        success = self.test_scenario_visualization(scenario, num_adversaries=1)
                else:
                    success = self.test_scenario_visualization(scenario)

                results[scenario] = success

            except Exception as e:
                print(f"Failed to test {scenario}: {e}")
                results[scenario] = False

        return results

def main():
    visualizer = ScenarioVisualizer()
    results = visualizer.test_all_scenarios()

    print("\n" + "="*60)
    print("VISUALIZATION TEST SUMMARY")
    print("="*60)

    success_count = 0
    for scenario, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{scenario:25} {status}")
        if success:
            success_count += 1

    print("-"*60)
    print(f"Total: {success_count}/{len(results)} scenarios successful")

    if success_count == len(results):
        print("🎉 All scenarios visualization successful!")
    else:
        print(f"⚠️  {len(results) - success_count} scenarios failed")

if __name__ == "__main__":
    main()