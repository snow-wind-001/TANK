#!/usr/bin/env python
"""
简化可视化脚本 - 基于训练数据创建可视化
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_training_data(exp_name):
    """加载训练数据"""
    try:
        rewards = pickle.load(open(f'./learning_curves/{exp_name}_rewards.pkl', 'rb'))
        agent_rewards = pickle.load(open(f'./learning_curves/{exp_name}_agrewards.pkl', 'rb'))
        return rewards, agent_rewards
    except FileNotFoundError:
        return None, None

def create_training_curve_plot(scenario_name, exp_name):
    """创建学习曲线图"""
    rewards, agent_rewards = load_training_data(exp_name)

    if rewards is None:
        print(f"No training data found for {exp_name}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 总奖励曲线
    ax1.plot(rewards, 'b-', linewidth=2, label='Total Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title(f'{scenario_name.replace("_", " ").title()} - Learning Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 个体智能体奖励曲线
    if agent_rewards and len(agent_rewards) > 0:
        agent_rewards_array = np.array(agent_rewards)

        if len(agent_rewards_array.shape) == 2:  # 多智能体情况
            num_agents = agent_rewards_array.shape[1]
            for i in range(num_agents):
                ax2.plot(agent_rewards_array[:, i], label=f'Agent {i+1}', alpha=0.7)
        elif len(agent_rewards_array.shape) == 1:  # 单智能体情况
            ax2.plot(agent_rewards_array, label='Agent 1', alpha=0.7)

        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Agent Reward')
        ax2.set_title('Individual Agent Rewards')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    save_path = f"./{scenario_name}_learning_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Learning curve saved to {save_path}")

    plt.close(fig)

def create_scenario_summary():
    """创建所有场景的总结"""
    scenarios = [
        ("simple", "simple_test"),
        ("simple_spread", "simple_spread_test"),
        ("simple_tag", "simple_tag_test"),
        ("simple_adversary", "simple_adversary_test"),
        ("simple_speaker_listener", "simple_speaker_listener_test"),
        ("simple_crypto", "simple_crypto_test"),
        ("simple_push", "simple_push_test"),
        ("simple_reference", "simple_ref_test")
    ]

    results = []

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (scenario, exp_name) in enumerate(scenarios):
        rewards, agent_rewards = load_training_data(exp_name)

        ax = axes[idx]

        if rewards is not None:
            # 计算移动平均
            window = min(5, len(rewards))
            if window > 0:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
                ax.plot(range(window-1, len(rewards)), moving_avg, 'b-', linewidth=2, label=f'MA{window}')

            ax.set_title(f'{scenario.replace("_", " ").title()}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
            ax.legend()

            final_reward = rewards[-1] if len(rewards) > 0 else 0
            avg_reward = np.mean(rewards) if len(rewards) > 0 else 0
            results.append({
                'scenario': scenario,
                'final_reward': final_reward,
                'avg_reward': avg_reward,
                'episodes': len(rewards),
                'status': 'SUCCESS'
            })
        else:
            ax.text(0.5, 0.5, f'{scenario}\nNo Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{scenario.replace("_", " ").title()} (No Data)')
            results.append({
                'scenario': scenario,
                'status': 'NO_DATA'
            })

    plt.tight_layout()
    plt.savefig('./all_scenarios_comparison.png', dpi=150, bbox_inches='tight')
    print("All scenarios comparison saved to all_scenarios_comparison.png")
    plt.close(fig)

    return results

def create_performance_analysis():
    """创建性能分析报告"""
    scenarios = [
        ("Simple", "simple_test"),
        ("Simple Spread", "simple_spread_test"),
        ("Simple Tag", "simple_tag_test"),
        ("Simple Adversary", "simple_adversary_test"),
        ("Simple Speaker-Listener", "simple_speaker_listener_test"),
        ("Simple Crypto", "simple_crypto_test"),
        ("Simple Push", "simple_push_test"),
        ("Simple Reference", "simple_ref_test")
    ]

    data = []

    for scenario_name, exp_name in scenarios:
        rewards, agent_rewards = load_training_data(exp_name)

        if rewards is not None:
            # 计算统计信息
            final_reward = rewards[-1] if len(rewards) > 0 else 0
            avg_reward = np.mean(rewards) if len(rewards) > 0 else 0
            best_reward = np.max(rewards) if len(rewards) > 0 else 0
            improvement = final_reward - rewards[0] if len(rewards) > 1 else 0

            # 分析智能体数量和类型
            if "tag" in exp_name:
                agent_type = "Adversarial (1 good + 3 adversaries)"
                num_agents = 4
            elif "adversary" in exp_name:
                agent_type = "Mixed (1 adversary + 2 good)"
                num_agents = 3
            elif "crypto" in exp_name:
                agent_type = "Crypto (Alice + Bob + Eve)"
                num_agents = 3
            elif "speaker_listener" in exp_name:
                agent_type = "Communication (speaker + listener)"
                num_agents = 2
            elif "reference" in exp_name:
                agent_type = "Reference-based communication"
                num_agents = 2
            elif "spread" in exp_name:
                agent_type = "Cooperative"
                num_agents = 3
            else:
                agent_type = "Single agent"
                num_agents = 1

            data.append({
                'Scenario': scenario_name,
                'Type': agent_type,
                'Agents': num_agents,
                'Episodes': len(rewards),
                'Final Reward': f"{final_reward:.2f}",
                'Average Reward': f"{avg_reward:.2f}",
                'Best Reward': f"{best_reward:.2f}",
                'Improvement': f"{improvement:.2f}",
                'Status': '✅ Success'
            })
        else:
            data.append({
                'Scenario': scenario_name,
                'Type': 'Unknown',
                'Agents': '-',
                'Episodes': 0,
                'Final Reward': '-',
                'Average Reward': '-',
                'Best Reward': '-',
                'Improvement': '-',
                'Status': '❌ No Data'
            })

    return data

def main():
    """主函数"""
    print("MADDPG Scenario Visualization Analysis")
    print("=" * 50)

    # 创建各个场景的学习曲线
    scenarios = [
        ("simple", "simple_test"),
        ("simple_spread", "simple_spread_test"),
        ("simple_tag", "simple_tag_test"),
        ("simple_adversary", "simple_adversary_test"),
        ("simple_speaker_listener", "simple_speaker_listener_test"),
        ("simple_crypto", "simple_crypto_test"),
        ("simple_push", "simple_push_test"),
        ("simple_reference", "simple_ref_test")
    ]

    print("Creating individual learning curves...")
    for scenario, exp_name in scenarios:
        create_training_curve_plot(scenario, exp_name)

    print("Creating combined comparison plot...")
    results = create_scenario_summary()

    print("Generating performance analysis...")
    performance_data = create_performance_analysis()

    # 打印性能分析报告
    print("\n" + "=" * 100)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 100)

    # 表格标题
    header = f"{'Scenario':<20} {'Type':<25} {'Agents':<8} {'Episodes':<10} {'Final Reward':<15} {'Improvement':<12} {'Status':<10}"
    print(header)
    print("-" * len(header))

    # 表格内容
    for row in performance_data:
        line = f"{row['Scenario']:<20} {row['Type']:<25} {row['Agents']:<8} {row['Episodes']:<10} {row['Final Reward']:<15} {row['Improvement']:<12} {row['Status']:<10}"
        print(line)

    print("-" * len(header))

    # 统计
    successful = sum(1 for row in performance_data if row['Status'] == '✅ Success')
    print(f"\nSummary: {successful}/{len(performance_data)} scenarios successfully trained and visualized")

    print("\nVisualization files created:")
    print("- Individual learning curves: *_learning_curve.png")
    print("- Combined comparison: all_scenarios_comparison.png")

if __name__ == "__main__":
    main()