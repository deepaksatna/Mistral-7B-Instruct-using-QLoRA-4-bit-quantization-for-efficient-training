#!/usr/bin/env python3
"""
Generate benchmark visualization graphs from distributed training results
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


class DistributedBenchmarkVisualizer:
    """Generate visualizations for distributed training benchmarks"""

    def __init__(self, results_dir="results/distributed"):
        self.results_dir = results_dir
        self.baseline_samples_per_sec = 1.46  # From single-node benchmarks
        self.baseline_cost_per_1k = 0.114  # From single-node benchmarks

    def load_results(self):
        """Load all distributed benchmark results"""
        results = []
        pattern = f"{self.results_dir}/**/*.json"
        files = glob.glob(pattern, recursive=True)

        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        return results

    def group_by_world_size(self, results):
        """Group results by world size and get master node data"""
        grouped = {}

        for result in results:
            world_size = result.get('world_size', 1)
            rank = result.get('rank', 0)

            # Only use master node (rank 0) data for summary
            if rank == 0:
                grouped[world_size] = result

        return grouped

    def generate_all_graphs(self, output_dir="../../images"):
        """Generate all benchmark graphs"""
        os.makedirs(output_dir, exist_ok=True)

        results = self.load_results()
        if not results:
            print("No results found!")
            return

        grouped = self.group_by_world_size(results)

        # Add baseline (1 node)
        grouped[1] = {
            'world_size': 1,
            'samples_per_second': self.baseline_samples_per_sec,
            'tokens_per_second': self.baseline_samples_per_sec * 512,
            'cost_per_1k_samples_usd': self.baseline_cost_per_1k,
            'train_time_sec': 686,  # For 100 steps
            'scaling_efficiency_percent': 100.0
        }

        print(f"Generating graphs for {len(grouped)} configurations...")

        # Generate each graph
        self.plot_scaling_performance(grouped, output_dir)
        self.plot_scaling_efficiency(grouped, output_dir)
        self.plot_cost_analysis(grouped, output_dir)
        self.plot_comprehensive_comparison(grouped, output_dir)

        print(f"\n✓ All graphs saved to: {output_dir}/")

    def plot_scaling_performance(self, grouped, output_dir):
        """Plot scaling performance: throughput vs nodes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        nodes = sorted(grouped.keys())
        samples_per_sec = [grouped[n]['samples_per_second'] for n in nodes]
        tokens_per_sec = [grouped[n]['tokens_per_second'] for n in nodes]

        # Ideal linear scaling
        ideal_samples = [self.baseline_samples_per_sec * n for n in nodes]
        ideal_tokens = [self.baseline_samples_per_sec * 512 * n for n in nodes]

        # Plot 1: Samples per second
        ax1.plot(nodes, samples_per_sec, 'o-', linewidth=3, markersize=12,
                color='#2E86AB', label='Actual Performance')
        ax1.plot(nodes, ideal_samples, '--', linewidth=2, color='#A23B72',
                label='Ideal Linear Scaling', alpha=0.7)

        ax1.set_xlabel('Number of GPU Nodes', fontweight='bold')
        ax1.set_ylabel('Samples per Second', fontweight='bold')
        ax1.set_title('Distributed Training Throughput\n(Samples per Second)',
                     fontweight='bold', pad=20)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(nodes)

        # Add value labels
        for i, (n, s) in enumerate(zip(nodes, samples_per_sec)):
            ax1.annotate(f'{s:.2f}', (n, s), textcoords="offset points",
                        xytext=(0,10), ha='center', fontweight='bold')

        # Plot 2: Tokens per second
        ax2.plot(nodes, tokens_per_sec, 'o-', linewidth=3, markersize=12,
                color='#F18F01', label='Actual Performance')
        ax2.plot(nodes, ideal_tokens, '--', linewidth=2, color='#A23B72',
                label='Ideal Linear Scaling', alpha=0.7)

        ax2.set_xlabel('Number of GPU Nodes', fontweight='bold')
        ax2.set_ylabel('Tokens per Second', fontweight='bold')
        ax2.set_title('Distributed Training Throughput\n(Tokens per Second)',
                     fontweight='bold', pad=20)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(nodes)

        # Add value labels
        for i, (n, t) in enumerate(zip(nodes, tokens_per_sec)):
            ax2.annotate(f'{t:.0f}', (n, t), textcoords="offset points",
                        xytext=(0,10), ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/08_Distributed_Throughput.png', dpi=150, bbox_inches='tight')
        print("✓ Generated: 08_Distributed_Throughput.png")
        plt.close()

    def plot_scaling_efficiency(self, grouped, output_dir):
        """Plot scaling efficiency"""
        fig, ax = plt.subplots(figsize=(12, 8))

        nodes = sorted(grouped.keys())
        efficiency = [grouped[n]['scaling_efficiency_percent'] for n in nodes]

        # Bar chart
        bars = ax.bar(nodes, efficiency, color=['#2E86AB', '#F18F01', '#C73E1D', '#6A994E'],
                     alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add 100% reference line
        ax.axhline(y=100, color='#A23B72', linestyle='--', linewidth=2,
                  label='Perfect Linear Scaling (100%)', alpha=0.7)

        # Add 90% and 95% reference lines
        ax.axhline(y=95, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=90, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_xlabel('Number of GPU Nodes', fontweight='bold')
        ax.set_ylabel('Scaling Efficiency (%)', fontweight='bold')
        ax.set_title('Distributed Training Scaling Efficiency\n(Actual vs Ideal Linear Scaling)',
                    fontweight='bold', pad=20, fontsize=16)
        ax.set_ylim(0, 110)
        ax.set_xticks(nodes)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (n, e) in enumerate(zip(nodes, efficiency)):
            ax.text(n, e + 2, f'{e:.1f}%', ha='center', va='bottom',
                   fontweight='bold', fontsize=14)

        # Add speedup labels
        for i, n in enumerate(nodes):
            speedup = grouped[n]['samples_per_second'] / self.baseline_samples_per_sec
            ax.text(n, 5, f'{speedup:.2f}× speedup', ha='center', va='bottom',
                   fontsize=11, style='italic', color='darkblue')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/09_Scaling_Efficiency.png', dpi=150, bbox_inches='tight')
        print("✓ Generated: 09_Scaling_Efficiency.png")
        plt.close()

    def plot_cost_analysis(self, grouped, output_dir):
        """Plot cost analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        nodes = sorted(grouped.keys())
        cost_per_1k = [grouped[n]['cost_per_1k_samples_usd'] for n in nodes]

        # Calculate total time for 10K samples
        time_for_10k = []
        total_cost_10k = []
        for n in nodes:
            samples_per_sec = grouped[n]['samples_per_second']
            time_sec = 10000 / samples_per_sec
            time_min = time_sec / 60
            cost = (time_min / 60) * 0.60 * n  # $0.60/hour per GPU
            time_for_10k.append(time_min)
            total_cost_10k.append(cost)

        # Plot 1: Cost per 1K samples
        bars1 = ax1.bar(nodes, cost_per_1k, color='#F18F01', alpha=0.8,
                       edgecolor='black', linewidth=1.5)

        ax1.set_xlabel('Number of GPU Nodes', fontweight='bold')
        ax1.set_ylabel('Cost per 1K Samples (USD)', fontweight='bold')
        ax1.set_title('Training Cost per 1,000 Samples\n(OCI A10 @ $0.60/hour)',
                     fontweight='bold', pad=20)
        ax1.set_xticks(nodes)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (n, c) in enumerate(zip(nodes, cost_per_1k)):
            ax1.text(n, c, f'${c:.4f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=12)

        # Plot 2: Time and total cost for 10K samples
        ax2_twin = ax2.twinx()

        bars2 = ax2.bar([n - 0.2 for n in nodes], time_for_10k, width=0.4,
                       color='#2E86AB', alpha=0.8, label='Time (minutes)',
                       edgecolor='black', linewidth=1.5)
        bars3 = ax2_twin.bar([n + 0.2 for n in nodes], total_cost_10k, width=0.4,
                            color='#C73E1D', alpha=0.8, label='Total Cost (USD)',
                            edgecolor='black', linewidth=1.5)

        ax2.set_xlabel('Number of GPU Nodes', fontweight='bold')
        ax2.set_ylabel('Training Time (minutes)', fontweight='bold', color='#2E86AB')
        ax2_twin.set_ylabel('Total Cost (USD)', fontweight='bold', color='#C73E1D')
        ax2.set_title('Time and Cost for 10,000 Samples', fontweight='bold', pad=20)
        ax2.set_xticks(nodes)
        ax2.tick_params(axis='y', labelcolor='#2E86AB')
        ax2_twin.tick_params(axis='y', labelcolor='#C73E1D')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (n, t) in enumerate(zip(nodes, time_for_10k)):
            ax2.text(n - 0.2, t, f'{t:.1f}m', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        for i, (n, c) in enumerate(zip(nodes, total_cost_10k)):
            ax2_twin.text(n + 0.2, c, f'${c:.2f}', ha='center', va='bottom',
                         fontweight='bold', fontsize=10)

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/10_Cost_Analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Generated: 10_Cost_Analysis.png")
        plt.close()

    def plot_comprehensive_comparison(self, grouped, output_dir):
        """Comprehensive comparison dashboard"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        nodes = sorted(grouped.keys())

        # 1. Throughput comparison
        ax1 = fig.add_subplot(gs[0, 0])
        samples_per_sec = [grouped[n]['samples_per_second'] for n in nodes]
        ax1.bar(nodes, samples_per_sec, color='#2E86AB', alpha=0.8, edgecolor='black')
        ax1.set_title('Throughput (Samples/sec)', fontweight='bold')
        ax1.set_ylabel('Samples per Second')
        ax1.grid(True, alpha=0.3, axis='y')
        for n, s in zip(nodes, samples_per_sec):
            ax1.text(n, s, f'{s:.2f}', ha='center', va='bottom', fontweight='bold')

        # 2. Scaling efficiency
        ax2 = fig.add_subplot(gs[0, 1])
        efficiency = [grouped[n]['scaling_efficiency_percent'] for n in nodes]
        ax2.bar(nodes, efficiency, color='#F18F01', alpha=0.8, edgecolor='black')
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Scaling Efficiency (%)', fontweight='bold')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_ylim(0, 110)
        ax2.grid(True, alpha=0.3, axis='y')
        for n, e in zip(nodes, efficiency):
            ax2.text(n, e, f'{e:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Speedup comparison
        ax3 = fig.add_subplot(gs[1, 0])
        speedup_actual = [grouped[n]['samples_per_second'] / self.baseline_samples_per_sec
                         for n in nodes]
        speedup_ideal = nodes
        x = np.arange(len(nodes))
        width = 0.35
        ax3.bar(x - width/2, speedup_actual, width, label='Actual',
               color='#6A994E', alpha=0.8, edgecolor='black')
        ax3.bar(x + width/2, speedup_ideal, width, label='Ideal',
               color='#A23B72', alpha=0.8, edgecolor='black')
        ax3.set_title('Speedup vs Single Node', fontweight='bold')
        ax3.set_ylabel('Speedup (×)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(nodes)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        for i, (a, ideal) in enumerate(zip(speedup_actual, speedup_ideal)):
            ax3.text(i - width/2, a, f'{a:.2f}×', ha='center', va='bottom', fontsize=9)
            ax3.text(i + width/2, ideal, f'{ideal}×', ha='center', va='bottom', fontsize=9)

        # 4. Cost per 1K samples
        ax4 = fig.add_subplot(gs[1, 1])
        cost = [grouped[n]['cost_per_1k_samples_usd'] for n in nodes]
        ax4.bar(nodes, cost, color='#C73E1D', alpha=0.8, edgecolor='black')
        ax4.set_title('Cost per 1K Samples (USD)', fontweight='bold')
        ax4.set_ylabel('Cost (USD)')
        ax4.grid(True, alpha=0.3, axis='y')
        for n, c in zip(nodes, cost):
            ax4.text(n, c, f'${c:.4f}', ha='center', va='bottom', fontweight='bold')

        # 5. Summary table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')

        table_data = []
        table_data.append(['Nodes', 'Samples/sec', 'Tokens/sec', 'Efficiency',
                          'Speedup', 'Cost/1K', 'Time 10K'])

        for n in nodes:
            data = grouped[n]
            speedup = data['samples_per_second'] / self.baseline_samples_per_sec
            time_10k = (10000 / data['samples_per_second']) / 60  # minutes

            table_data.append([
                f"{n}",
                f"{data['samples_per_second']:.2f}",
                f"{data['tokens_per_second']:.0f}",
                f"{data['scaling_efficiency_percent']:.1f}%",
                f"{speedup:.2f}×",
                f"${data['cost_per_1k_samples_usd']:.4f}",
                f"{time_10k:.1f}m"
            ])

        table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.1, 0.15, 0.15, 0.15, 0.12, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # Style header row
        for i in range(7):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(7):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8F4F8')

        # Main title
        fig.suptitle('Distributed Training Benchmark - Comprehensive Results\nMistral-7B QLoRA on 4× NVIDIA A10 GPUs',
                    fontsize=18, fontweight='bold', y=0.98)

        plt.savefig(f'{output_dir}/11_Comprehensive_Dashboard.png', dpi=150, bbox_inches='tight')
        print("✓ Generated: 11_Comprehensive_Dashboard.png")
        plt.close()


def main():
    """Generate all benchmark graphs"""
    import sys

    # Allow specifying custom results directory
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results/distributed"

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "../../images"

    print("=" * 80)
    print("DISTRIBUTED TRAINING BENCHMARK - GRAPH GENERATION")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()

    visualizer = DistributedBenchmarkVisualizer(results_dir)
    visualizer.generate_all_graphs(output_dir)

    print()
    print("=" * 80)
    print("✓ All graphs generated successfully!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - 08_Distributed_Throughput.png")
    print("  - 09_Scaling_Efficiency.png")
    print("  - 10_Cost_Analysis.png")
    print("  - 11_Comprehensive_Dashboard.png")
    print()


if __name__ == "__main__":
    main()
