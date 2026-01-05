#!/usr/bin/env python3
"""
Analyze distributed training benchmark results
"""

import json
import os
import glob
from pathlib import Path
import statistics


def load_results(results_dir="results/distributed"):
    """Load all distributed benchmark results"""
    results = []

    # Find all result JSON files
    pattern = f"{results_dir}/**/*.json"
    files = glob.glob(pattern, recursive=True)

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                data['filepath'] = filepath
                results.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return results


def group_by_world_size(results):
    """Group results by world size"""
    grouped = {}

    for result in results:
        world_size = result.get('world_size', 1)
        if world_size not in grouped:
            grouped[world_size] = []
        grouped[world_size].append(result)

    return grouped


def calculate_statistics(results):
    """Calculate statistics for a group of results"""
    if not results:
        return {}

    # Extract metrics from rank 0 (master node)
    master_results = [r for r in results if r.get('rank') == 0]
    if not master_results:
        master_results = results  # Fallback to all results

    master = master_results[0]

    # Get samples_per_second from all nodes
    samples_per_sec_list = [r.get('samples_per_second', 0) for r in results]

    stats = {
        'world_size': master.get('world_size', 1),
        'samples_per_second': master.get('samples_per_second', 0),
        'tokens_per_second': master.get('tokens_per_second', 0),
        'train_time_sec': master.get('train_time_sec', 0),
        'train_loss': master.get('train_loss', 0),
        'scaling_efficiency_percent': master.get('scaling_efficiency_percent', 0),
        'cost_per_1k_samples_usd': master.get('cost_per_1k_samples_usd', 0),
        'cost_total_usd': master.get('cost_total_usd', 0),
        'num_nodes': len(results),
        'samples_per_sec_variance': statistics.stdev(samples_per_sec_list) if len(samples_per_sec_list) > 1 else 0
    }

    return stats


def print_comparison_table(grouped_stats):
    """Print comparison table across different world sizes"""
    print("\n" + "=" * 120)
    print("DISTRIBUTED TRAINING BENCHMARK COMPARISON")
    print("=" * 120)
    print()

    # Header
    header = f"{'Nodes':<8} {'Samples/sec':<15} {'Tokens/sec':<15} {'Time (s)':<12} {'Loss':<10} {'Efficiency':<15} {'Cost/1K':<15}"
    print(header)
    print("-" * 120)

    # Sort by world size
    for world_size in sorted(grouped_stats.keys()):
        stats = grouped_stats[world_size]

        efficiency_str = f"{stats['scaling_efficiency_percent']:.1f}%" if stats['scaling_efficiency_percent'] > 0 else "N/A"

        row = (
            f"{stats['world_size']:<8} "
            f"{stats['samples_per_second']:<15.2f} "
            f"{stats['tokens_per_second']:<15.0f} "
            f"{stats['train_time_sec']:<12.1f} "
            f"{stats['train_loss']:<10.4f} "
            f"{efficiency_str:<15} "
            f"${stats['cost_per_1k_samples_usd']:<14.4f}"
        )
        print(row)

    print("=" * 120)
    print()


def print_scaling_analysis(grouped_stats):
    """Print scaling analysis"""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)
    print()

    if 1 not in grouped_stats:
        print("Note: Single-node baseline not found. Using 1.46 samples/sec from previous benchmarks.")
        baseline_throughput = 1.46
    else:
        baseline_throughput = grouped_stats[1]['samples_per_second']

    print(f"Baseline (1 node): {baseline_throughput:.2f} samples/sec")
    print()

    for world_size in sorted(grouped_stats.keys()):
        if world_size == 1:
            continue

        stats = grouped_stats[world_size]
        throughput = stats['samples_per_second']

        # Calculate scaling metrics
        ideal_throughput = baseline_throughput * world_size
        actual_speedup = throughput / baseline_throughput
        ideal_speedup = world_size
        efficiency = (actual_speedup / ideal_speedup) * 100

        print(f"{world_size} nodes:")
        print(f"  Actual throughput:    {throughput:.2f} samples/sec")
        print(f"  Ideal throughput:     {ideal_throughput:.2f} samples/sec (linear scaling)")
        print(f"  Actual speedup:       {actual_speedup:.2f}×")
        print(f"  Ideal speedup:        {ideal_speedup:.0f}×")
        print(f"  Scaling efficiency:   {efficiency:.1f}%")
        print(f"  Communication overhead: {100 - efficiency:.1f}%")
        print()


def print_cost_analysis(grouped_stats):
    """Print cost analysis"""
    print("\n" + "=" * 80)
    print("COST ANALYSIS")
    print("=" * 80)
    print()

    print(f"{'Nodes':<8} {'Time (min)':<12} {'Total Cost':<15} {'Cost/1K samples':<20} {'Savings vs 1 node':<20}")
    print("-" * 80)

    baseline_cost = grouped_stats.get(1, {}).get('cost_per_1k_samples_usd', 0.114)

    for world_size in sorted(grouped_stats.keys()):
        stats = grouped_stats[world_size]

        time_min = stats['train_time_sec'] / 60
        total_cost = stats['cost_total_usd']
        cost_per_1k = stats['cost_per_1k_samples_usd']

        if world_size > 1:
            savings = ((baseline_cost - cost_per_1k) / baseline_cost) * 100
            savings_str = f"{savings:.1f}%"
        else:
            savings_str = "baseline"

        print(f"{world_size:<8} {time_min:<12.1f} ${total_cost:<14.4f} ${cost_per_1k:<19.4f} {savings_str:<20}")

    print("=" * 80)
    print()


def main():
    """Main analysis function"""
    print("Loading distributed benchmark results...")

    results = load_results()

    if not results:
        print("No results found in results/distributed/")
        print("Run benchmarks first:")
        print("  ./run_distributed_benchmark.sh 2  # 2 nodes")
        print("  ./run_distributed_benchmark.sh 4  # 4 nodes")
        return

    print(f"Loaded {len(results)} result files")

    # Group by world size
    grouped = group_by_world_size(results)

    # Calculate statistics for each group
    grouped_stats = {}
    for world_size, group_results in grouped.items():
        grouped_stats[world_size] = calculate_statistics(group_results)

    # Print analysis
    print_comparison_table(grouped_stats)
    print_scaling_analysis(grouped_stats)
    print_cost_analysis(grouped_stats)

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Find best configuration
    best_efficiency = max((s['scaling_efficiency_percent'] for s in grouped_stats.values() if s['scaling_efficiency_percent'] > 0), default=0)
    best_cost = min((s['cost_per_1k_samples_usd'] for s in grouped_stats.values()), default=0)

    print("Best Configurations:")
    print()

    for world_size, stats in grouped_stats.items():
        if stats['scaling_efficiency_percent'] == best_efficiency and best_efficiency > 0:
            print(f"✓ Best Scaling Efficiency: {world_size} nodes ({best_efficiency:.1f}%)")

        if stats['cost_per_1k_samples_usd'] == best_cost:
            print(f"✓ Best Cost: {world_size} nodes (${best_cost:.4f} per 1K samples)")

    print()

    # General recommendations
    if 4 in grouped_stats:
        stats_4 = grouped_stats[4]
        if stats_4['scaling_efficiency_percent'] > 90:
            print("✓ Excellent scaling to 4 nodes - recommended for production")
        elif stats_4['scaling_efficiency_percent'] > 80:
            print("✓ Good scaling to 4 nodes - acceptable for production")
        else:
            print("⚠ Scaling efficiency <80% - check network latency and batch size")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
