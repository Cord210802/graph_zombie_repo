# Global configuration for visualization options
print("Running bulk simulations...")
SKIP_CITY_ANALYSIS = True
POLICY_NAME =  "EvacuationPolicy"
CONFIG = {
        'node_range': {
            'min': 20,
            'max': 70
        },
        'n_runs': 500,  # Total number of cities to simulate
        'base_seed': 7354681  # For reproducibility
    }

from public.tools.run_bulk import BulkRunner
from public.student_code.solution import EvacuationPolicy
from public.visualization.bulk_analysis import generate_all_visualizations
from public.visualization.city_analysis import analyze_city_scenario
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run bulk simulations')
    parser.add_argument('--skip-city-analysis', action='store_true',
                        help='Skip individual city analysis to save time')
    args = parser.parse_args()
    
    # Determine whether to skip city analysis:
    # It will be skipped if either the command-line flag is provided or the global variable is True.
    skip_city_analysis = args.skip_city_analysis or SKIP_CITY_ANALYSIS
    
    # Configuration for bulk runs
    config = CONFIG
    policy_name = POLICY_NAME
    
    # Create bulk runner
    runner = BulkRunner(
        policy_name=policy_name,
        base_seed=config['base_seed']
    )
    
    # Create policy
    policy = EvacuationPolicy()
    
    # Run batch of simulations
    results, experiment_id = runner.run_batch(policy, config)
    
    # Print summary of results
    core_metrics = results['core_metrics']
    resource_metrics = results['resource_metrics']
    
    print("\nEvacuation Mission Results:")
    print(f"Total Missions: {core_metrics['metadata']['total_runs']}")
    print(f"Mission Success Rate: {core_metrics['overall_performance']['success_rate']*100:.1f}%")
    print(f"Average Mission Time: {core_metrics['overall_performance']['avg_time']:.2f} seconds")
    print(f"Average Path Distance: {core_metrics['overall_performance']['avg_path_length']:.2f}")
    print(f"Average Resources Allocated: {core_metrics['overall_performance']['resources_allocated']:.1f}")
    print(f"Average Resources Used: {core_metrics['overall_performance']['resources_used']:.1f}")
    print(f"Overall Resource Efficiency: {core_metrics['overall_performance']['resource_efficiency']*100:.1f}%")
    
    print("\nResource Efficiency:")
    for rt, metrics in resource_metrics['overall'].items():
        print(f"{rt.replace('_', ' ').title()}:")
        print(f"  Average Allocated: {metrics['avg_allocated']:.1f}")
        print(f"  Average Used: {metrics['avg_used']:.1f}")
        print(f"  Efficiency: {metrics['efficiency']*100:.1f}%")
    
    print("\nResults by City Size:")
    for size, metrics in core_metrics['by_city_size'].items():
        print(f"\nCity Size: {size} nodes")
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Average Time: {metrics['avg_time']:.2f} seconds")
        print(f"  Average Path Length: {metrics['avg_path_length']:.2f}")
        print(f"  Resources Allocated: {metrics['resources_allocated']:.1f}")
        print(f"  Resources Used: {metrics['resources_used']:.1f}")
        print(f"  Resource Efficiency: {metrics['resource_efficiency']*100:.1f}%")
    
    print("\nGenerating experiment-level visualizations...")
    generate_all_visualizations(results, policy_name, experiment_id)
    
    if not skip_city_analysis:
        print("\nAnalyzing individual city scenarios...")
        exp_dir = os.path.join('data', 'policies', policy_name, 'experiments', experiment_id, 'cities')
        for city_dir in os.listdir(exp_dir):
            if city_dir.startswith('city_'):
                city_id = city_dir.replace('city_', '')
                print(f"  Analyzing {city_id}...")
                analyze_city_scenario(city_id, policy_name, experiment_id)
    
    print("\nAnalysis complete. Results saved in:")
    print(f"data/policies/{policy_name}/experiments/{experiment_id}/")

if __name__ == "__main__":
    main()
