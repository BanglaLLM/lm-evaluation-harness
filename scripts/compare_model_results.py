import matplotlib.pyplot as plt
import json
import numpy as np
import os
from pathlib import Path
import argparse

def load_results(filepath):
    """Load results from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(results, metric_type='acc_norm,none'):
    """Extract metrics from results for all datasets."""
    metrics = {}
    for dataset, values in results['results'].items():
        if metric_type in values:
            metrics[dataset] = values[metric_type]
    return metrics

def create_comparison_plot(model1_metrics, model2_metrics, model1_name, model2_name, shot_count, metric_name, output_dir):
    """Create a horizontal bar chart comparing metrics between two models."""
    # Get all datasets
    all_datasets = sorted(set(list(model1_metrics.keys()) + list(model2_metrics.keys())))
    
    # Prepare data for plotting
    model1_values = [model1_metrics.get(dataset, 0) for dataset in all_datasets]
    model2_values = [model2_metrics.get(dataset, 0) for dataset in all_datasets]
    
    # Create a figure
    plt.figure(figsize=(10, max(6, len(all_datasets) * 0.6)))
    
    # Set up positions for bars
    y_pos = np.arange(len(all_datasets))
    bar_width = 0.35
    
    # Create horizontal bars
    plt.barh(y_pos - bar_width/2, model1_values, bar_width, label=model1_name, alpha=0.7)
    plt.barh(y_pos + bar_width/2, model2_values, bar_width, label=model2_name, alpha=0.7)
    
    # Customize plot
    plt.yticks(y_pos, all_datasets)
    plt.xlabel(f'{metric_name} Score')
    plt.title(f'Comparison of {metric_name} ({shot_count}-shot) between Models')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Add values on bars
    for i, v in enumerate(model1_values):
        plt.text(v + 0.01, i - bar_width/2, f'{v:.3f}', va='center')
    
    for i, v in enumerate(model2_values):
        plt.text(v + 0.01, i + bar_width/2, f'{v:.3f}', va='center')
    
    # Save the figure
    output_path = os.path.join(output_dir, f'model_comparison_{shot_count}shot_{metric_name.replace(",", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

def compare_models(model1_0shot, model1_5shot, model2_0shot, model2_5shot, model1_name, model2_name, output_dir="outputs/comparisons"):
    """Compare models using both 0-shot and 5-shot results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    model1_0shot_results = load_results(model1_0shot)
    model1_5shot_results = load_results(model1_5shot)
    model2_0shot_results = load_results(model2_0shot)
    model2_5shot_results = load_results(model2_5shot)

    # Extract metrics
    metrics_types = ['acc,none', 'acc_norm,none']
    
    for metric in metrics_types:
        # 0-shot comparison
        model1_0shot_metrics = extract_metrics(model1_0shot_results, metric)
        model2_0shot_metrics = extract_metrics(model2_0shot_results, metric)
        create_comparison_plot(
            model1_0shot_metrics, model2_0shot_metrics, 
            model1_name, model2_name, 
            '0', metric, output_dir
        )
        
        # 5-shot comparison
        model1_5shot_metrics = extract_metrics(model1_5shot_results, metric)
        model2_5shot_metrics = extract_metrics(model2_5shot_results, metric)
        create_comparison_plot(
            model1_5shot_metrics, model2_5shot_metrics, 
            model1_name, model2_name, 
            '5', metric, output_dir
        )
        
        # Additionally, create progress plots showing 0-shot vs 5-shot for each model
        for model_name, shot0_metrics, shot5_metrics in [
            (model1_name, model1_0shot_metrics, model1_5shot_metrics), 
            (model2_name, model2_0shot_metrics, model2_5shot_metrics)
        ]:
            create_comparison_plot(
                shot0_metrics, shot5_metrics,
                f"{model_name} (0-shot)", f"{model_name} (5-shot)",
                'progress', metric, output_dir
            )

def main():
    parser = argparse.ArgumentParser(description='Compare model evaluation results')
    parser.add_argument('--model1_0shot', type=str, required=True, help='Path to model1 0-shot results')
    parser.add_argument('--model1_5shot', type=str, required=True, help='Path to model1 5-shot results')
    parser.add_argument('--model2_0shot', type=str, required=True, help='Path to model2 0-shot results')
    parser.add_argument('--model2_5shot', type=str, required=True, help='Path to model2 5-shot results')
    parser.add_argument('--model1_name', type=str, default='Model1', help='Name for model1')
    parser.add_argument('--model2_name', type=str, default='Model2', help='Name for model2')
    parser.add_argument('--output_dir', type=str, default='outputs/comparisons', help='Output directory for plots')
    
    args = parser.parse_args()
    
    compare_models(
        args.model1_0shot,
        args.model1_5shot,
        args.model2_0shot,
        args.model2_5shot,
        args.model1_name,
        args.model2_name,
        args.output_dir
    )

if __name__ == "__main__":
    main()
