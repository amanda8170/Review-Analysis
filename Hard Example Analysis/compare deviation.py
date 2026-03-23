import json
import numpy as np
from scipy import stats
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_rating_metric(file_path, metric):
    """
    Analyzes a single jsonl file to calculate a specific metric (std, kurtosis, or skewness)
    for the review ratings of each paper.
    """
    per_paper_metrics = []

    # Define conditions for each metric
    conditions = {
        'std': {'min_len': 2, 'func': lambda r: np.std(r, ddof=1)},
        'kurtosis': {'min_len': 4, 'func': lambda r: stats.kurtosis(r, bias=False)},
        'skewness': {'min_len': 3, 'func': lambda r: stats.skew(r, bias=False)}
    }

    if metric not in conditions:
        raise ValueError("Metric must be 'std', 'kurtosis', or 'skewness'")

    min_len = conditions[metric]['min_len']
    metric_func = conditions[metric]['func']

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            reviews = data.get('reviews', [])
            ratings = [float(r.get('rating', 0)) for r in reviews if 'rating' in r]

            # Calculate metric if there are enough ratings and they are not all the same
            if len(ratings) >= min_len and np.std(ratings) > 0:
                per_paper_metrics.append(metric_func(ratings))

    if not per_paper_metrics:
        return None

    metric_array = np.array(per_paper_metrics)

    return {
        "file_name": os.path.basename(file_path),
        "total_papers_analyzed": len(per_paper_metrics),
        f"rating_{metric}_stats": {
            "mean": np.mean(metric_array),
            "variance": np.var(metric_array),
            "std_dev": np.std(metric_array),
            "skewness": stats.skew(metric_array),
            "kurtosis": stats.kurtosis(metric_array)
        },
        f"raw_{metric}s": per_paper_metrics
    }


def print_metric_comparison(all_stats, metric, title_prefixes):
    """
    Prints a comparison table for a single metric (std, kurtosis, skewness).
    """
    valid_stats = [s for s in all_stats if s]
    if not valid_stats:
        print(f"No statistics available for {metric} to compare.")
        return

    metric_name_capitalized = metric.capitalize()
    stats_key = f"rating_{metric}_stats"

    data = {
        'Metric': [f'--- Review Rating ({metric_name_capitalized}) ---', f'Mean of {metric_name_capitalized}s',
                   'Variance', 'Std Dev', 'Skewness', 'Kurtosis', '', 'Total Papers Analyzed']
    }

    for i, stats_item in enumerate(valid_stats):
        column_name = title_prefixes[i] if i < len(title_prefixes) else stats_item['file_name']
        stats_dict = stats_item[stats_key]
        data[column_name] = [
            '',
            f"{stats_dict['mean']:.4f}",
            f"{stats_dict['variance']:.4f}",
            f"{stats_dict['std_dev']:.4f}",
            f"{stats_dict['skewness']:.4f}",
            f"{stats_dict['kurtosis']:.4f}",
            '',
            stats_item['total_papers_analyzed']
        ]

    df = pd.DataFrame(data)
    print("\n" + "=" * 80)
    print(f" " * 25 + f"{metric_name_capitalized} Statistics Comparison")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


def plot_combined_distributions(all_stats_map, title_prefixes):
    """
    Plots the distributions for std dev, kurtosis, and skewness in a single 3x3 figure.
    """
    plt.rcParams.update({'font.size': 14})

    metrics = ['std', 'kurtosis', 'skewness']
    num_files = len(title_prefixes)

    fig, axes = plt.subplots(nrows=3, ncols=num_files, figsize=(6 * num_files, 15), squeeze=False)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    plot_configs = {
        'std': {'name': 'Std Dev', 'xlim': (0, 4.5), 'xticks': np.arange(0, 5, 1)},
        'kurtosis': {'name': 'Kurtosis', 'xlim': (-3, 6), 'xticks': np.arange(-3, 7, 1)},
        'skewness': {'name': 'Skewness', 'xlim': (-3, 3), 'xticks': np.arange(-3, 4, 1)}
    }

    for row, metric in enumerate(metrics):
        stats_list = all_stats_map.get(metric, [])
        valid_stats = [s for s in stats_list if s]
        if not valid_stats:
            continue

        # Unify Y-axis for the current row
        max_y_percent = 0
        for i, stats_item in enumerate(valid_stats):
            temp_ax = fig.add_subplot(1, 1, 1)  # Create a temporary axis to calculate height
            sns.histplot(stats_item[f'raw_{metric}s'], ax=temp_ax, stat='percent', bins=30, kde=True)
            max_y_percent = max(max_y_percent, temp_ax.get_ylim()[1])
            temp_ax.remove()

        # Draw the plots for the current row
        for col, stats_item in enumerate(valid_stats):
            ax = axes[row, col]
            config = plot_configs[metric]

            total_papers = stats_item['total_papers_analyzed']
            mean_of_metric = stats_item[f'rating_{metric}_stats']['mean']
            raw_data = stats_item[f'raw_{metric}s']

            sns.histplot(raw_data, ax=ax, stat='percent', bins=30, color=colors[col % len(colors)], kde=True)

            ax.set_ylim(0, max_y_percent * 1.05)
            ax.set_ylabel("Paper Percentage (%)")

            ax_count = ax.twinx()
            ax_count.set_ylabel("Paper Count")
            y_limit_count = (ax.get_ylim()[1] / 100) * total_papers
            ax_count.set_ylim(0, y_limit_count)

            ax.axvline(x=mean_of_metric, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_of_metric:.3f}')
            ax.legend()

            title = f"{title_prefixes[col]}\nReview Rating {config['name']} Distribution"
            ax.set_title(title)

            ax.set_xlabel(f"{config['name']} of Review Ratings")
            ax.set_xlim(config['xlim'])
            ax.set_xticks(config['xticks'])

    plt.tight_layout(pad=3.0)
    output_filename = 'combined_rating_distributions.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Combined distribution plot saved as: '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    file1_path = '../../../Datasets/hard example/final_difficult_sample_pool.jsonl'
    file2_path = '../../../Datasets/hard example/final_simple_accept_sample_pool.jsonl'
    file3_path = '../../../Datasets/hard example/final_simple_reject_sample_pool.jsonl'

    files = [file1_path, file2_path, file3_path]
    title_prefixes = ["Hard Sample", "Simple Accept Sample", "Simple Reject Sample"]

    all_stats_map = {
        'std': [analyze_rating_metric(f, 'std') for f in files],
        'kurtosis': [analyze_rating_metric(f, 'kurtosis') for f in files],
        'skewness': [analyze_rating_metric(f, 'skewness') for f in files]
    }

    # Print comparison tables for each metric
    print_metric_comparison(all_stats_map['std'], 'std', title_prefixes)
    print_metric_comparison(all_stats_map['kurtosis'], 'kurtosis', title_prefixes)
    print_metric_comparison(all_stats_map['skewness'], 'skewness', title_prefixes)

    # Plot the combined 3x3 figure
    plot_combined_distributions(all_stats_map, title_prefixes)