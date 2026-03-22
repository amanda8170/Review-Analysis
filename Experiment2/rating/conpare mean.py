import json
import numpy as np
from scipy import stats
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def analyze_jsonl_stats(file_path):
    """
    Analyzes a jsonl file to extract statistics for prediction confidences and review ratings.
    This version has no try...except blocks and will raise an error on file/data issues.
    """
    per_paper_mean_confidences = []
    per_paper_mean_ratings = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            data = json.loads(line)

            predictions = data.get('predictions', [])
            confidences = [p.get('predict_confidence', 0) for p in predictions if 'predict_confidence' in p]
            if confidences:
                per_paper_mean_confidences.append(np.mean(confidences))

            reviews = data.get('reviews', [])
            ratings = [float(r.get('rating', 0)) for r in reviews if 'rating' in r]
            if ratings:
                per_paper_mean_ratings.append(np.mean(ratings))

    if not per_paper_mean_confidences or not per_paper_mean_ratings:
        print(f"Warning: No valid data found for analysis in file '{file_path}'.")
        return None

    confidences_array = np.array(per_paper_mean_confidences)
    ratings_array = np.array(per_paper_mean_ratings)

    results = {
        "file_name": os.path.basename(file_path),
        "total_papers": len(per_paper_mean_confidences),
        "prediction_confidence": {
            "mean": np.mean(confidences_array),
            "variance": np.var(confidences_array),
            "std_dev": np.std(confidences_array),
            "skewness": stats.skew(confidences_array),
            "kurtosis": stats.kurtosis(confidences_array)
        },
        "review_rating": {
            "mean": np.mean(ratings_array),
            "variance": np.var(ratings_array),
            "std_dev": np.std(ratings_array),
            "skewness": stats.skew(ratings_array),
            "kurtosis": stats.kurtosis(ratings_array)
        },
        "raw_mean_confidences": per_paper_mean_confidences,
        "raw_mean_ratings": per_paper_mean_ratings
    }
    return results


def print_comparison(*all_stats):
    """Prints a comparison of statistics for any number of files in a table format."""
    valid_stats = [s for s in all_stats if s]
    if not valid_stats:
        print("No statistics provided for comparison.")
        return

    data = {'Metric': ['--- Prediction Confidence ---', 'Mean', 'Variance', 'Std Dev',
                       'Skewness', 'Kurtosis', '', '--- Review Rating ---', 'Mean',
                       'Variance', 'Std Dev', 'Skewness', 'Kurtosis', '',
                       '--- General ---', 'Total Papers Analyzed']}
    for stats_item in valid_stats:
        column_name = stats_item['file_name']
        data[column_name] = ['', f"{stats_item['prediction_confidence']['mean']:.4f}",
                             f"{stats_item['prediction_confidence']['variance']:.4f}",
                             f"{stats_item['prediction_confidence']['std_dev']:.4f}",
                             f"{stats_item['prediction_confidence']['skewness']:.4f}",
                             f"{stats_item['prediction_confidence']['kurtosis']:.4f}", '', '',
                             f"{stats_item['review_rating']['mean']:.4f}",
                             f"{stats_item['review_rating']['variance']:.4f}",
                             f"{stats_item['review_rating']['std_dev']:.4f}",
                             f"{stats_item['review_rating']['skewness']:.4f}",
                             f"{stats_item['review_rating']['kurtosis']:.4f}", '', '', stats_item['total_papers']]
    df = pd.DataFrame(data)
    print("\n" + "=" * 80);
    print(" " * 25 + "File Statistics Comparison");
    print("=" * 80)
    print(df.to_string(index=False));
    print("=" * 80 + "\n")


# --- Final Plotting Function (with Custom Titles) ---
def plot_distributions(*all_stats):
    """Plots distributions with custom titles, smaller figure size, and larger font."""
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 14})

    valid_stats = [s for s in all_stats if s]
    num_files = len(valid_stats)
    if num_files == 0:
        print("No valid statistics available for plotting.")
        return

    # --- Define the custom titles for the plots in order ---
    title_prefixes = ["Hard Sample", "Simple Accept Sample", "Simple Reject Sample"]

    fig, axes = plt.subplots(nrows=2, ncols=num_files, figsize=(6 * num_files, 9), squeeze=False)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # --- Confidence Plot ---
    max_y_conf_percent = 0
    for i, stats_item in enumerate(valid_stats):
        ax_conf = axes[0, i]
        sns.histplot(stats_item['raw_mean_confidences'], ax=ax_conf, stat='percent', bins=30,
                     color=colors[i % len(colors)], kde=True)
        max_y_conf_percent = max(max_y_conf_percent, ax_conf.get_ylim()[1])

    for i, stats_item in enumerate(valid_stats):
        ax_conf = axes[0, i];
        ax_conf.clear()
        total_papers_conf = stats_item['total_papers']
        mean_confidence = stats_item['prediction_confidence']['mean']
        sns.histplot(stats_item['raw_mean_confidences'], ax=ax_conf, stat='percent', bins=30,
                     color=colors[i % len(colors)], kde=True)
        ax_conf.set_ylim(0, max_y_conf_percent * 1.05)
        ax_conf.set_ylabel("Paper Percentage (%)")
        ax_conf_count = ax_conf.twinx()
        ax_conf_count.set_ylabel("Paper Count")
        y_limit_count_conf = (ax_conf.get_ylim()[1] / 100) * total_papers_conf
        ax_conf_count.set_ylim(0, y_limit_count_conf)
        ax_conf.axvline(mean_confidence, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_confidence:.3f}')
        ax_conf.legend()

        # --- Use custom title ---
        title_prefix = title_prefixes[i] if i < len(title_prefixes) else stats_item['file_name']
        ax_conf.set_title(f"{title_prefix}\nPrediction Confidence Distribution")

        ax_conf.set_xlabel("Average Confidence Score")
        ax_conf.set_xlim(0.5, 1.0)

    # --- Rating Plot ---
    max_y_rating_percent = 0
    rating_bins = np.arange(0, 10.5, 0.5)

    for i, stats_item in enumerate(valid_stats):
        ax_rating = axes[1, i]
        raw_ratings = stats_item['raw_mean_ratings']
        sns.histplot(raw_ratings, ax=ax_rating, bins=rating_bins, stat='percent', kde=False)
        max_y_rating_percent = max(max_y_rating_percent, ax_rating.get_ylim()[1])

    for i, stats_item in enumerate(valid_stats):
        ax_rating = axes[1, i];
        ax_rating.clear()
        total_papers_rating = stats_item['total_papers']
        mean_rating = stats_item['review_rating']['mean']
        raw_ratings = stats_item['raw_mean_ratings']

        sns.histplot(raw_ratings, ax=ax_rating, bins=rating_bins,
                     stat='percent', color=colors[i % len(colors)], kde=False)

        counts, bin_edges = np.histogram(raw_ratings, bins=rating_bins)

        if total_papers_rating > 0:
            percentages = (counts / total_papers_rating) * 100
        else:
            percentages = np.zeros_like(counts)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax_rating.plot(bin_centers, percentages, color=colors[i % len(colors)], linewidth=2)
        ax_rating.set_ylabel("Paper Percentage (%)")
        ax_rating.set_ylim(0, max_y_rating_percent * 1.05)

        ax_rating_count = ax_rating.twinx()
        ax_rating_count.set_ylabel("Paper Count")
        y_limit_count_rating = (ax_rating.get_ylim()[1] / 100) * total_papers_rating
        ax_rating_count.set_ylim(0, y_limit_count_rating)

        ax_rating.axvline(x=mean_rating, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_rating:.3f}')
        ax_rating.legend()

        # --- Use custom title ---
        title_prefix = title_prefixes[i] if i < len(title_prefixes) else stats_item['file_name']
        ax_rating.set_title(f"{title_prefix}\nReview Rating Distribution")

        ax_rating.set_xlabel("Average Review Score")
        ax_rating.set_xticks(range(1, 11))
        ax_rating.set_xlim(0.5, 10.5)

    plt.tight_layout(pad=3.0)
    output_filename = 'score_distributions_custom_titles.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Distribution plots saved as '{output_filename}'")
    plt.show()


# --- Main Program ---
if __name__ == "__main__":
    file1_path = '../../../Datasets/hard example/final_difficult_sample_pool.jsonl'
    file2_path = '../../../Datasets/hard example/final_simple_accept_sample_pool.jsonl'
    file3_path = '../../../Datasets/hard example/final_simple_reject_sample_pool.jsonl'

    stats1 = analyze_jsonl_stats(file1_path)
    stats2 = analyze_jsonl_stats(file2_path)
    stats3 = analyze_jsonl_stats(file3_path)

    print_comparison(stats1, stats2, stats3)
    plot_distributions(stats1, stats2, stats3)