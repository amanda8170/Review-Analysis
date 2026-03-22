import numpy as np
import json
from typing import List, Dict, Tuple


def load_jsonl(file_path: str) -> List[Dict]:
    """
    从JSONL文件中加载数据。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"警告：跳过格式错误的行: {line.strip()}")
                    continue
    return data


def extract_ratings(paper_data: Dict) -> List[float]:
    """
    从单个paper数据中提取所有有效的rating值。
    """
    ratings = []
    if 'reviews' in paper_data:
        for review in paper_data['reviews']:
            rating_str = review.get('rating', '')
            if rating_str and rating_str != "-1":
                try:
                    rating_value = float(rating_str.split(':')[0].strip()) if ':' in rating_str else float(rating_str)
                    ratings.append(rating_value)
                except (ValueError, TypeError):
                    continue
    return ratings


def get_true_label(paper_data: Dict) -> int:
    """
    获取论文的真实标签（1 for Accept, 0 for Reject）。
    """
    decision = paper_data.get('paper_decision', '').lower()
    if 'accept' in decision:
        return 1
    elif 'reject' in decision or 'withdraw' in decision:
        return 0
    return None


def predict_with_rating_threshold(paper_data: Dict, threshold: float = 5.8) -> int:
    """
    基于rating均值阈值进行预测。
    """
    ratings = extract_ratings(paper_data)
    if not ratings:
        return None
    mean_rating = np.mean(ratings)
    return 1 if mean_rating > threshold else 0


def calculate_accuracy(data: List[Dict], threshold: float = 5.8) -> float:
    """
    评估阈值分类器的准确率。
    """
    correct_predictions, total_predictions = 0, 0
    for paper in data:
        true_label = get_true_label(paper)
        predicted_label = predict_with_rating_threshold(paper, threshold)
        if true_label is not None and predicted_label is not None:
            total_predictions += 1
            if predicted_label == true_label:
                correct_predictions += 1
    if total_predictions == 0:
        print("警告：数据集中没有可用于评估的有效样本。")
        return 0.0
    return correct_predictions / total_predictions


# --- NEW FUNCTION: To prepare data for the report ---
def prepare_dataset_for_baseline(data: List[Dict]) -> List[Dict]:
    """筛选出可以被基线模型处理的论文"""
    valid_papers = []
    for paper in data:
        if get_true_label(paper) is not None and extract_ratings(paper):
            valid_papers.append(paper)
    return valid_papers


# --- NEW FUNCTION: To create the detailed dataset ---
def create_detailed_predictions_jsonl_for_baseline(
        valid_papers: List[Dict],
        threshold: float
) -> List[Dict]:
    """为有效论文列表创建包含预测结果的详细数据集"""
    print("\n生成详细的预测数据集 (JSONL 格式)...")
    if not valid_papers:
        print("没有有效数据来生成详细数据集。")
        return []

    jsonl_data = []
    for paper in valid_papers:
        predicted_label_int = predict_with_rating_threshold(paper, threshold)
        predicted_decision_str = "Accept" if predicted_label_int == 1 else "Reject/Withdrawn"

        paper_data = {
            'paper_title': paper.get('paper_title', 'N/A'),
            'real_decision': paper.get('paper_decision', 'N/A'),
            'predict_decision': predicted_decision_str,
            'method':'threshold',
            'reviews': []
        }
        for review in paper.get('reviews', []):
            rating_val = review.get('rating', '')
            confidence_val = review.get('confidence', '')
            if rating_val and rating_val != '-1' and confidence_val and confidence_val != '-1':
                paper_data['reviews'].append({
                    'reviewer': review.get('reviewer', 'N/A'),
                    'rating': rating_val,
                    'confidence': confidence_val
                })
        jsonl_data.append(paper_data)

    print(f"成功为 {len(jsonl_data)} 篇论文生成了详细记录。")
    return jsonl_data


def main():
    """
    主函数：加载数据，划分测试集，并评估基线模型。
    """
    test_val_file = '../../../Datasets/formatted_dataset/ICLR.cc_2025_formatted.jsonl'

    try:
        print(f"正在从 {test_val_file} 加载数据...")
        file_2_data = load_jsonl(test_val_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {test_val_file}。")
        return

    np.random.seed(42)
    indices = np.random.permutation(len(file_2_data))
    test_indices = indices[1000:]
    test_data = [file_2_data[i] for i in test_indices]

    print(f"\n已准备好 {len(test_data)} 条测试数据。")

    # Define rule parameter
    mean_rating_threshold = 5.8

    print(f"--- 开始评估基线模型 (规则: 平均分 > {mean_rating_threshold} 则接受) ---")

    accuracy = calculate_accuracy(test_data, threshold=mean_rating_threshold)

    print("\n=== 测试集评估结果 ===")
    print(f"模型准确率 (Accuracy): {accuracy:.4f}")

    # --- NEW: Generate and save the detailed dataset ---
    test_data_valid = prepare_dataset_for_baseline(test_data)
    if test_data_valid:
        detailed_jsonl = create_detailed_predictions_jsonl_for_baseline(test_data_valid, mean_rating_threshold)
        if detailed_jsonl:
            output_filename = '../../../Datasets/predict_result_new/rating/threshold.jsonl'
            with open(output_filename, 'w', encoding='utf-8') as f:
                for entry in detailed_jsonl:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"\n详细数据集已成功保存到: {output_filename}")



if __name__ == "__main__":
    main()