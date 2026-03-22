import json
import numpy as np
import re
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

# 设置随机种子以保证结果可重复
np.random.seed(42)


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# 这个函数返回4个值
def extract_features_and_labels(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[str]]:
    """从论文数据中提取特征、标签以及对应的有效论文对象"""
    features_list = []
    labels_list = []
    valid_papers = []

    feature_names = [
        'avg_rating', 'min_rating', 'max_rating', 'rating_std',
        'avg_confidence', 'min_confidence', 'max_confidence'
    ]

    for paper in data:
        ratings, confidences = [], []
        for review in paper.get('reviews', []):
            rating_str = review.get('rating', '')
            if rating_str and rating_str != '-1':
                rating_match = re.search(r'^(\d+\.?\d*)', rating_str)
                if rating_match:
                    ratings.append(float(rating_match.group(1)))

            confidence_str = review.get('confidence', '')
            if confidence_str and confidence_str != '-1':
                confidence_match = re.search(r'^(\d+\.?\d*)', confidence_str)
                if confidence_match:
                    confidences.append(float(confidence_match.group(1)))

        if not ratings:
            continue

        decision = paper.get('paper_decision', '').lower()
        if 'accept' in decision:
            label = 1
        elif 'reject' in decision or 'withdraw' in decision:
            label = 0
        else:
            continue

        avg_rating = np.mean(ratings)
        min_rating = np.min(ratings)
        max_rating = np.max(ratings)
        rating_std = np.std(ratings) if len(ratings) > 1 else 0

        if confidences:
            avg_confidence, min_confidence, max_confidence = np.mean(confidences), np.min(confidences), np.max(
                confidences)
        else:
            avg_confidence, min_confidence, max_confidence = 0, 0, 0

        features_list.append([
            avg_rating, min_rating, max_rating, rating_std,
            avg_confidence, min_confidence, max_confidence
        ])
        labels_list.append(label)
        valid_papers.append(paper)

    return np.array(features_list), np.array(labels_list), valid_papers, feature_names


def evaluate_model_accuracy(model, X, y):
    """评估模型性能，只计算和打印准确率"""
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"评估样本数: {len(y)}")
    print(f"模型准确率: {accuracy:.4f}")


def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)

    print("\n--- 特征重要性分析 ---")
    for feature, importance in sorted_importance:
        print(f"{feature:<20}: {importance:.4f}")


def create_detailed_predictions_jsonl(
        model: RandomForestClassifier,
        valid_papers: List[Dict],
        X_features: np.ndarray
) -> List[Dict]:
    """为有效论文列表创建包含预测结果的详细数据集"""
    print("\n生成详细的预测数据集 (JSONL 格式)...")
    if not valid_papers:
        print("没有有效数据来生成详细数据集。")
        return []

    predictions = model.predict(X_features)
    decision_map = {0: 'Reject/Withdrawn', 1: 'Accept'}
    predicted_decisions_str = [decision_map.get(p, 'Unknown') for p in predictions]

    jsonl_data = []
    for i, paper in enumerate(valid_papers):
        paper_data = {
            'paper_title': paper.get('paper_title', 'N/A'),
            'real_decision': paper.get('paper_decision', 'N/A'),
            'predict_decision': predicted_decisions_str[i],
            'reviews': [],
            'method':'RF'
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
    # 1. 修改：将单个字符串改为包含多个文件路径的列表
    train_files = [
        '../../../Datasets/formatted_dataset/ICLR.cc_2022_formatted.jsonl',
        '../../../Datasets/formatted_dataset/ICLR.cc_2023_formatted.jsonl',
        '../../../Datasets/formatted_dataset/ICLR.cc_2024_formatted.jsonl'
    ]

    test_val_file = '../../../Datasets/formatted_dataset/ICLR.cc_2025_formatted.jsonl'

    try:
        train_data = []

        for file_path in train_files:
            print(f"正在从 {file_path} 加载训练数据...")
            train_data.extend(load_jsonl(file_path))

        print(f"所有训练数据加载完毕，共 {len(train_data)} 条。")

        print(f"正在从 {test_val_file} 加载验证和测试数据...")
        file_2_data = load_jsonl(test_val_file)

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。")
        return
    except json.JSONDecodeError as e:
        print(f"错误: 文件格式不正确。错误详情: {e}")
        return

    print("加载数据...")


    indices = np.random.permutation(len(file_2_data))
    val_indices, test_indices = indices[:1000], indices[1000:]
    val_data = [file_2_data[i] for i in val_indices]
    test_data = [file_2_data[i] for i in test_indices]

    print(f"训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条, 测试集: {len(test_data)} 条")

    print("\n提取特征...")
    X_train, y_train, _, feature_names = extract_features_and_labels(train_data)
    X_val, y_val, _, _ = extract_features_and_labels(val_data)
    X_test, y_test, test_data_valid, _ = extract_features_and_labels(test_data)

    print(f"训练集特征形状: {X_train.shape}, 标签分布: {np.bincount(y_train) if len(y_train) > 0 else '[]'}")
    if len(y_val) > 0: print(f"验证集特征形状: {X_val.shape}, 标签分布: {np.bincount(y_val)}")
    if len(y_test) > 0: print(f"测试集特征形状: {X_test.shape}, 标签分布: {np.bincount(y_test)}")

    print("\n开始训练随机森林模型...")
    model = RandomForestClassifier(n_estimators=200, min_samples_split=10, min_samples_leaf=2,
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("模型训练完成。")

    if len(y_val) > 0:
        print("\n=== 验证集评估 ===")
        evaluate_model_accuracy(model, X_val, y_val)

    if len(y_test) > 0:
        print("\n=== 测试集评估 ===")
        evaluate_model_accuracy(model, X_test, y_test)

    analyze_feature_importance(model, feature_names)

    if test_data_valid:
        detailed_jsonl = create_detailed_predictions_jsonl(model, test_data_valid, X_test)
        if detailed_jsonl:
            output_filename = '../../../Datasets/predict_result_new/rating/RF.jsonl'
            with open(output_filename, 'w', encoding='utf-8') as f:
                for entry in detailed_jsonl:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"\n详细数据集已成功保存到: {output_filename}")


if __name__ == "__main__":
    main()