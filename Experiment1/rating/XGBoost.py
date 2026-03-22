import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
from typing import List, Dict, Tuple

# Set random seed for reproducibility
np.random.seed(42)
warnings.filterwarnings('ignore')


def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# --- MODIFIED: The function now returns valid_papers as well ---
def extract_features_and_labels(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Extract features, labels, and the corresponding valid paper objects"""
    features = []
    labels = []
    valid_papers = []  # To store papers that are successfully processed

    for item in data:
        decision = item.get('paper_decision', '').lower()
        if 'accept' in decision:
            label = 1
        elif 'reject' in decision or 'withdraw' in decision:
            label = 0
        else:
            continue

        reviews = item.get('reviews', [])
        ratings, confidences = [], []

        for review in reviews:
            rating_str = review.get('rating', '')
            if rating_str and rating_str != '-1':
                try:
                    rating_num = float(rating_str.split(':')[0])
                    ratings.append(rating_num)
                except ValueError:  # Changed to more specific exception
                    continue

            confidence_str = review.get('confidence', '')
            if confidence_str and confidence_str != '-1':
                try:
                    confidence_num = float(confidence_str.split(':')[0])
                    confidences.append(confidence_num)
                except ValueError:
                    continue

        if not ratings:
            continue

        feature_vector = [
            np.mean(ratings), np.min(ratings), np.max(ratings),
            np.std(ratings) if len(ratings) > 1 else 0, np.median(ratings),
        ]
        if confidences:
            feature_vector.extend([
                np.mean(confidences), np.min(confidences), np.max(confidences),
                np.std(confidences) if len(confidences) > 1 else 0, np.median(confidences),
            ])
        else:
            feature_vector.extend([0] * 5)

        features.append(feature_vector)
        labels.append(label)
        valid_papers.append(item)  # Add the corresponding paper

    return np.array(features), np.array(labels), valid_papers


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with validation"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        'colsample_bytree': 0.8,'learning_rate': 0.05,'max_depth': 3,'n_estimators': 100,'subsample': 1
    }
    model = xgb.train(
        params, dtrain, num_boost_round=100,
        evals=[(dval, 'validation')],
        early_stopping_rounds=10, verbose_eval=False
    )
    return model


def evaluate_model_accuracy(model, X, y, set_name=""):
    """Evaluate model performance and print only the accuracy"""
    dtest = xgb.DMatrix(X)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y, y_pred)
    print(f"\n=== {set_name} 评估 ===")
    print(f"评估样本数: {len(y)}")
    print(f"模型准确率: {accuracy:.4f}")


# --- NEW FUNCTION: To create the detailed dataset ---
def create_detailed_jsonl_dataset(model, X_features: np.ndarray, valid_papers: List[Dict]) -> List[Dict]:
    """
    Creates a detailed dataset with predictions in a format suitable for JSONL.
    """
    print("\n生成详细的预测数据集 (JSONL 格式)...")
    if not valid_papers:
        print("没有有效数据来生成详细数据集。")
        return []

    # 1. Predict decisions for all papers
    dmatrix = xgb.DMatrix(X_features)
    y_pred_proba = model.predict(dmatrix)
    y_pred = (y_pred_proba > 0.5).astype(int)
    decision_map = {0: 'Reject/Withdrawn', 1: 'Accept'}
    predicted_decisions_str = [decision_map[p] for p in y_pred]

    # 2. Build the hierarchical dictionary for each paper
    jsonl_data = []
    for i, paper in enumerate(valid_papers):
        paper_data = {
            'paper_title': paper.get('paper_title', 'N/A'),
            'real_decision': paper.get('paper_decision', 'N/A'),
            'predict_decision': predicted_decisions_str[i],
            'reviews': [],
            'method':'Xgboost'
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
            train_data.extend(load_jsonl_data(file_path))

        print(f"所有训练数据加载完毕，共 {len(train_data)} 条。")

        print(f"正在从 {test_val_file} 加载验证和测试数据...")
        file_2_data = load_jsonl_data(test_val_file)

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。")
        return
    except json.JSONDecodeError as e:
        print(f"错误: 文件格式不正确。错误详情: {e}")
        return


    print("分割数据...")
    indices = np.random.permutation(len(file_2_data))
    val_indices = indices[:1000]
    test_indices = indices[1000:]
    val_data = [file_2_data[i] for i in val_indices]
    test_data = [file_2_data[i] for i in test_indices]

    print(f"训练集: {len(train_data)} 篇, 验证集: {len(val_data)} 篇, 测试集: {len(test_data)} 篇")

    print("\n提取特征...")
    # --- MODIFIED: Capture the valid_papers list for the test set ---
    X_train, y_train, _ = extract_features_and_labels(train_data)
    X_val, y_val, _ = extract_features_and_labels(val_data)
    X_test, y_test, test_data_valid = extract_features_and_labels(test_data)

    print(f"特征矩阵形状: 训练 {X_train.shape}, 验证 {X_val.shape}, 测试 {X_test.shape}")

    print("\n训练XGBoost模型...")
    if len(y_val) == 0:
        print("错误：验证集为空，无法进行模型训练。")
        return

    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    print("模型训练完成。")

    evaluate_model_accuracy(model, X_test, y_test, "测试集")

    if len(y_test) > 0 and test_data_valid:
        detailed_jsonl = create_detailed_jsonl_dataset(model, X_test, test_data_valid)

        if detailed_jsonl:
            output_filename = '../../../Datasets/predict_result_new/rating/XGBoost.jsonl'
            with open(output_filename, 'w', encoding='utf-8') as f:
                for entry in detailed_jsonl:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"\n详细数据集已成功保存到: {output_filename}")

if __name__ == "__main__":
    main()