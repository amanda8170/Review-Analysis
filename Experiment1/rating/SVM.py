import json
import numpy as np
import re
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')
np.random.seed(42)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class PaperDecisionSVM:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features_and_labels(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[str]]:
        """
        Extract features, labels, and the corresponding valid paper objects.
        """
        features_list = []
        labels_list = []
        valid_papers_list = []  # 新增：用于存储被成功处理的论文

        feature_names = [
            'rating_mean', 'rating_std', 'rating_min', 'rating_max', 'rating_median',
            'confidence_mean', 'confidence_std', 'confidence_min', 'confidence_max', 'confidence_median',
        ]

        for paper in data:
            ratings, confidences = [], []
            for review in paper.get('reviews', []):
                rating_str = review.get('rating', '')
                confidence_str = review.get('confidence', '')

                if rating_str and rating_str != '-1':
                    rating_match = re.search(r'^(\d+\.?\d*)', rating_str)
                    if rating_match: ratings.append(float(rating_match.group(1)))

                if confidence_str and confidence_str != '-1':
                    confidence_match = re.search(r'^(\d+\.?\d*)', confidence_str)
                    if confidence_match: confidences.append(float(confidence_match.group(1)))

            if not ratings:
                continue

            feature_vector = [
                np.mean(ratings), np.std(ratings) if len(ratings) > 1 else 0.0,
                np.min(ratings), np.max(ratings), np.median(ratings),
                np.mean(confidences) if confidences else 0.0,
                np.std(confidences) if len(confidences) > 1 else 0.0,
                np.min(confidences) if confidences else 0.0,
                np.max(confidences) if confidences else 0.0,
                np.median(confidences) if confidences else 0.0,
                len(ratings)
            ]

            decision = paper.get('paper_decision', '').lower()
            if 'accept' in decision:
                label = 1
            elif 'reject' in decision or 'withdraw' in decision:
                label = 0
            else:
                continue

            features_list.append(feature_vector)
            labels_list.append(label)
            valid_papers_list.append(paper)  # 将有效论文的原始数据加入列表

        return np.array(features_list), np.array(labels_list), valid_papers_list, feature_names

    def train(self, X_train, y_train):
        """
        Train the SVM model using Fixed Parameters (No Search).
        """
        print("\n开始训练SVM模型 (使用固定参数)...")

        # 1. 标准化数据
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = SVC(
            C=10,  # 正则化强度
            kernel='linear',  # 核函数
            gamma=1,  # 核系数
            random_state=42
        )

        # 3. 拟合模型
        self.model.fit(X_train_scaled, y_train)

        self.is_trained = True
        print("模型训练完成！")

    def evaluate(self, X_test, y_test, set_name=""):
        """
        Evaluate the trained model.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please call train() first.")
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n=== {set_name} 评估结果 ===")
        print(f"评估样本数: {len(y_test)}")
        print(f"模型准确率: {accuracy:.4f}")

    # --- 新增：在类中添加生成数据集的方法 ---
    def create_detailed_predictions_jsonl(self, valid_papers: List[Dict], X_features: np.ndarray) -> List[Dict]:
        """
        Creates a detailed dataset in a list-of-dicts format suitable for JSONL.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please call train() first.")

        print("\n生成详细的预测数据集 (JSONL 格式)...")
        if not valid_papers:
            print("没有有效数据来生成详细数据集。")
            return []

        # 1. Predict decisions for all papers
        X_features_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_features_scaled)
        decision_map = {0: 'Reject/Withdrawn', 1: 'Accept'}
        predicted_decisions_str = [decision_map[p] for p in predictions]

        # 2. Build the hierarchical dictionary for each paper
        jsonl_data = []
        for i, paper in enumerate(valid_papers):
            paper_data = {
                'paper_title': paper.get('paper_title', 'N/A'),
                'real_decision': paper.get('paper_decision', 'N/A'),
                'predict_decision': predicted_decisions_str[i],
                'reviews': [],
                'method':'SVM'
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

    indices = np.random.permutation(len(file_2_data))
    val_indices = indices[:1000]
    test_indices = indices[1000:]
    val_data = [file_2_data[i] for i in val_indices]
    test_data = [file_2_data[i] for i in test_indices]

    print(f"训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条, 测试集: {len(test_data)} 条")

    svm_classifier = PaperDecisionSVM()

    print("\n提取特征...")
    # --- 修改：接收第四个返回值 valid_papers ---
    X_train, y_train, _, _ = svm_classifier.extract_features_and_labels(train_data)
    X_val, y_val, _, _ = svm_classifier.extract_features_and_labels(val_data)
    # 只有测试集需要保留 valid_papers 用于生成最终报告
    X_test, y_test, test_data_valid, _ = svm_classifier.extract_features_and_labels(test_data)

    svm_classifier.train(X_train, y_train)

    if len(y_val) > 0:
        svm_classifier.evaluate(X_val, y_val, set_name="验证集")

    if len(y_test) > 0:
        svm_classifier.evaluate(X_test, y_test, set_name="测试集")

    # --- 新增：生成并保存详细的 JSONL 数据集 ---
    if len(y_test) > 0 and test_data_valid:
        # 调用新方法生成数据
        detailed_jsonl = svm_classifier.create_detailed_predictions_jsonl(test_data_valid, X_test)

        if detailed_jsonl:
            output_filename = '../../../Datasets/predict_result_new/rating/SVM.jsonl'
            with open(output_filename, 'w', encoding='utf-8') as f:
                for entry in detailed_jsonl:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"\n详细数据集已成功保存到: {output_filename}")



if __name__ == "__main__":
    main()