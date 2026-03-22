import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


def load_jsonl(file_path: str) -> List[Dict]:
    """
    从JSONL文件中加载数据。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class PaperDecisionClassifier:
    def __init__(self):
        self.model = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    # --- 步骤 1: 特征提取函数返回三个值 ---
    def extract_features_and_labels(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        从论文数据中提取特征、标签以及对应的有效论文对象。
        """
        features = []
        labels = []
        valid_papers = []  # 用于存储被成功处理的论文

        for paper in data:
            ratings = []
            for review in paper.get('reviews', []):
                rating_str = review.get('rating', '')
                if rating_str and rating_str != '-1':
                    try:
                        rating_num = float(rating_str.split(':')[0]) if ':' in rating_str else float(rating_str)
                        ratings.append(rating_num)
                    except (ValueError, TypeError):
                        continue

            if ratings:
                decision = paper.get('paper_decision', '').lower()
                if 'reject' in decision:
                    label = 0
                elif 'accept' in decision:
                    label = 1
                else:
                    continue

                avg_rating = np.mean(ratings)

                features.append(avg_rating)
                labels.append(label)
                valid_papers.append(paper)  # 添加对应的论文对象

        return np.array(features).reshape(-1, 1), np.array(labels), valid_papers

    def train(self, train_data: List[Dict]):
        """
        训练逻辑回归模型
        """
        print("正在提取训练数据特征 (评分均值)...")
        X_train, y_train, _ = self.extract_features_and_labels(train_data)

        if len(X_train) == 0:
            print("错误：训练数据中没有找到有效的样本。")
            return self

        print(f"训练数据: {len(X_train)} 个样本")
        print(f"Accept: {np.sum(y_train)} 个, Reject: {len(y_train) - np.sum(y_train)} 个")

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        print("模型训练完成！")
        return self

    def predict(self, test_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        对测试数据进行预测
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")

        X_test, y_test, _ = self.extract_features_and_labels(test_data)

        if len(X_test) == 0:
            print("警告：测试数据中没有找到有效的样本。")
            return np.array([]), np.array([])

        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)

        return predictions, y_test

    def evaluate(self, test_data: List[Dict]) -> float:
        """
        评估模型性能，仅计算并打印准确率。
        """
        predictions, y_true = self.predict(test_data)
        if len(y_true) == 0:
            print("无法进行评估，因为没有有效的测试样本。")
            return 0.0
        accuracy = accuracy_score(y_true, predictions)
        print(f"\n--- 模型评估 ---")
        print(f"测试样本数: {len(y_true)}")
        print(f"模型准确率: {accuracy:.4f}")
        return accuracy

    def create_detailed_predictions_jsonl(self, valid_papers: List[Dict], X_features: np.ndarray) -> List[Dict]:
        """为有效论文列表创建包含预测结果的详细数据集"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")

        print("\n生成详细的预测数据集 (JSONL 格式)...")
        if not valid_papers:
            print("没有有效数据来生成详细数据集。")
            return []

        # 确保 X_features 是二维的，以供 scaler 使用
        if X_features.ndim == 1:
            X_features = X_features.reshape(-1, 1)

        X_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_scaled)

        decision_map = {0: 'Reject/Withdrawn', 1: 'Accept'}
        predicted_decisions_str = [decision_map.get(p, 'Unknown') for p in predictions]

        jsonl_data = []
        for i, paper in enumerate(valid_papers):
            paper_data = {
                'paper_title': paper.get('paper_title', 'N/A'),
                'real_decision': paper.get('paper_decision', 'N/A'),
                'predict_decision': predicted_decisions_str[i],
                'reviews': [],
                'method':'LR'
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


    np.random.seed(42)
    indices = np.random.permutation(len(file_2_data))
    val_indices, test_indices = indices[:1000], indices[1000:]
    val_data = [file_2_data[i] for i in val_indices]
    test_data = [file_2_data[i] for i in test_indices]

    classifier = PaperDecisionClassifier()
    classifier.train(train_data)

    if not classifier.is_trained:
        return

    if val_data:
        print("\n=== 验证集评估 ===")
        classifier.evaluate(val_data)

    if test_data:
        print("\n=== 测试集评估 ===")
        classifier.evaluate(test_data)

    if test_data:
        X_test, y_test, test_data_valid = classifier.extract_features_and_labels(test_data)

        if test_data_valid:
            detailed_jsonl = classifier.create_detailed_predictions_jsonl(test_data_valid, X_test)
            if detailed_jsonl:
                # 定义输出文件名
                output_filename = '../../../Datasets/predict_result_new/rating/LR_MeanRating.jsonl'
                # 写入文件
                with open(output_filename, 'w', encoding='utf-8') as f:
                    for entry in detailed_jsonl:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

                print(f"\n详细数据集已成功保存到: {output_filename}")

    print(f"\n=== 模型参数 ===")
    print(f"截距 (Intercept): {classifier.model.intercept_[0]:.4f}")
    print(f"评分均值的系数 (Coefficient): {classifier.model.coef_[0][0]:.4f}")


if __name__ == "__main__":
    main()