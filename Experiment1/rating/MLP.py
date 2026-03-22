import json
import numpy as np
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from typing import List, Dict, Tuple

# Set random seed for reproducibility
np.random.seed(42)
warnings.filterwarnings('ignore')


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class PaperMLPClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features_and_labels(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Extract features from paper data.
        Returns:
            - features: A numpy array of feature vectors for each valid paper.
            - labels: A numpy array of labels (0 or 1) for each valid paper.
            - valid_papers: A list of the original paper dictionaries that were successfully processed.
        """
        features = []
        labels = []
        valid_papers = []

        for paper in data:
            decision = paper.get('paper_decision', '').lower()
            if 'accept' in decision:
                label = 1
            elif 'reject' in decision or 'withdrawn' in decision:
                label = 0
            else:
                continue

            reviews = paper.get('reviews', [])
            ratings, confidences = [], []

            for review in reviews:
                rating = review.get('rating', '')
                if rating and rating != '-1':
                    try:
                        rating_num = float(rating.split(':')[0]) if ':' in rating else float(rating)
                        ratings.append(rating_num)
                    except (ValueError, TypeError):
                        pass

                confidence = review.get('confidence', '')
                if confidence and confidence != '-1':
                    try:
                        conf_num = float(confidence.split(':')[0]) if ':' in confidence else float(confidence)
                        confidences.append(conf_num)
                    except (ValueError, TypeError):
                        pass

            if not ratings or not confidences:
                continue

            feature_vector = [
                np.mean(ratings), np.std(ratings) if len(ratings) > 1 else 0,
                np.min(ratings), np.max(ratings), np.median(ratings),
                np.mean(confidences), np.std(confidences) if len(confidences) > 1 else 0,
                np.min(confidences), np.max(confidences), np.median(confidences),
                len(ratings)
            ]

            features.append(feature_vector)
            labels.append(label)
            valid_papers.append(paper)

        return np.array(features), np.array(labels), valid_papers

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the MLP model using Fixed Parameters.
        """
        print("\n开始训练MLP模型 (使用固定参数)...")

        # 1. 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 2. 直接定义模型，填入你想要的固定参数
        self.model = MLPClassifier(
            hidden_layer_sizes=(100,),  # 隐藏层神经元数量，可以改，例如 (50, 25)
            alpha=0.0001,  # 正则化系数
            learning_rate_init=0.01,  # 初始学习率
            random_state=42,
            early_stopping=True,  # 保持早停以防止过拟合
        )

        # 3. 训练模型
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        print("模型训练完成！")

    def evaluate(self, X: np.ndarray, y: np.ndarray, set_name: str = ""):
        """Evaluate the trained model on a given dataset."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Please call tune_and_train() first.")

        print(f"\n=== {set_name} 评估 ===")
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        print(f"评估样本数: {len(y)}")
        print(f"模型准确率: {accuracy:.4f}")
        return accuracy

    # --- MODIFIED: 增加了置信度输出 ---
    def create_detailed_predictions_jsonl(self, valid_papers: List[Dict], X_features: np.ndarray) -> List[Dict]:
        """Creates a detailed dataset with predictions and confidences."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Please call tune_and_train() first.")

        print("\n生成详细的预测数据集 (JSONL 格式)...")
        if not valid_papers:
            print("没有有效数据来生成详细数据集。")
            return []

        X_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidences = np.max(probabilities, axis=1)

        decision_map = {0: 'Reject/Withdrawn', 1: 'Accept'}

        jsonl_data = []
        for i, paper in enumerate(valid_papers):
            paper_data = {
                'paper_title': paper.get('paper_title', 'N/A'),
                'real_decision': paper.get('paper_decision', 'N/A'),
                'predict_decision': decision_map.get(predictions[i], 'Unknown'),
                'predict_confidence': float(confidences[i]),
                'reviews': [],
                'method': 'MLP'
            }
            for review in paper.get('reviews', []):
                rating_val = review.get('rating', '')
                confidence_val = review.get('confidence', '')
                if rating_val and rating_val != '-1':
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

    try:
        print("划分验证集和测试集...")
        indices = np.random.permutation(len(file_2_data))
        val_indices = indices[:1000]  # 前1000条做验证
        test_indices = indices[1000:]  # 剩下的做测试

        val_data = [file_2_data[i] for i in val_indices]
        test_data = [file_2_data[i] for i in test_indices]

        print("\n提取特征...")
        # 为了避免重复实例化，我们先创建一个实例
        classifier_helper = PaperMLPClassifier()

        # 1. 训练集特征 (来自 2022-2024)
        X_train, y_train, _ = classifier_helper.extract_features_and_labels(train_data)
        # 2. 验证集特征 (来自 2025 的随机1000条)
        X_val, y_val, _ = classifier_helper.extract_features_and_labels(val_data)
        # 3. 测试集特征 (来自 2025 的剩余部分)
        X_test, y_test, test_data_valid = classifier_helper.extract_features_and_labels(test_data)

        print(f"\n数据集划分完成:")
        print(f"训练集样本数: {len(y_train)} (来自 2022-2024)")
        print(f"验证集样本数: {len(y_val)} (来自 2025, 用于调参参考)")
        print(f"测试集样本数: {len(y_test)} (来自 2025, 用于最终结果)")
        print(f"训练集特征形状: {X_train.shape}, 测试集特征形状: {X_test.shape}")

        # 2. 训练模型
        if len(y_train) == 0:
            print("\n错误：训练集为空，无法进行模型训练。")
            return

        classifier = PaperMLPClassifier()
        classifier.train(X_train, y_train)

        # 3. 评估模型
        # 在独立的测试集上评估最终性能
        if len(y_test) > 0:
            classifier.evaluate(X_test, y_test, "测试集")

        # 4. 生成并保存详细的预测结果
        if len(y_test) > 0:
            detailed_jsonl = classifier.create_detailed_predictions_jsonl(test_data_valid, X_test)
            if detailed_jsonl:
                output_filename = '../../../Datasets/predict_result_new/rating/MLP.jsonl'

                output_dir = os.path.dirname(output_filename)
                os.makedirs(output_dir, exist_ok=True)

                with open(output_filename, 'w', encoding='utf-8') as f:
                    for entry in detailed_jsonl:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

                print(f"\n详细数据集已成功保存到: {output_filename}")

    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件。 {e}")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()