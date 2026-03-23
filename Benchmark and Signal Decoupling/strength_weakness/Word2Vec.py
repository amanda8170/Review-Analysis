import json
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import re
import warnings
from typing import List, Dict, Tuple
import os

# --- 基础设置 ---
warnings.filterwarnings('ignore')
np.random.seed(42)


# --- 数据处理和特征提取函数 ---
def load_data(file_path: str) -> List[Dict]:
    """加载单文件数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def preprocess_text(text: str) -> List[str]:
    if not text:
        return []
    # 去除标点和数字，转小写
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return simple_preprocess(text)


def extract_strengths_and_weaknesses(text: str) -> str:
    """
    【提取逻辑】只提取 Strengths 和 Weaknesses。
    """
    if not text:
        return ""

    # 1. 基础去HTML
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. 正则表达式
    # 匹配 Strengths 或 Weaknesses 开头，直到遇到 Summary, Questions 等关键词或结尾
    pattern = re.compile(
        r'(Strengths?|Weaknesses?)\s*:\s*(.*?)(?=\s*(?:Summary|Strengths?|Weaknesses?|Questions|Soundness|Presentation|Contribution|Rating|Confidence|Correctness)\s*:|$)',
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.findall(text)

    extracted_parts = []
    for header, content in matches:
        clean_content = " ".join(content.split())
        if clean_content:
            header_normalized = header.title()
            # 保留标题以便阅读，Word2Vec 预处理时会把方括号去掉，只保留单词语义
            extracted_parts.append(f"[{header_normalized}]: {clean_content}")

    if extracted_parts:
        return "\n".join(extracted_parts)
    else:
        return ""


# --- 新增函数 2: 遍历所有 Reviewer 并提取 S&W ---
def extract_all_reviews_sw(reviews: List[Dict]) -> str:
    """
    提取所有审稿人的 Strengths 和 Weaknesses。
    已移除 'Opinion from...' 标记。
    """
    if not reviews:
        return ""

    all_content = []

    for review in reviews:
        # 1. 筛选角色
        reviewer_id = review.get('reviewer', 'Unknown')
        if any(role in reviewer_id for role in ['Program_Chair', 'Area_Chair', 'Author']):
            continue

        # 2. 获取 Dialogue
        dialogue_list = review.get('dialogue', [])
        valid_items = [
            item for item in dialogue_list
            if isinstance(item, dict) and 'time' in item and 'content' in item and item['content']
        ]

        if valid_items:
            # 3. 找最早发言
            first_item = min(valid_items, key=lambda d: d['time'])

            # 4. 【提取 S & W】
            sw_text = extract_strengths_and_weaknesses(first_item['content'])

            if sw_text:
                all_content.append(sw_text)

    return "\n\n".join(all_content)


def prepare_features(data: List[Dict]) -> Tuple[List[str], List[int], List[Dict]]:
    labels, texts, valid_papers = [], [], []
    for item in data:
        # --- 修改点：调用新的 S&W 提取函数 ---
        dialogue_text = extract_all_reviews_sw(item.get('reviews', []))

        # 只有提取到了内容才纳入训练/测试
        if dialogue_text:
            decision = item.get('paper_decision', '').lower()
            if 'accept' in decision:
                label = 1
            elif 'reject' in decision or 'withdrawn' in decision:
                label = 0
            else:
                continue

            texts.append(dialogue_text)
            labels.append(label)
            valid_papers.append(item)
    return texts, labels, valid_papers


# --- 预测器类 (逻辑保持不变) ---
class PaperAcceptancePredictor:
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=10,
                 hidden_layer_sizes=(128, 64), alpha=0.001, learning_rate_init=0.001):
        # Word2Vec parameters
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs

        # MLP parameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init

        # Models and scaler
        self.word2vec_model = None
        self.mlp_model = None
        self.scaler = StandardScaler()

    def train_word2vec(self, texts: List[str]):
        processed_texts = [preprocess_text(text) for text in texts]
        self.word2vec_model = Word2Vec(
            sentences=processed_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=1
        )
        print(f"Word2Vec模型训练完成，词汇量: {len(self.word2vec_model.wv)}")

    def text_to_vector(self, text: str) -> np.ndarray:
        words = preprocess_text(text)
        if not words: return np.zeros(self.vector_size)
        word_vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(self.vector_size)

    def train(self, train_texts: List[str], train_labels: List[int]):
        print(f"开始训练流程... 样本数量: {len(train_texts)}")

        print("1. 训练Word2Vec模型...")
        self.train_word2vec(train_texts)

        print("2. 转换文本为向量...")
        train_vectors = np.array([self.text_to_vector(text) for text in train_texts])

        print("3. 数据标准化...")
        train_vectors_scaled = self.scaler.fit_transform(train_vectors)

        print("4. 训练MLP模型...")
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            activation='relu',
            solver='adam',
            batch_size=32,
            max_iter=200,
            random_state=42
        )
        self.mlp_model.fit(train_vectors_scaled, train_labels)

        train_pred = self.mlp_model.predict(train_vectors_scaled)
        train_accuracy = accuracy_score(train_labels, train_pred)
        print(f"训练集准确率: {train_accuracy:.4f}")

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.word2vec_model or not self.mlp_model:
            raise ValueError("模型未训练，请先调用train方法")
        vectors = np.array([self.text_to_vector(text) for text in texts])
        vectors_scaled = self.scaler.transform(vectors)
        predictions = self.mlp_model.predict(vectors_scaled)
        probabilities = self.mlp_model.predict_proba(vectors_scaled)
        return predictions, probabilities

    def create_detailed_predictions_jsonl(self, valid_papers: List[Dict], texts: List[str]) -> List[Dict]:
        print("\n生成详细的预测数据集 (JSONL 格式)...")
        if not valid_papers: return []

        predictions, probabilities = self.predict(texts)
        confidences = np.max(probabilities, axis=1)
        decision_map = {0: 'Reject', 1: 'Accept'}

        jsonl_data = []
        for i, paper in enumerate(valid_papers):
            paper_data = {
                'paper_title': paper.get('paper_title', 'N/A'),
                'real_decision': paper.get('paper_decision', 'N/A'),
                'predict_decision': decision_map.get(predictions[i].item(), 'Unknown'),
                'predict_confidence': float(confidences[i]),
                'reviews': [],
                'method': 'word2vec_strengths_weaknesses'
            }
            for review in paper.get('reviews', []):
                simple_review = {
                    "reviewer": review.get("reviewer", "N/A"),
                    "rating": review.get("rating", "-1").split(':')[0].strip(),
                    "confidence": review.get("confidence", "-1").split(':')[0].strip()
                }
                paper_data['reviews'].append(simple_review)
            jsonl_data.append(paper_data)

        print(f"成功为 {len(jsonl_data)} 篇论文生成了详细记录。")
        return jsonl_data


def main():
    """主函数"""
    train_files = [
        '../../../Datasets/formatted_dataset/ICLR.cc_2022_formatted.jsonl',
        '../../../Datasets/formatted_dataset/ICLR.cc_2023_formatted.jsonl',
        '../../../Datasets/formatted_dataset/ICLR.cc_2024_formatted.jsonl'
    ]
    test_val_file = '../../../Datasets/formatted_dataset/ICLR.cc_2025_formatted.jsonl'

    # 1. 统一加载所有训练数据
    train_data_all = []
    try:
        for tf in train_files:
            print(f"正在从 {tf} 加载训练数据...")
            train_data_all.extend(load_data(tf))
        print(f"训练数据加载完毕，共 {len(train_data_all)} 条。")
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}")
        return

    # 2. 提前提取特征 (Strengths & Weaknesses)
    print("正在提取训练集特征 (Strengths & Weaknesses)...")
    train_texts, train_labels, _ = prepare_features(train_data_all)

    if not train_texts:
        print("错误：未从训练集中提取到任何有效文本，请检查数据格式。")
        return

    # 3. 配置参数
    best_params = {
        'vector_size': 200,
        'window': 6,
        'min_count': 4,
        'epochs': 13,
        'hidden_layer_sizes': (128,),
        'alpha': 0.0051796654578141485,
        'learning_rate_init': 0.00011920655340939226
    }
    print("使用参数:", best_params)

    # 4. 初始化模型
    predictor = PaperAcceptancePredictor(
        vector_size=best_params['vector_size'],
        window=best_params['window'],
        min_count=best_params['min_count'],
        epochs=best_params['epochs'],
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init']
    )

    # 训练
    predictor.train(train_texts, train_labels)

    # 5. 加载测试集
    try:
        print(f"\n正在从 {test_val_file} 加载验证/测试数据...")
        file_2_data = load_data(test_val_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {test_val_file}")
        return

    # 6. 分割验证集和测试集
    indices = np.random.permutation(len(file_2_data))
    val_indices, test_indices = indices[:1000], indices[1000:]
    val_data = [file_2_data[i] for i in val_indices]
    test_data = [file_2_data[i] for i in test_indices]

    print("\n=== 验证集评估 ===")
    val_texts, val_labels, _ = prepare_features(val_data)
    if val_texts:
        val_pred, _ = predictor.predict(val_texts)
        print(f"验证集准确率: {accuracy_score(val_labels, val_pred):.4f}")

    print("\n=== 测试集评估 ===")
    test_texts, test_labels, test_data_valid = prepare_features(test_data)
    if test_texts:
        test_pred, _ = predictor.predict(test_texts)
        print(f"测试集准确率: {accuracy_score(test_labels, test_pred):.4f}")

    # 7. 保存结果
    if test_data_valid:
        # --- 修改点：输出文件名 ---
        output_filename = '../../../Datasets/predict_result_new/strength_weakness/word2vec.jsonl'

        detailed_jsonl = predictor.create_detailed_predictions_jsonl(test_data_valid, test_texts)

        with open(output_filename, 'w', encoding='utf-8') as f:
            for entry in detailed_jsonl:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"\n详细数据集已成功保存到: {output_filename}")


if __name__ == "__main__":
    main()