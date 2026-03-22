import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import warnings
from typing import List, Dict, Tuple
import os

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab_to_idx, max_length=4000):
        self.texts = texts
        self.labels = labels
        self.vocab_to_idx = vocab_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] if self.labels else -1

        tokens = text.lower().split()
        indices = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]

        if len(indices) < self.max_length:
            indices.extend([self.vocab_to_idx['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, num_filters=150, filter_sizes=[3, 4, 5], num_classes=2,
                 dropout=0.56):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        conv_outputs = [torch.max_pool1d(torch.relu(conv(x)), conv(x).size(2)).squeeze(2) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_data(file_path: str) -> List[Dict]:
    """加载JSONL数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_earliest_reviewer_content(reviews: List[Dict]) -> str:
    earliest_reviews_content = []
    if not reviews:
        return ""
    for review in reviews:
        reviewer_id = review.get('reviewer', '')
        if any(role in reviewer_id for role in ['Program_Chair', 'Area_Chair', 'Author']):
            continue

        dialogue_items = [
            item for item in review.get('dialogue', [])
            if 'time' in item and 'content' in item and item['content']
        ]

        if dialogue_items:
            earliest_item = min(dialogue_items, key=lambda d: d['time'])
            earliest_reviews_content.append(earliest_item['content'])

    return ' '.join(earliest_reviews_content)


def preprocess_text(text: str) -> str:
    """文本预处理"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def build_vocab(texts: List[str], min_freq=2) -> Dict[str, int]:
    """构建词汇表"""
    word_counts = Counter(word for text in texts for word in text.split())
    # <PAD> 用于填充，<UNK> 用于未知词
    vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.items() if count >= min_freq]
    return {word: idx for idx, word in enumerate(vocab)}


def prepare_data(data: List[Dict]) -> Tuple[List[str], List[int], List[Dict]]:
    texts = []
    labels = []
    valid_papers = []

    for item in data:
        dialogue_text = extract_earliest_reviewer_content(item.get('reviews', []))
        processed_text = preprocess_text(dialogue_text)

        if processed_text:
            decision = item.get('paper_decision', 'Reject').lower()
            if 'accept' in decision:
                labels.append(1)
            else:
                labels.append(0)

            texts.append(processed_text)
            valid_papers.append(item)

    return texts, labels, valid_papers


def train_model(model, train_loader, val_loader, model_path, num_epochs=8, lr=0.0001,):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0

    print(f"Start training on {device}...")
    for epoch in range(num_epochs):
        model.train()
        train_correct, train_total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_correct += output.argmax(dim=1).eq(target).sum().item()
            train_total += target.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_correct += output.argmax(dim=1).eq(target).sum().item()
                val_total += target.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f'Epoch {epoch + 1}/{num_epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

    return best_val_acc


def evaluate_model(model, test_loader, model_path):
    """评估模型，只返回准确率"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    return accuracy_score(true_labels, predictions)


def create_detailed_predictions_jsonl(
        model: TextCNN,
        vocab_to_idx: Dict[str, int],
        valid_papers: List[Dict],
        X_raw_texts: List[str],
        model_path: str,
        max_length: int,
        batch_size: int
) -> List[Dict]:
    """为有效论文列表创建包含预测结果的详细数据集"""
    print("\n生成详细的预测数据集 (JSONL 格式)...")
    if not valid_papers:
        return []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pred_dataset = ReviewDataset(X_raw_texts, labels=[], vocab_to_idx=vocab_to_idx, max_length=max_length)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data, _ in pred_loader:
            data = data.to(device)
            outputs = model(data)
            pred = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)

            all_predictions.extend(pred.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    decision_map = {0: 'Reject', 1: 'Accept'}

    jsonl_data = []
    for i, paper in enumerate(valid_papers):
        pred_label = all_predictions[i]
        confidence = float(all_probabilities[i][pred_label])

        paper_data = {
            'paper_title': paper.get('paper_title', 'N/A'),
            'real_decision': paper.get('paper_decision', 'N/A'),
            'predict_decision': decision_map.get(pred_label, 'Unknown'),
            'predict_confidence': confidence,
            'reviews': [],
            'method': 'CNN_initial_review'
        }
        for review in paper.get('reviews', []):
            rating_val = review.get('rating', '').split(':')[0].strip()
            confidence_val = review.get('confidence', '').split(':')[0].strip()

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

    output_filename = '../../../Datasets/predict_result_new/initial review/CNN.jsonl'

    model_path = 'TextCNN_initial_reviewer_best_model.pth'

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # 1. 加载数据
    train_data = []
    try:
        for tf in train_files:
            print(f"正在从 {tf} 加载训练数据...")
            train_data.extend(load_data(tf))
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}")
        return

    try:
        print(f"正在从 {test_val_file} 加载验证/测试数据...")
        file_2_data = load_data(test_val_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {test_val_file}")
        return

    # 2. 划分数据集
    np.random.seed(42)
    indices = np.random.permutation(len(file_2_data))

    split_point = 1000
    if len(file_2_data) < 2000:
        split_point = len(file_2_data) // 5

    val_indices = indices[:split_point]
    test_indices = indices[split_point:]

    val_data = [file_2_data[i] for i in val_indices]
    test_data = [file_2_data[i] for i in test_indices]

    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

    # 3. 提取特征
    print("Preparing data (Extracting earliest reviews)...")
    train_texts, train_labels, _ = prepare_data(train_data)
    val_texts, val_labels, _ = prepare_data(val_data)
    test_texts, test_labels, test_data_valid = prepare_data(test_data)

    if not train_texts:
        print("No valid data found.")
        return

    # 4. 构建词表 (关键修改：只用训练集)
    print("Building vocabulary (using TRAIN data only)...")
    # --- 修改点：这里只传入了 train_texts，防止数据泄露 ---
    vocab_to_idx = build_vocab(train_texts, min_freq=2)
    print(f"Vocabulary size: {len(vocab_to_idx)}")

    max_length = 4000
    batch_size = 16

    print(f"Max Sequence Length set to: {max_length}")

    # 6. DataLoader
    # 注意：val 和 test 中出现的生词会被自动转为 <UNK>，这是符合逻辑的
    train_loader = DataLoader(ReviewDataset(train_texts, train_labels, vocab_to_idx, max_length), batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(ReviewDataset(val_texts, val_labels, vocab_to_idx, max_length), batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(ReviewDataset(test_texts, test_labels, vocab_to_idx, max_length), batch_size=batch_size,
                             shuffle=False)

    # 7. 训练与评估
    print("Creating and training model...")
    model = TextCNN(vocab_size=len(vocab_to_idx), num_classes=2)

    best_val_acc = train_model(model, train_loader, val_loader, model_path, num_epochs=8, lr=0.001)

    print("\nEvaluating model on test set...")
    test_accuracy = evaluate_model(model, test_loader, model_path)

    print(f"\n--- Final Results ---\nBest Validation Accuracy: {best_val_acc:.4f}\nTest Accuracy: {test_accuracy:.4f}")

    if test_data_valid:
        detailed_jsonl = create_detailed_predictions_jsonl(
            model, vocab_to_idx, test_data_valid, test_texts, model_path, max_length, batch_size
        )
        if detailed_jsonl:
            with open(output_filename, 'w', encoding='utf-8') as f:
                for entry in detailed_jsonl:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"\n详细数据集已成功保存到: {output_filename}")


if __name__ == "__main__":
    main()