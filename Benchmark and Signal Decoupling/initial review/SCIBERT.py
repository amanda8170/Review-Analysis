import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
import os
from datetime import datetime
from typing import List, Dict
import optuna
from tqdm.auto import tqdm
import gc
from torch.optim import AdamW

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def load_jsonl_data(file_path: str) -> List[Dict]:
    """从单个 .jsonl 文件加载数据"""
    print(f"正在从 {file_path} 加载数据...")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}。请检查路径是否正确。")
        raise
    return data


class PaperReviewDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length=512):
        self.texts = []
        self.labels = []
        self.original_data = []
        for item in tqdm(data, desc="处理数据中"):
            dialogue_text = self._extract_earliest_reviewer_content(item)
            if not dialogue_text:
                continue
            label = 1 if 'accept' in item.get('paper_decision', '').lower() else 0
            self.texts.append(dialogue_text)
            self.labels.append(label)
            self.original_data.append(item)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _extract_earliest_reviewer_content(self, paper_data: Dict) -> str:
        """
        提取每位评审员（非主席或作者）的最早一条评论内容。
        """
        earliest_reviews_content = []
        reviews = paper_data.get('reviews', [])
        if not reviews:
            return ""
        for review in reviews:
            reviewer_id = review.get('reviewer', '')
            if any(role in reviewer_id for role in ['Program_Chair', 'Area_Chair', 'Author']):
                continue
            dialogue_items = [
                item for item in review.get('dialogue', [])
                if 'time' in item and 'content' in item and isinstance(item['content'], str) and item['content'].strip()
            ]
            if dialogue_items:
                try:
                    earliest_item = min(dialogue_items, key=lambda d: datetime.strptime(d['time'], '%Y-%m-%d %H:%M:%S'))
                    earliest_reviews_content.append(earliest_item['content'])
                except (ValueError, TypeError):
                    continue
        return ' '.join(earliest_reviews_content)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)


def train_for_trial(trial, model, train_loader, val_loader, device, params):
    optimizer = AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    total_steps = len(train_loader) * params['num_train_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for epoch in range(params['num_train_epochs']):
        model.train()
        for batch in tqdm(train_loader, desc=f'Trial {trial.number} Epoch {epoch + 1}'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        val_accuracy = evaluate(model, val_loader, device)
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return evaluate(model, val_loader, device)


def objective(trial, train_data, val_data, tokenizer, model_name, device):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 4),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    }
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    train_dataset = PaperReviewDataset(train_data, tokenizer)
    val_dataset = PaperReviewDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=params['per_device_train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    accuracy = train_for_trial(trial, model, train_loader, val_loader, device, params)
    gc.collect()
    torch.cuda.empty_cache()
    return accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_files = [
        '../../../Datasets/formatted_dataset/ICLR.cc_2022_formatted.jsonl',
        '../../../Datasets/formatted_dataset/ICLR.cc_2023_formatted.jsonl',
        '../../../Datasets/formatted_dataset/ICLR.cc_2024_formatted.jsonl'
    ]
    # 将2025数据指定为验证和测试的源文件
    test_val_file = '../../../Datasets/formatted_dataset/ICLR.cc_2025_formatted.jsonl'

    OUTPUT_DIR = '../../../Datasets/predict_result_new/initial review'
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在
    output_filename = os.path.join(OUTPUT_DIR, 'Scibert.jsonl')

    # --- 数据加载和划分逻辑已更新 ---
    print("加载训练数据 (2022-2024)...")
    train_data = []
    for tf in train_files:
        train_data.extend(load_jsonl_data(tf))

    print("加载用于验证和测试的数据 (2025)...")
    data_2025 = load_jsonl_data(test_val_file)

    print("正在从2025数据中划分验证集(1000条)和测试集(剩余)...")
    if len(data_2025) < 1000:
        raise ValueError("2025年的数据少于1000条，无法按要求划分验证集。")

    # 使用train_test_split进行划分
    # test_size=1000 指定了第二个列表（val_data）的大小
    # 第一个列表（test_data）将包含剩余的数据
    test_data, val_data = train_test_split(
        data_2025,
        test_size=1000,
        random_state=42,  # 保证每次划分结果一致
        shuffle=True
    )

    print(f"数据加载与划分完毕: 训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")
    # --- 逻辑更新结束 ---

    model_name = "allenai/scibert_scivocab_uncased"
    print(f"正在使用模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    print("\n开始超参数搜索...")
    study.optimize(lambda trial: objective(trial, train_data, val_data, tokenizer, model_name, device), n_trials=15)

    best_params = study.best_params
    print("\n--- 搜索完成 ---")
    print(f"最佳试验准确率: {study.best_value:.4f}")
    print(f"最佳超参数: {best_params}")

    print("\n使用最佳参数在全量训练数据(train+val)上重新训练最终模型...")
    # 合并原始训练集和验证集以进行最终训练
    final_train_data = train_data + val_data
    final_train_dataset = PaperReviewDataset(final_train_data, tokenizer)
    test_dataset = PaperReviewDataset(test_data, tokenizer)

    final_train_loader = DataLoader(final_train_dataset, batch_size=best_params['per_device_train_batch_size'],
                                    shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    final_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = AdamW(final_model.parameters(), lr=best_params['learning_rate'],
                      weight_decay=best_params['weight_decay'])

    for epoch in range(best_params['num_train_epochs']):
        final_model.train()
        progress_bar = tqdm(final_train_loader,
                            desc=f"Final Training Epoch {epoch + 1}/{best_params['num_train_epochs']}")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = final_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})

    model_save_path = os.path.join(OUTPUT_DIR, 'best_model')
    final_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"最终模型已保存到: {model_save_path}")

    print("\n在测试集上评估最终模型并生成详细报告...")
    final_model.eval()
    all_preds, all_labels, all_confidences = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = final_model(input_ids, attention_mask=attention_mask, labels=labels)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidences, preds = torch.max(probabilities, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"最终测试集准确率: {test_accuracy:.4f}")

    decision_map = {0: 'Reject', 1: 'Accept'}
    with open(output_filename, 'w', encoding='utf-8') as f:
        for i in range(len(test_dataset.original_data)):
            original_paper = test_dataset.original_data[i]
            paper_data = {
                'paper_title': original_paper.get('paper_title', 'N/A'),
                'real_decision': original_paper.get('paper_decision', 'N/A'),
                'predict_decision': decision_map.get(all_preds[i].item(), 'Unknown'),
                'predict_confidence': all_confidences[i].item(),
                'reviews': original_paper.get('reviews', []),
                'method':'Scibert_initial review'
            }
            f.write(json.dumps(paper_data, ensure_ascii=False) + '\n')

    print(f"详细预测结果已保存到: {output_filename}")


if __name__ == "__main__":
    main()