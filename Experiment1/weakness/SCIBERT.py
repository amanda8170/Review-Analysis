import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
import os
from datetime import datetime
from typing import List, Dict, Tuple
import optuna
from tqdm.auto import tqdm
import gc
from torch.optim import AdamW
import re

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def load_jsonl_data(file_path: str) -> List[Dict]:
    """从.jsonl文件中加载数据"""
    print(f"正在从 {file_path} 加载数据...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_weakness_only(text: str) -> str:
    """【核心提取逻辑】只从审稿意见文本中提取 Weakness 部分。"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    pattern = re.compile(
        r'(?i)(?:Weaknesses?|Weakness)\s*:\s*(.*?)(?=\s*(?:Summary|Strengths?|Questions|Soundness|Presentation|Contribution|Rating|Confidence|Correctness)\s*:|$)',
        re.DOTALL
    )
    matches = pattern.findall(text)
    extracted_parts = [" ".join(content.split()) for content in matches if content.strip()]
    return "\n".join(extracted_parts) if extracted_parts else ""


class PaperReviewDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length=512):
        self.texts = []
        self.labels = []
        self.original_data = []
        for item in tqdm(data, desc="处理数据中 (提取 Weakness)"):
            weakness_text = self._extract_all_reviews_weaknesses(item.get('reviews', []))
            if not weakness_text:
                continue
            label = 1 if 'accept' in item.get('paper_decision', '').lower() else 0
            self.texts.append(weakness_text)
            self.labels.append(label)
            self.original_data.append(item)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _extract_all_reviews_weaknesses(self, reviews: List[Dict]) -> str:
        """从一篇论文的所有审稿意见中，提取每个审稿人首次提交意见里的 Weaknesses 部分。"""
        if not reviews:
            return ""
        all_weaknesses = []
        for review in reviews:
            reviewer_id = review.get('reviewer', 'Unknown')
            if any(role in reviewer_id for role in ['Program_Chair', 'Area_Chair', 'Author']):
                continue
            dialogue_items = [
                item for item in review.get('dialogue', [])
                if isinstance(item, dict) and 'time' in item and 'content' in item and item['content']
            ]
            if dialogue_items:
                try:
                    first_item = min(dialogue_items, key=lambda d: datetime.strptime(d['time'], '%Y-%m-%d %H:%M:%S'))
                    weakness_text = extract_weakness_only(first_item['content'])
                    if weakness_text:
                        all_weaknesses.append(weakness_text)
                except (ValueError, TypeError):
                    continue
        return ' [SEP] '.join(all_weaknesses)

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


# --- 2. 训练与评估循环 (无变化) ---
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
        for batch in tqdm(train_loader, desc=f'Trial {trial.number} Epoch {epoch + 1}', leave=False):
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


# --- 3. Optuna 目标函数 (无变化) ---
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
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    DATA_DIR = '../../../Datasets/formatted_dataset/'
    OUTPUT_DIR = '../../../Datasets/predict_result_new/weakness'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- MODIFICATION START ---
    # 定义所有数据文件的路径
    train_files = [
        os.path.join(DATA_DIR, 'ICLR.cc_2022_formatted.jsonl'),
        os.path.join(DATA_DIR, 'ICLR.cc_2023_formatted.jsonl'),
        os.path.join(DATA_DIR, 'ICLR.cc_2024_formatted.jsonl')
    ]
    val_test_file = os.path.join(DATA_DIR, 'ICLR.cc_2025_formatted.jsonl')

    # 加载 2022-2024 数据作为训练集
    print("加载训练数据 (2022-2024)...")
    train_data = []
    for file_path in train_files:
        if os.path.exists(file_path):
            train_data.extend(load_jsonl_data(file_path))
        else:
            print(f"警告: 训练文件 {file_path} 不存在，已跳过。")

    # 加载 2025 数据并划分为验证集和测试集
    print("加载并划分验证/测试数据 (2025)...")
    val_data, test_data = [], []
    if os.path.exists(val_test_file):
        data_2025 = load_jsonl_data(val_test_file)

        # 过滤掉没有review的数据，以防划分出错
        data_2025_filtered = [d for d in data_2025 if d.get('reviews')]

        # 检查数据量是否足够划分
        if len(data_2025_filtered) > 1000:
            # 使用 train_test_split 进行随机划分
            # 将 1000 条数据分给 val_data，其余的给 test_data
            test_data, val_data = train_test_split(
                data_2025_filtered,
                test_size=1000,  # 指定验证集的大小为 1000
                random_state=42,  # 确保每次划分结果一致
                shuffle=True  # 随机打乱数据
            )
        else:
            print("警告: 2025年数据不足1000条，无法按要求划分。请检查数据。")
            # 可以根据需要设置备用逻辑，例如使用更小的验证集
            val_data = data_2025_filtered
            test_data = []

    else:
        print(f"警告: 验证/测试文件 {val_test_file} 不存在。")

    print(f"\n数据划分完成:")
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"测试集样本数: {len(test_data)}")

    model_name = "allenai/scibert_scivocab_uncased"
    print(f"\n正在使用模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    initial_params = {
        "learning_rate": 1.072096096148636e-05,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "weight_decay": 0.010783161637122668
    }

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.enqueue_trial(initial_params)

    print("\n开始超参数搜索...")
    study.optimize(lambda trial: objective(trial, train_data, val_data, tokenizer, model_name, device), n_trials=10)

    best_params = study.best_params
    print("\n--- 搜索完成 ---")
    print(f"最佳试验结果 (验证集准确率): {study.best_value:.4f}")
    print(f"最佳超参数: {best_params}")

    print("\n使用最佳参数在 '训练集(22-24)+验证集(25部分)' 上重新训练最终模型...")
    # 注意：这里我们将 Optuna 使用的训练集和验证集合并，以利用所有标注数据进行最终训练
    final_train_data = train_data + val_data
    final_train_dataset = PaperReviewDataset(final_train_data, tokenizer)
    test_dataset = PaperReviewDataset(test_data, tokenizer)

    final_train_loader = DataLoader(final_train_dataset, batch_size=best_params['per_device_train_batch_size'],
                                    shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    final_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = AdamW(final_model.parameters(), lr=best_params['learning_rate'],
                      weight_decay=best_params['weight_decay'])
    total_steps = len(final_train_loader) * best_params['num_train_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(best_params['num_train_epochs']):
        final_model.train()
        for batch in tqdm(final_train_loader,
                          desc=f"Final Training Epoch {epoch + 1}/{best_params['num_train_epochs']}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = final_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

    model_save_path = os.path.join(OUTPUT_DIR, 'best_model_weakness_only')
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

    print("\n--- 最终测试集评估结果 ---")
    # 检查测试集是否为空
    if all_labels:
        print(classification_report(all_labels, all_preds, target_names=['Reject/Withdrawn', 'Accept']))
    else:
        print("测试集为空，无法生成评估报告。")

    decision_map = {0: 'Reject/Withdrawn', 1: 'Accept'}
    output_filename = os.path.join(OUTPUT_DIR, 'SCIBERT.jsonl')
    with open(output_filename, 'w', encoding='utf-8') as f:
        for i in range(len(test_dataset.original_data)):
            original_paper = test_dataset.original_data[i]
            paper_data = {
                'paper_title': original_paper.get('paper_title', 'N/A'),
                'real_decision': original_paper.get('paper_decision', 'N/A'),
                'predict_decision': decision_map.get(all_preds[i].item(), 'Unknown'),
                'predict_confidence': all_confidences[i].item(),
                'method':'SCIBERT_weakness'
            }
            f.write(json.dumps(paper_data, ensure_ascii=False) + '\n')

    print(f"详细预测结果已保存到: {output_filename}")


if __name__ == "__main__":
    main()