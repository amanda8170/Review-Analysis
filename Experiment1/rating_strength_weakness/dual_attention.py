import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import optuna
import logging
import gc
from datetime import datetime
import re
from typing import List, Dict
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(42)
np.random.seed(42)


def load_jsonl(file_path):
    logger.info(f"Loading data from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return data

### MODIFICATION START: 替换为新的文本提取函数 ###
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
### MODIFICATION END ###


class PaperDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paper = self.data[idx]

        ### MODIFICATION: 调用新的函数提取 Strengths 和 Weaknesses ###
        combined_review_text = extract_all_reviews_sw(paper.get('reviews', []))

        if not combined_review_text.strip() and 'paper_abstract' in paper:
            combined_review_text = paper.get('paper_abstract', '')

        ratings = []
        for review in paper.get('reviews', []):
            try:
                rating_str = review.get('rating', '-1').split(':')[0].strip()
                if rating_str and rating_str != '-1':
                    ratings.append(float(rating_str))
            except (ValueError, TypeError):
                continue

        if ratings:
            rating_features = [np.mean(ratings), np.std(ratings) if len(ratings) > 1 else 0, max(ratings), min(ratings),
                               len(ratings)]
        else:
            rating_features = [0, 0, 0, 0, 0]

        decision = paper.get('paper_decision', '')
        label = 1 if 'Accept' in decision else 0


        encoding = self.tokenizer(combined_review_text, truncation=True, padding='longest', return_tensors='pt')

        simple_reviews = []
        for review in paper.get('reviews', []):
            simple_reviews.append({
                'reviewer': review.get('reviewer', 'N/A'),
                'rating': review.get('rating', '-1').split(':')[0].strip(),
                'confidence': review.get('confidence', '-1').split(':')[0].strip()
            })

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rating_features': torch.tensor(rating_features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'original_data': {'paper_title': paper.get('paper_title', 'N/A'), 'real_decision': decision,
                              'reviews': simple_reviews}
        }


def custom_collate_fn(batch):
    model_inputs = [{k: v for k, v in item.items() if k != 'original_data'} for item in batch]
    original_data = [item['original_data'] for item in batch]

    keys = model_inputs[0].keys()
    collated_model_inputs = {}
    for key in keys:
        if key in ['input_ids', 'attention_mask']:
            # 找到这个batch中最大的长度
            max_len = max(item[key].shape[0] for item in model_inputs)
            # 手动进行padding
            collated_model_inputs[key] = torch.stack([
                torch.cat([item[key], torch.zeros(max_len - item[key].shape[0], dtype=item[key].dtype)])
                for item in model_inputs
            ])
        else:
            # 对于其他已经是tensor的特征，直接堆叠
            collated_model_inputs[key] = torch.stack([item[key] for item in model_inputs])

    collated_original_data = {}
    if original_data:
        for key in original_data[0].keys():
            collated_original_data[key] = [d[key] for d in original_data]
    collated_model_inputs['original_data'] = collated_original_data
    return collated_model_inputs


# --- 3. Model Architecture (No Changes) ---
class DualBranchAttentionModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_rating_features=5, hidden_dim=768, num_classes=2,
                 dropout_rate=0.3):
        super(DualBranchAttentionModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.text_hidden_dim = self.text_encoder.config.hidden_size
        self.rating_branch = nn.Sequential(
            nn.Linear(num_rating_features, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, hidden_dim)
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.text_projection = nn.Linear(self.text_hidden_dim, hidden_dim)
        self.fusion_layer = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask, rating_features):
        batch_size = input_ids.size(0)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = text_outputs.pooler_output
        text_projected = self.text_projection(text_pooled)
        rating_features_processed = self.rating_branch(rating_features)
        combined_features = torch.stack([text_projected, rating_features_processed], dim=1)
        attended_features, _ = self.attention(combined_features, combined_features, combined_features)
        attended_flat = attended_features.reshape(batch_size, -1)
        fused_features = self.fusion_layer(attended_flat)
        logits = self.classifier(fused_features)
        return logits, None


# --- 4. Training and Evaluation Functions (No changes, but custom_collate_fn is now more important) ---
def evaluate(model, data_loader, device, is_validation=False):
    model.eval()
    all_preds, all_labels = [], []
    output_records = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", disable=is_validation):
            input_ids, attention_mask, rating_features, labels = batch['input_ids'].to(device), batch[
                'attention_mask'].to(device), batch['rating_features'].to(device), batch['label'].to(device)
            logits, _ = model(input_ids, attention_mask, rating_features)
            probabilities = torch.softmax(logits, dim=1)
            confidences, preds = torch.max(probabilities, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if not is_validation:
                original_data = batch['original_data']
                for i in range(len(labels)):
                    pred_str = "Accept" if preds[i].item() == 1 else "Reject"
                    output_records.append({
                        "paper_title": original_data['paper_title'][i],
                        "real_decision": original_data['real_decision'][i],
                        "predict_decision": pred_str,
                        "predict_confidence": confidences[i].item(),
                        "reviews": original_data['reviews'][i],
                        'method':'dual_attention'
                    })
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, output_records


def train_for_trial(model, train_loader, val_loader, device, params):
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(params['num_epochs']):
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{params["num_epochs"]}', leave=False):
            input_ids, attention_mask, rating_features, labels = batch['input_ids'].to(device), batch[
                'attention_mask'].to(device), batch['rating_features'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask, rating_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_acc, _ = evaluate(model, val_loader, device, is_validation=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= params['early_stopping_patience']:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    return best_val_acc


def objective(trial, train_data, val_data, tokenizer, device):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        'num_epochs': trial.suggest_int('num_epochs', 3, 10),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'early_stopping_patience': trial.suggest_int('early_stopping_patience', 2, 3)
    }

    train_loader = DataLoader(PaperDataset(train_data, tokenizer), batch_size=params['batch_size'], shuffle=True,
                              collate_fn=custom_collate_fn, num_workers=0)
    val_loader = DataLoader(PaperDataset(val_data, tokenizer), batch_size=params['batch_size'], shuffle=False,
                            collate_fn=custom_collate_fn, num_workers=0)
    model = DualBranchAttentionModel(dropout_rate=params['dropout_rate']).to(device)

    val_accuracy = train_for_trial(model, train_loader, val_loader, device, params)

    gc.collect()
    torch.cuda.empty_cache()

    return val_accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(device)}")

    DATA_DIR = '../../../Datasets/formatted_dataset/'
    OUTPUT_DIR = '../../../Datasets/predict_result_new/rating_strength_weakness/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    known_best_params = {
        'learning_rate': 6.389750914285708e-06,
        'dropout_rate': 0.22053146807269458,
        'batch_size': 16,
        'num_epochs': 6,
        'weight_decay': 0.0026526308330624914,
        'early_stopping_patience': 2
    }

    ### MODIFICATION: 更改数据集加载和划分逻辑 ###
    print("Loading datasets...")
    # 加载 2022, 2023, 2024 数据作为训练集
    data_2022 = load_jsonl(os.path.join(DATA_DIR, 'ICLR.cc_2022_formatted.jsonl'))
    data_2023 = load_jsonl(os.path.join(DATA_DIR, 'ICLR.cc_2023_formatted.jsonl'))
    data_2024 = load_jsonl(os.path.join(DATA_DIR, 'ICLR.cc_2024_formatted.jsonl'))
    train_data = data_2022 + data_2023 + data_2024

    # 加载 2025 数据，并划分为验证集和测试集
    data_2025 = load_jsonl(os.path.join(DATA_DIR, 'ICLR.cc_2025_formatted.jsonl'))
    random.shuffle(data_2025)  # 随机打乱

    val_size = 1000
    if len(data_2025) > val_size:
        val_data = data_2025[:val_size]
        test_data = data_2025[val_size:]
    else:  # 如果2025数据不足1000，全部作为验证集
        val_data = data_2025
        test_data = []
        print("Warning: 2025 data has less than 1000 samples, all will be used for validation.")

    print(f"Dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # --- Optuna Hyperparameter Search ---
    print("\nStarting Optuna hyperparameter search...")
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

    # Enqueue the known trial first (Warm Start)
    study.enqueue_trial(known_best_params)

    study.optimize(
        lambda trial: objective(trial, train_data, val_data, tokenizer, device),
        n_trials=20  # Total trials, including the one we enqueued
    )

    best_params = study.best_params
    print("\n--- Search Finished ---")
    print(f"Best trial validation accuracy: {study.best_value:.4f}")
    print(f"Best hyperparameters found: {best_params}")

    # --- Final Training using Best Hyperparameters ---
    print("\nTraining final model with best hyperparameters...")
    # 使用 2022-2024 的全部数据进行最终训练
    final_train_loader = DataLoader(PaperDataset(train_data, tokenizer), batch_size=best_params['batch_size'],
                                    shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
    val_loader_for_final_train = DataLoader(PaperDataset(val_data, tokenizer), batch_size=best_params['batch_size'],
                                            shuffle=False,
                                            collate_fn=custom_collate_fn, num_workers=0)
    test_loader = DataLoader(PaperDataset(test_data, tokenizer), batch_size=best_params['batch_size'], shuffle=False,
                             collate_fn=custom_collate_fn, num_workers=0)

    final_model = DualBranchAttentionModel(dropout_rate=best_params['dropout_rate']).to(device)
    # 使用找到的最佳参数，在完整的训练集上重新训练模型
    train_for_trial(final_model, final_train_loader, val_loader_for_final_train, device, best_params)

    # --- Final Evaluation and Saving ---
    print("\nEvaluating final model on the test set...")
    if test_data:
        test_accuracy, output_records = evaluate(final_model, test_loader, device)
        print(f"\n--- FINAL TEST ACCURACY: {test_accuracy:.4f} ---")

        results_file_path = os.path.join(OUTPUT_DIR, 'dual_attention.jsonl')
        with open(results_file_path, 'w', encoding='utf-8') as f_out:
            for record in output_records:
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Prediction results saved to: {results_file_path}")
    else:
        test_accuracy = -1.0
        print("No test data to evaluate.")

    model_file_path = os.path.join(OUTPUT_DIR, 'dual_branch_attention_model.pth')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'best_params': best_params,
        'test_accuracy': test_accuracy
    }, model_file_path)
    print(f"Model and parameters saved to: {model_file_path}")


if __name__ == "__main__":
    main()