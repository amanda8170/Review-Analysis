import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
from typing import List, Dict, Tuple

# Set random seed for reproducibility
np.random.seed(42)


def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_earliest_reviewer_content(paper_data: Dict) -> str:
    """Extracts the chronologically earliest review from each non-chair reviewer."""
    earliest_reviews = []
    if 'reviews' in paper_data:
        for review in paper_data['reviews']:
            reviewer_id = review.get('reviewer', '')
            # Using 'in' is safer for future-proofing
            if 'Area_Chair' in reviewer_id or 'Program_Chair' in reviewer_id or 'Author' in reviewer_id:
                continue

            reviewer_dialogues = []
            if 'dialogue' in review:
                for dialogue_item in review['dialogue']:
                    if 'time' in dialogue_item and 'content' in dialogue_item:
                        content = dialogue_item.get('content')
                        if isinstance(content, str) and content.strip():
                            reviewer_dialogues.append(dialogue_item)

            if reviewer_dialogues:
                earliest_review_item = min(reviewer_dialogues, key=lambda d: d['time'])
                content = earliest_review_item.get('content')
                if isinstance(content, str) and content.strip():
                    earliest_reviews.append(content)
    return ' '.join(earliest_reviews)


def get_paper_decision(paper_data: Dict) -> str:
    """Extract paper decision (accept/reject)"""
    decision = paper_data.get('paper_decision', '').lower()
    if 'accept' in decision:
        return 'accept'
    return 'reject'


def preprocess_text(text: str) -> str:
    """Basic text preprocessing"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_dataset(data: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
    """Prepare dataset and return corresponding valid paper objects."""
    X = []
    y = []
    valid_papers = []  # To store papers that are successfully processed

    for paper in data:
        earliest_review_content = extract_earliest_reviewer_content(paper)
        if not earliest_review_content.strip():
            continue

        decision = get_paper_decision(paper)
        processed_content = preprocess_text(earliest_review_content)

        if processed_content:
            X.append(processed_content)
            y.append(decision)
            valid_papers.append(paper)  # Add the corresponding paper object

    return X, y, valid_papers


def create_detailed_predictions_jsonl(
        model: LogisticRegression,
        vectorizer: TfidfVectorizer,
        valid_papers: List[Dict],
        X_raw_texts: List[str]
) -> List[Dict]:
    """Creates a detailed dataset with predictions in a format suitable for JSONL."""
    print("\n生成详细的预测数据集 (JSONL 格式)...")
    if not valid_papers:
        print("没有有效数据来生成详细数据集。")
        return []

    X_tfidf = vectorizer.transform(X_raw_texts)
    predictions = model.predict(X_tfidf)

    decision_map = {'reject': 'Reject', 'accept': 'Accept'}
    predicted_decisions_str = [decision_map.get(p, 'Unknown') for p in predictions]

    jsonl_data = []
    for i, paper in enumerate(valid_papers):
        paper_data = {
            'paper_title': paper.get('paper_title', 'N/A'),
            'real_decision': paper.get('paper_decision', 'N/A'),
            'predict_decision': predicted_decisions_str[i],
            'reviews': [],
            'method':'TFIDF_initial review'
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

    print("加载数据...")

    np.random.seed(42)
    indices = np.random.permutation(len(file_2_data))
    val_indices, test_indices = indices[:1000], indices[1000:]
    val_data = [file_2_data[i] for i in val_indices]
    test_data = [file_2_data[i] for i in test_indices]

    print("Preparing datasets (using earliest reviewer comments)...")
    # --- MODIFIED: Capture the valid_papers for the test set ---
    X_train, y_train, _ = prepare_dataset(train_data)
    X_val, y_val, _ = prepare_dataset(val_data)
    X_test, y_test, test_data_valid = prepare_dataset(test_data)

    print(f"\nDataset Statistics:\nTraining: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    if not y_train or not y_test:
        print("\nError: Not enough data after filtering. Exiting.")
        return

    print("\nApplying TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=20000,  # 稍微增加一点特征数，或者保持 15000
        min_df=3,  # 保持不变，过滤极低频词
        max_df=0.6,  # 修改：从 0.9 降到 0.6，过滤掉"paper"这种太常见的词
        stop_words='english',
        ngram_range=(1, 2),  # 保持不变，二元语法对捕捉 "not good" 很重要
        sublinear_tf=True,  # 新增：平滑词频，非常重要！
        strip_accents='unicode'  # 新增：字符标准化
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"TF-IDF feature dimensions: {X_train_tfidf.shape[1]}")

    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=5.0)
    lr_model.fit(X_train_tfidf, y_train)

    y_val_pred = lr_model.predict(X_val_tfidf)
    y_test_pred = lr_model.predict(X_test_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\n--- Final Results ---\nValidation Accuracy: {val_accuracy:.4f}\nTest Accuracy: {test_accuracy:.4f}")

    # --- NEW: Generate and save the detailed dataset ---
    if test_data_valid:
        detailed_jsonl = create_detailed_predictions_jsonl(lr_model, tfidf_vectorizer, test_data_valid, X_test)

        if detailed_jsonl:
            # Descriptive filename for this specific experiment
            output_filename = '../../../Datasets/predict_result_new/initial review/TFIDF.jsonl'
            with open(output_filename, 'w', encoding='utf-8') as f:
                for entry in detailed_jsonl:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"\n详细数据集已成功保存到: {output_filename}")


if __name__ == "__main__":
    main()