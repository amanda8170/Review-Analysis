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


# --- NEW FUNCTION 1: 核心提取逻辑 (Weakness Only) ---
def extract_weakness_only(text: str) -> str:
    """
    【核心提取逻辑】只提取 Weakness 部分。
    """
    if not text:
        return ""

    # 1. 基础去HTML
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. 正则表达式
    pattern = re.compile(
        r'(?:Weaknesses?|Weakness)\s*:\s*(.*?)(?=\s*(?:Summary|Strengths?|Questions|Soundness|Presentation|Contribution|Rating|Confidence|Correctness)\s*:|$)',
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.findall(text)

    extracted_parts = []
    for content in matches:
        clean_content = " ".join(content.split())
        if clean_content:
            extracted_parts.append(clean_content)

    if extracted_parts:
        return "\n".join(extracted_parts)
    else:
        return ""


# --- NEW FUNCTION 2: 遍历所有 Reviewer 并提取 Weaknesses ---
def extract_all_reviews_weaknesses(reviews: List[Dict]) -> str:
    """
    提取所有审稿人的 Weaknesses。
    """
    if not reviews:
        return ""

    all_weaknesses = []

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

            # 4. 【只提取 Weakness】
            weakness_text = extract_weakness_only(first_item['content'])

            if weakness_text:
                comment_str = f"--- Weakness pointed out by {reviewer_id} ---\n{weakness_text}"
                all_weaknesses.append(comment_str)

    return "\n\n".join(all_weaknesses)


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
        # --- MODIFIED: 调用新的 Weakness 提取函数 ---
        reviews_list = paper.get('reviews', [])
        earliest_review_content = extract_all_reviews_weaknesses(reviews_list)

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
            'method': 'TFIDF_weakness'
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

    print("Preparing datasets (using Weakness Only extraction)...")
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
        max_features=20000,
        min_df=3,
        max_df=0.6,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode'
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
            output_filename = '../../../Datasets/predict_result_new/weakness/TFIDF.jsonl'
            with open(output_filename, 'w', encoding='utf-8') as f:
                for entry in detailed_jsonl:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"\n详细数据集已成功保存到: {output_filename}")


if __name__ == "__main__":
    main()