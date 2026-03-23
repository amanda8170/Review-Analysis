import json
import re
import os
import time
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
import httpx
from tqdm import tqdm

# --- 1. 配置区域 ---
# 请在 CloseAI 平台获取 sk-开头的令牌
API_KEY = "sk-G1p0ufbiOpPDHqpGUb4LwDkueMYgssh117Ei4wP00ke4VVYm"
# CloseAI 的接口地址，必须加 /v1
BASE_URL = "https://api.closeai-asia.com/v1"
# 你想用的模型，例如: gpt-4o, gpt-4-turbo, claude-3-5-sonnet, gemini-1.5-pro
MODEL_NAME = "gemini-2.5-pro"

TEST_DATA_FILE = '../../../../Datasets/formatted_dataset/ICLR.cc_2025_formatted.jsonl'
OUTPUT_FILE = '../../../../Datasets/predict_result/llm/test/gemini/initial review_rating.jsonl'

# --- 数据划分配置 ---
VALIDATION_SET_SIZE = 100


def setup_llm_api(api_key: str):
    global client
    # 初始化 OpenAI 客户端，指向 CloseAI 的中转地址
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL
    )


def load_data(file_path: str) -> List[Dict]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {file_path}")
    return data


def clean_specific_fields(text: str) -> str:
    """
    【核心清洗逻辑】
    只删除 those "Key: Number" 格式的行。
    保留包含文字描述的行。
    """
    if not text:
        return ""

    # 1. 基础去HTML
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. 定义严格的删除模式
    pattern = r'(?m)^\s*(?:Soundness|Presentation|Contribution|Rating|Confidence|Correctness|Summary|Questions|Strengths|Weaknesses)\s*:\s*\-?\d+\s*$'

    # 将匹配到的行替换为空字符串
    text = re.sub(pattern, '', text)

    # 3. 清理残留的空行和多余空格
    cleaned_text = " ".join(text.split())

    return cleaned_text


def extract_initial_comments_per_reviewer(reviews: List[Dict]) -> str:
    if not reviews:
        return ""

    all_initial_comments = []

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

            # 4. 【调用清洗函数】
            content_cleaned = clean_specific_fields(first_item['content'])

            comment_str = f"--- Opinion from {reviewer_id} ---\n{content_cleaned}"
            all_initial_comments.append(comment_str)

    return "\n\n".join(all_initial_comments)


def get_prediction_from_llm(content: str, scores: List[int], max_retries: int = 3) -> str:
    # --- Prompt 修改：只要求 Decision，不要 Reason ---
    prompt_template = """Based on the provided Reviewer Ratings and Review Content, predict the final acceptance decision for this paper, drawing upon your knowledge of ICLR conference standards and historical acceptance trends.
1. Rating Scale:10 (Strong Accept), 8 (Accept), 6 (Marginally Accept), 5 (Marginally Reject), 3 (Reject), 1 (Strong Reject).
2. Review Content: Includes Summary, Strengths, Weaknesses, and Questions from reviewers.
Reviewer Ratings: {scores_list}
Reviewer Comments: {content}
You must output the result strictly in the following format:
Decision: [Accept or Reject]
Do not provide any reasons or explanations."""

    prompt = prompt_template.format(content=content, scores_list=str(scores))

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,  # 使用 gpt-5
                messages=[{"role": "user", "content": prompt}],
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"API调用异常 (Attempt {attempt + 1}): {e}")
            time.sleep(1)

    return "API_FAILED"


def parse_prediction(raw_output: str) -> str:
    """
    解析模型输出，提取 Accept 或 Reject。
    """
    if not raw_output:
        return "N/A"

    text = raw_output.strip().lower()

    if "decision: accept" in text or "decision:accept" in text:
        return "Accept"
    elif "decision: reject" in text or "decision:reject" in text:
        return "Reject"
    # 容错
    elif text == "accept":
        return "Accept"
    elif text == "reject":
        return "Reject"

    return "N/A"


def format_output_data(original_paper: Dict, scores: List[int], extracted_content: str, model_raw_output: str,
                       predict_decision: str) -> Dict[str, Any]:
    return {
        "paper_title": original_paper.get("paper_title", "N/A"),
        "rating": scores,
        "real_decision": original_paper.get("paper_decision", ""),
        "input_extracted_text": extracted_content[:200] + "...",  # 预览
        "raw_output": model_raw_output,
        "predict_decision": predict_decision
    }


def predict_and_save(papers_to_process: List[Dict], output_file: str):
    # 路径检查
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"开始处理 {len(papers_to_process)} 篇论文，结果将保存至 {output_file}...")

    total_valid_predictions = 0
    correct_predictions = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for paper in tqdm(papers_to_process, desc="Processing"):

            # 1. 提取并清洗文本
            combined_content = extract_initial_comments_per_reviewer(paper.get('reviews', []))

            # 2. 提取分数
            scores = []
            for review in paper.get('reviews', []):
                rating_str = review.get("rating", "")
                if rating_str and rating_str != "-1":
                    try:
                        scores.append(int(rating_str.split(':')[0]))
                    except ValueError:
                        continue

            if not combined_content and not scores:
                continue

            # 3. 预测
            model_raw_output = get_prediction_from_llm(combined_content, scores)

            # 4. 解析
            predict_decision = parse_prediction(model_raw_output)

            # 5. 保存
            output_data = format_output_data(paper, scores, combined_content, model_raw_output, predict_decision)
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')

            # 6. 计算 Accuracy
            if predict_decision != "N/A":
                real_decision_str = paper.get("paper_decision", "").lower()
                normalized_real = "Accept" if "accept" in real_decision_str else (
                    "Reject" if "reject" in real_decision_str else "Unknown")

                if normalized_real != "Unknown":
                    total_valid_predictions += 1
                    if predict_decision == normalized_real:
                        correct_predictions += 1

            time.sleep(0.5)

    # 打印统计结果
    print("-" * 30)
    print(f"处理完成！")
    print(f"有效预测总数: {total_valid_predictions}")
    print(f"预测正确数: {correct_predictions}")
    if total_valid_predictions > 0:
        accuracy = correct_predictions / total_valid_predictions
        print(f"Accuracy: {accuracy:.2%}")
    else:
        print("Accuracy: N/A (无有效预测)")
    print(f"结果已保存至 {output_file}")
    print("-" * 30)


def main():
    try:
        setup_llm_api(API_KEY)
    except ValueError as e:
        print(f"配置错误: {e}")
        return

    print(f"从 {TEST_DATA_FILE} 加载数据...")
    source_data = load_data(TEST_DATA_FILE)
    if not source_data:
        return

    # --- 保持完全一致的随机划分 ---
    print("根据 random_seed=42 划分并提取测试集...")
    np.random.seed(42)
    indices = np.random.permutation(len(source_data))
    test_indices = indices[VALIDATION_SET_SIZE:]
    test_data = [source_data[i] for i in test_indices]

    if test_data:
        # --- 测试 100 条 ---
        papers_to_process = test_data[:100]

        print(f"验证：第一篇论文标题为 -> {papers_to_process[0].get('paper_title', 'N/A')}")
        predict_and_save(papers_to_process, OUTPUT_FILE)
    else:
        print("没有需要处理的数据。")


if __name__ == "__main__":
    main()