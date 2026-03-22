import json
import re
import os
import time
import numpy as np
from typing import List, Dict, Any
import dashscope
from http import HTTPStatus
from tqdm import tqdm

# --- 1. 配置区域 ---
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-aea5b38d511d4e359db4e0b881f516a0")
TEST_DATA_FILE = '../../../../Datasets/formatted_dataset/ICLR.cc_2025_formatted.jsonl'
OUTPUT_FILE = '../../../../Datasets/predict_result/llm/test/Qwen/initial review.jsonl'

# --- 数据划分配置 ---
VALIDATION_SET_SIZE = 100


def setup_qwen_api(api_key: str):
    if not api_key or "sk-" not in api_key:
        raise ValueError("API Key无效或未设置。")
    dashscope.api_key = api_key


def load_data(file_path: str) -> List[Dict]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {file_path}")
    return data


def clean_text_basic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    return " ".join(text.split())


def extract_initial_reviews_full(reviews: List[Dict]) -> str:
    """
    提取逻辑：
    1. 排除 Chair/Author
    2. 找每个 Reviewer 最早发言
    3. 全量拼接，无任何截断
    """
    all_initial_comments = []

    if not reviews:
        return ""

    for review in reviews:
        reviewer_id = review.get('reviewer', 'Unknown')

        # 1. 筛选角色
        if any(role in reviewer_id for role in ['Program_Chair', 'Area_Chair', 'Author']):
            continue

        dialogue_list = review.get('dialogue', [])
        valid_items = [
            item for item in dialogue_list
            if isinstance(item, dict) and 'time' in item and 'content' in item and item['content']
        ]

        if valid_items:
            # 2. 找最早发言
            first_item = min(valid_items, key=lambda d: d['time'])
            content_text = clean_text_basic(first_item['content'])

            # 3. 格式化
            comment_str = f"--- Opinion from {reviewer_id} ---\n{content_text}"
            all_initial_comments.append(comment_str)

    return '\n\n'.join(all_initial_comments)


def get_decision_from_qwen(content: str, max_retries: int = 3) -> str:
    # --- Prompt 修改：只要求 Decision，不要 Reason ---
    prompt_template = """Drawing upon your extensive knowledge of ICLR conference standards and historical acceptance trends, analyze the content of the provided initial reviews to predict the final decision for this paper.
Reviewer Comments:
{reviews}
You must output the result strictly in the following format:
Decision: [Accept or Reject]
Do not provide any reasons or explanations.
"""

    # 这里将提取到的文本填入占位符
    prompt = prompt_template.format(reviews=content)

    for attempt in range(max_retries):
        try:
            response = dashscope.Generation.call(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
                temperature=0.01,
            )
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                print(f"API Error (Attempt {attempt + 1}): {response.code}")
                time.sleep(1)
        except Exception as e:
            print(f"Exception (Attempt {attempt + 1}): {e}")
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
    # 容错：如果模型直接输出了单词
    elif text == "accept":
        return "Accept"
    elif text == "reject":
        return "Reject"

    return "N/A"


def format_output_data(original_paper: Dict, full_extracted_content: str, model_raw_output: str,
                       predict_decision: str) -> Dict[str, Any]:
    return {
        "paper_title": original_paper.get("paper_title", "N/A"),
        "real_decision": original_paper.get("paper_decision", "N/A"),
        "input_extracted_text": full_extracted_content[:200] + "...",  # 预览
        "raw_output": model_raw_output,
        "predict_decision": predict_decision
    }


def predict_and_save(papers_to_process: List[Dict], output_file: str):
    print(f"开始处理 {len(papers_to_process)} 篇论文...")

    total_valid_predictions = 0
    correct_predictions = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for paper in tqdm(papers_to_process, desc="Processing"):

            # 1. 提取全量文本
            content = extract_initial_reviews_full(paper.get('reviews', []))

            if not content:
                print(f"Skipping {paper.get('paper_title')[:20]}... (No valid text)")
                continue

            # 2. 预测
            model_raw_output = get_decision_from_qwen(content)

            # 3. 解析
            predict_decision = parse_prediction(model_raw_output)

            # 4. 保存
            output_data = format_output_data(paper, content, model_raw_output, predict_decision)
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')

            # 5. 计算 Accuracy
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
        setup_qwen_api(API_KEY)
    except ValueError as e:
        print(f"错误: {e}")
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
        # 修改为前 100 条
        papers_to_process = test_data[:100]

        print(f"验证：第一篇论文标题为 -> {papers_to_process[0].get('paper_title', 'N/A')}")

        predict_and_save(papers_to_process, OUTPUT_FILE)
    else:
        print("无数据。")


if __name__ == "__main__":
    main()