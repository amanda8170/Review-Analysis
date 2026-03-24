import json
import os
import time
import numpy as np
from typing import List, Dict, Any
import dashscope
from http import HTTPStatus
from tqdm import tqdm
import re

# --- 1. 配置区域 ---
API_KEY = os.getenv("DASHSCOPE_API_KEY", "xxx")

# --- 文件路径 ---
DATA_FILE = '../../../../Datasets/formatted_dataset/ICLR.cc_2025_formatted.jsonl'
OUTPUT_FILE = '../../../../Datasets/predict_result/llm/test/Qwen/weakness_strength.jsonl'

# --- 数据划分配置 ---
VALIDATION_SET_SIZE = 100


def setup_llm_api(api_key: str):
    if not api_key or "sk-" not in api_key:
        raise ValueError("API Key无效或未设置。")
    dashscope.api_key = api_key


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


def extract_strengths_and_weaknesses(text: str) -> str:
    """
    【提取逻辑】只提取 Strengths 和 Weaknesses。
    """
    if not text:
        return ""

    # 1. 基础去HTML
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. 正则表达式
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
            extracted_parts.append(f"[{header_normalized}]: {clean_content}")

    if extracted_parts:
        return "\n".join(extracted_parts)
    else:
        return ""


def extract_all_reviews_sw(reviews: List[Dict]) -> str:
    """
    提取所有审稿人的 Strengths 和 Weaknesses。
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
                comment_str = f"--- Opinion from {reviewer_id} ---\n{sw_text}"
                all_content.append(comment_str)

    return "\n\n".join(all_content)


def get_prediction_from_llm(content: str, max_retries: int = 3) -> str:
    # --- Prompt 修改：只要求 Decision，不要求 Reason ---
    prompt_template = """Drawing upon your extensive knowledge of ICLR conference standards and historical acceptance trends, critically analyze the trade-off between the provided Strengths and Weaknesses to predict the final decision for this paper.
Provided Strengths and Weaknesses: {review_strengths_and_weaknesses}
You must output the result strictly in the following format:
Decision: [Accept or Reject]
Do not provide any reasons or explanations."""

    # 只传入文本
    prompt = prompt_template.format(review_strengths_and_weaknesses=content)

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


def format_output_data(original_paper: Dict, scores: List[int], extracted_content: str, model_raw_output: str,
                       predict_decision: str) -> Dict[str, Any]:
    # 输出包含 paper_title, rating(作为参考), real_decision, raw_output, predict_decision
    return {
        "paper_title": original_paper.get("paper_title", "N/A"),
        "rating": scores,  # 虽然 Prompt 没用到，但保留在 JSON 中方便分析
        "real_decision": original_paper.get("paper_decision", ""),
        "input_extracted_text": extracted_content[:200] + "...",  # 只存一部分预览，防止文件过大
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

            # 1. 提取 Strengths + Weaknesses
            combined_sw = extract_all_reviews_sw(paper.get('reviews', []))

            # 2. 提取 Rating (仅用于输出文件记录，不进 Prompt)
            scores = []
            for review in paper.get('reviews', []):
                rating_str = review.get("rating", "")
                if rating_str and rating_str != "-1":
                    try:
                        scores.append(int(rating_str.split(':')[0]))
                    except ValueError:
                        continue

            # 如果没有提取到任何 Strengths/Weaknesses，跳过
            if not combined_sw:
                continue

            # 3. 预测 (仅传入文本)
            model_raw_output = get_prediction_from_llm(combined_sw)

            # 4. 解析
            predict_decision = parse_prediction(model_raw_output)

            # 5. 保存
            output_data = format_output_data(paper, scores, combined_sw, model_raw_output, predict_decision)
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

    print(f"从 {DATA_FILE} 加载数据...")
    source_data = load_data(DATA_FILE)
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
