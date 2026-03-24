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
API_KEY = "xxx"
# CloseAI 的接口地址，必须加 /v1
BASE_URL = "https://api.closeai-asia.com/v1"
# 你想用的模型，例如: gpt-4o, gpt-4-turbo, claude-3-5-sonnet, gemini-1.5-pro
MODEL_NAME = "gemini-2.5-pro"

TEST_DATA_FILE = '../../../../Datasets/formatted_dataset/ICLR.cc_2025_formatted.jsonl'
OUTPUT_FILE = '../../../../Datasets/predict_result/llm/test/gemini/rating.jsonl'

# --- 数据划分配置 ---
VALIDATION_SET_SIZE = 100


def setup_llm_api(api_key: str):
    global client
    # 初始化 OpenAI 客户端，指向 CloseAI 的中转地址
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL
    )

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {file_path}")
    return data


def get_decision_from_qwen(scores: List[int], max_retries: int = 3) -> str:
    # --- 修改点 1: 修改 Prompt，只要求输出 Decision，不要 Reason ---
    prompt_template = """Drawing upon your extensive knowledge of ICLR conference standards and historical acceptance trends, predict the final decision for this paper based solely on the provided reviewer scores: {scores_list}.
The scoring scale is: 10: Strong Accept; 8: Accept; 6: Marginally Accept; 5: Marginally Reject; 3: Reject; 1: Strong Reject
You must output the result strictly in the following format:
Decision: [Accept or Reject]
Do not provide any reasons or explanations.
"""
    prompt = prompt_template.format(scores_list=str(scores))

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

    return "API调用失败"


def parse_prediction(raw_output: str) -> str:
    """
    解析模型输出，提取 Accept 或 Reject。
    如果无法提取，返回 N/A。
    """
    if not raw_output:
        return "N/A"

    text = raw_output.strip().lower()

    # 简单的关键词匹配逻辑
    if "decision: accept" in text or "decision:accept" in text:
        return "Accept"
    elif "decision: reject" in text or "decision:reject" in text:
        return "Reject"
    # 如果模型只输出了单词
    elif text == "accept":
        return "Accept"
    elif text == "reject":
        return "Reject"

    return "N/A"


def format_output_data(original_paper: Dict, scores: List[int], raw_output: str, predict_decision: str) -> Dict[
    str, Any]:
    # --- 修改点 2: 输出字段包含 rating(scores), real_decision, raw_output, predict_decision ---
    return {
        "paper_title": original_paper.get("paper_title", "N/A"),
        "rating": scores,
        "real_decision": original_paper.get("paper_decision", ""),
        "raw_output": raw_output,
        "predict_decision": predict_decision
    }


def predict_and_save(papers_to_process: List[Dict], output_file: str):
    print(f"开始处理 {len(papers_to_process)} 篇论文，结果将保存至 {output_file}...")

    total_valid_predictions = 0
    correct_predictions = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for paper in tqdm(papers_to_process, desc="正在生成预测"):
            scores = [
                int(review['rating'].split(':')[0])
                for review in paper.get('reviews', [])
                if review.get("rating") and review.get("rating") != "-1"
            ]

            if not scores:
                continue

            # 获取模型原始输出
            raw_output = get_decision_from_qwen(scores)

            # 解析预测结果
            predict_decision = parse_prediction(raw_output)

            # 格式化输出
            output_data = format_output_data(paper, scores, raw_output, predict_decision)
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')

            # --- 修改点 3: 计算 Accuracy ---
            if predict_decision != "N/A":
                real_decision_str = paper.get("paper_decision", "").lower()

                # 归一化真实标签：只要包含 accept 就算 Accept，包含 reject 就算 Reject
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


# --- 3. 主程序入口 ---
def main():
    """主执行函数"""
    try:
        setup_llm_api(API_KEY)
    except ValueError as e:
        print(f"错误: {e}")
        return

    print(f"从 {TEST_DATA_FILE} 加载数据...")
    source_data = load_jsonl(TEST_DATA_FILE)
    if not source_data:
        return

    print("根据 random_seed=42 划分并提取测试集...")
    np.random.seed(42)
    indices = np.random.permutation(len(source_data))

    # --- 这里的逻辑是划分出验证集 ---
    test_indices = indices[VALIDATION_SET_SIZE:]
    test_data = [source_data[i] for i in test_indices]

    if test_data:
        # --- 修改点 4: 测试 100 条 ---
        papers_to_process = test_data[:100]

        print(f"已截取测试集前 100 条数据进行预测。")
        predict_and_save(papers_to_process, OUTPUT_FILE)
    else:
        print("测试集为空，没有数据可供处理。")


if __name__ == "__main__":
    main()
