import json
import re


# 定义提取函数
def extract_prediction(raw_text):
    if not raw_text:
        return "Unknown"

    # 逻辑1：使用正则匹配 Decision: [Accept/Reject]
    # pattern 解释：寻找 "Decision:" 后面跟着空格，然后是 "["，捕获内容，最后是 "]"
    match = re.search(r"Decision:\s*\[(Accept|Reject)\]", raw_text, re.IGNORECASE)

    if match:
        # group(1) 是括号内捕获的内容
        return match.group(1).title()  # .title() 保证首字母大写，例如 "reject" -> "Reject"

    # 逻辑2：如果正则没匹配到（比如格式乱了），可以尝试关键词兜底
    lower_text = raw_text.lower()
    if "accept" in lower_text:
        return "Accept"
    elif "reject" in lower_text:
        return "Reject"

    return "Unknown"


# 初始化计数器
total_count = 0
correct_count = 0

# 存储处理后的数据（如果需要保存）
processed_data = []

file_path = '../../../../Datasets/predict_result/llm/test/gemini/initial review_rating.jsonl'  # 请修改为你的实际文件名

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            item = json.loads(line)

            # 1. 获取真实标签
            real_decision = item.get('real_decision', '').strip()

            # 2. 从 raw_output 提取预测
            raw_output = item.get('raw_output', '')
            prediction = extract_prediction(raw_output)

            # 3. 更新字段（可选，用于查看结果）
            item['predict_decision'] = prediction

            # 4. 只有当提取出有效结果且真实标签存在时才计算准确率
            if prediction != "Unknown" and real_decision:
                total_count += 1
                # 忽略大小写进行比较
                if prediction.lower() == real_decision.lower():
                    correct_count += 1

            processed_data.append(item)

    # 计算准确率
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"总样本数: {total_count}")
        print(f"预测正确数: {correct_count}")
        print(f"准确率 (Accuracy): {accuracy:.2%}")
    else:
        print("未找到有效样本进行计算。")

    with open('data_processed.jsonl', 'w', encoding='utf-8') as f_out:
        for item in processed_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
except json.JSONDecodeError:
    print("错误：JSON 格式解析失败，请检查文件内容。")