import openreview
import json
import time

# 使用旧版API初始化客户端
client = openreview.Client(
    baseurl='https://api.openreview.net',
    username="yingxuan.wen@edu.em-lyon.com",
    password="Amanda2310504"
)

# 会议信息
venue_id = 'ICLR.cc/2024/Conference'
invitations = [
    f'{venue_id}/-/Blind_Submission',
    f'{venue_id}/-/Withdrawn_Submission',
    f'{venue_id}/-/Desk_Rejected_Submission',
]

all_submissions = []

for inv in invitations:
    print(f"\n正在获取 invitation: {inv}")
    offset = 0

    while True:
        try:
            batch = client.get_notes(
                invitation=inv,
                offset=offset,
                limit=1000,
                details='replies'
            )

            if not batch:
                break

            print(f"获取到 {len(batch)} 条")
            all_submissions.extend(batch)

            if len(batch) < 1000:
                break

            offset += 1000
            time.sleep(1)

        except Exception as e:
            print(f"错误: {e}")
            break

print(f"总共获取到 {len(all_submissions)} 篇提交")

# 保存数据（使用第一个代码的简单方式）
file_name = f"{venue_id.split('/')[0]}_{venue_id.split('/')[1]}.jsonl"

with open(file_name, 'w', encoding='utf-8') as f:
    for submission in all_submissions:
        review_list = []

        # 获取评审数据
        if hasattr(submission, 'details') and 'replies' in submission.details:
            for reply in submission.details['replies']:
                # 简化处理：直接转换为JSON或使用原始数据
                try:
                    if hasattr(reply, 'to_json'):
                        review = reply.to_json()
                    else:
                        review = reply  # 如果已经是字典格式，直接使用
                    review_list.append(review)
                except:
                    # 如果转换失败，跳过这个回复
                    continue

        # 转换提交数据（简化处理）
        try:
            sub_json = submission.to_json()
        except:
            # 如果没有to_json方法，尝试直接使用对象的__dict__
            sub_json = submission.__dict__ if hasattr(submission, '__dict__') else {}

        # 添加评审列表
        sub_json['reviews'] = review_list

        # 写入文件
        json_line = json.dumps(sub_json, ensure_ascii=False)
        f.write(json_line + '\n')
        f.flush()

print(f"数据已保存到 {file_name}")

# 简单统计
with_reviews = sum(1 for sub in all_submissions
                   if hasattr(sub, 'details') and 'replies' in sub.details and sub.details['replies'])

print(f"\n=== 数据统计 ===")
print(f"总提交数: {len(all_submissions)}")
print(f"有评审的提交数: {with_reviews}")