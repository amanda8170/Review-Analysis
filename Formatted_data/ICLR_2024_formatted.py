import json
import jsonlines
from typing import Dict, List, Any, Set
from collections import defaultdict
from datetime import datetime


class OpenReviewProcessor:
    @staticmethod
    def _get_signature(review: Dict) -> str:
        """安全地获取第一个签名 (改为静态方法)"""
        signatures = review.get('signatures')
        return signatures[0] if signatures and isinstance(signatures, list) and len(signatures) > 0 else "Unknown"

    # --- 1. 新增: 统一的决策逻辑函数 ---
    @staticmethod
    def get_final_decision(data: Dict) -> str:
        """
        确定一篇论文的最终决定，核心逻辑集中在此。
        """
        # 获取最原始的决定
        original_decision = OpenReviewProcessor.extract_decision(data)

        # 如果是 'withdraw'，则应用特殊逻辑
        if original_decision and 'withdraw' in original_decision.lower():
            reviews = data.get('reviews', [])
            # 检查是否存在任何签名中明确包含 "Reviewer" 的评论
            has_review_comments = any(
                'Reviewer' in OpenReviewProcessor._get_signature(r)
                for r in reviews
            )

            # 根据有无评审意见，返回最终的决定
            if has_review_comments:
                return 'Reject'  # 有评审意见，视为 Reject
            else:
                return 'Withdraw'  # 无评审意见，标记为 Withdraw，以便后续过滤

        # 如果不是 'withdraw'，直接返回原始决定
        return original_decision

    # --- 2. 修改: extract_paper_info 使用新函数 ---
    def extract_paper_info(self, data: Dict) -> Dict:
        """提取论文基本信息"""
        content = data.get('content', {})
        original_track = content.get('primary_area', {}).get('value', '')
        paper_info = {
            'paper_title': content.get('title', {}).get('value', ''),
            'paper_authors': content.get('authors', {}).get('value', []),
            'paper_abstract': content.get('abstract', {}).get('value', ''),
            'paper_keywords': content.get('keywords', {}).get('value', []),
            'paper_tldr': content.get('TLDR', {}).get('value', ''),
            'paper_track': original_track.replace('_', ' ') if original_track else '',
            'paper_venue': content.get('venue', {}).get('value', ''),
            # 直接使用统一的决策函数获取最终决定
            'paper_decision': OpenReviewProcessor.get_final_decision(data)
        }
        return paper_info

    @staticmethod
    def extract_decision(data: Dict) -> str:
        """
        从数据中提取论文最原始的决定。
        按优先级顺序检查: 'venueid', 'venue', 然后是 'reviews' 列表。
        """
        content = data.get('content', {})

        # 优先级1: 检查 venueid
        venue_id = content.get('venueid', '')
        if venue_id:
            if 'Withdrawn_Submission' in venue_id:
                return 'Withdraw'
            if 'Desk_Rejected' in venue_id:
                return 'Desk Reject'

        # 优先级2: 检查 venue 字段
        venue = content.get('venue', {}).get('value', '')
        if venue:  # 确保 venue 字符串存在
            if 'Withdrawn Submission' in venue:
                return 'Withdraw'
            # --- 【这是关键的新增行】 ---
            if 'Desk Rejected' in venue:
                return 'Desk Reject'
            # --- 【新增行结束】 ---

        # 优先级3: 在 reviews 列表中寻找 decision
        reviews = data.get('reviews', [])
        for review in reviews:
            review_content = review.get('content', {})
            if 'decision' in review_content:
                decision_value = review_content['decision'].get('value', '') if isinstance(
                    review_content['decision'],
                    dict) else str(
                    review_content['decision'])
                # 返回前进行标准化，统一大小写
                if decision_value:
                    if 'oral' in decision_value.lower(): return 'Accept (oral)'
                    if 'spotlight' in decision_value.lower(): return 'Accept (spotlight)'
                    if 'poster' in decision_value.lower(): return 'Accept (poster)'
                    return decision_value
                return 'Pending'

        return 'Pending'

    # --- 4. 修改: filter_papers 使用新函数 ---
    @staticmethod
    def filter_papers(papers_data: List[Dict]) -> List[Dict]:
        """根据论文的最终决定进行过滤"""
        filtered_papers = []
        # 使用您代码中提供的准确列表
        decisions_to_keep = ['Accept (oral)', 'Accept (poster)', 'Accept (spotlight)', 'Reject']

        for paper in papers_data:
            # 使用统一的决策函数获取最终决定
            final_decision = OpenReviewProcessor.get_final_decision(paper)

            # 如果最终决定是我们想要保留的类型，则加入列表
            if final_decision in decisions_to_keep:
                filtered_papers.append(paper)

        return filtered_papers

    def determine_review_type(self, review: Dict) -> str:
        """确定review类型"""
        signature = OpenReviewProcessor._get_signature(review)
        if 'Authors' in signature:
            return 'author_response'
        else:
            return 'reviewer_review'

    def extract_review_content(self, review: Dict) -> str:
        """提取review内容"""
        content = review.get('content', {})
        if 'summary' in content:
            parts = []
            if content.get('summary', {}).get('value'):
                parts.append(f"Summary: {content['summary']['value']}")
            if content.get('soundness', {}).get('value'):
                parts.append(f"Soundness: {content['soundness']['value']}")
            if content.get('presentation', {}).get('value'):
                parts.append(f"Presentation: {content['presentation']['value']}")
            if content.get('contribution', {}).get('value'):
                parts.append(f"Contribution: {content['contribution']['value']}")
            if content.get('strengths', {}).get('value'):
                parts.append(f"Strengths: {content['strengths']['value']}")
            if content.get('weaknesses', {}).get('value'):
                parts.append(f"Weaknesses: {content['weaknesses']['value']}")
            if content.get('questions', {}).get('value'):
                parts.append(f"Questions: {content['questions']['value']}")

            return "\n\n".join(parts)
        elif 'comment' in content:
            return content.get('comment', {}).get('value', '')
        elif 'rebuttal' in content:
            return content.get('rebuttal', {}).get('value', '')
        elif 'metareview' in content:
            parts = []
            if content.get('metareview', {}).get('value'):
                parts.append(f"Metareview: {content['metareview']['value']}")
            if content.get('justification_for_why_not_higher_score', {}).get('value'):
                parts.append(
                    f"Justification For Why Not Higher Score: {content['justification_for_why_not_higher_score']['value']}")
            if content.get('justification_for_why_not_lower_score', {}).get('value'):
                parts.append(
                    f"Justification For Why Not Lower Score: {content['justification_for_why_not_lower_score']['value']}")
            return "\n\n".join(parts) if parts else ''

    def extract_rating(self, review: Dict) -> str:
        """提取评分"""
        content = review.get('content', {})
        rating = content.get('rating', {}).get('value', '')
        return str(rating) if rating else '-1'

    def extract_confidence(self, review: Dict) -> str:
        """提取审稿人自信度"""
        content = review.get('content', {})
        confidence = content.get('confidence', {}).get('value', '')
        return str(confidence) if confidence else '-1'

    def group_reviews_by_thread(self, reviews: List[Dict]) -> Dict[str, List[Dict]]:
        if not reviews:
            return {}
        review_map = {r['id']: r for r in reviews if 'id' in r}
        memo = {}

        def find_root(comment_id: str, path: Set[str]) -> str:
            if comment_id in memo: return memo[comment_id]
            if comment_id in path: return comment_id
            path.add(comment_id)
            review = review_map.get(comment_id)
            if not review:
                memo[comment_id] = comment_id
                return comment_id
            parent_id = review.get('replyto')
            forum_id = review.get('forum')
            if not parent_id or parent_id == forum_id or parent_id not in review_map:
                memo[comment_id] = comment_id
                return comment_id
            root = find_root(parent_id, path)
            memo[comment_id] = root
            return root

        threads = defaultdict(list)
        for review in reviews:
            if 'id' in review:
                root_id = find_root(review['id'], set())
                threads[root_id].append(review)
        for thread_id in threads:
            threads[thread_id].sort(key=lambda x: x.get('cdate', 0))
        return dict(threads)

    def classify_paper(self, reviews: List[Dict]) -> str:
        """
        修正的分类逻辑：
        1. self_reply: 作者发起的线程
        2. cross_reply: 同一个线程中出现了2个或以上不同的审稿人
        3. normal: 其他情况
        """
        if not reviews:
            return "normal"

        threads = self.group_reviews_by_thread(reviews)
        if not threads:
            threads = {'_main_': sorted(reviews, key=lambda x: x.get('cdate', 0))}

        # 步骤1: 检查是否存在 self-reply 线程（作者发起的线程）
        is_self_reply = False
        for thread_reviews in threads.values():
            if not thread_reviews:
                continue

            # 检查线程发起者是否为作者
            first_poster_sig = OpenReviewProcessor._get_signature(thread_reviews[0])
            if 'Authors' in first_poster_sig:
                is_self_reply = True
                break

        if is_self_reply:
            return "self_reply"

        # 步骤2: 检查是否存在 cross-reply（同一线程中有多个不同审稿人）
        is_cross_reply = False

        for thread_reviews in threads.values():
            if not thread_reviews:
                continue

            # 提取这个线程中所有审稿人的签名
            reviewer_signatures_in_thread = set()
            for review in thread_reviews:
                sig = OpenReviewProcessor._get_signature(review)
                # 只统计审稿人签名
                if '/Reviewer_' in sig:
                    reviewer_signatures_in_thread.add(sig)

            # 如果同一个线程中有2个或以上不同的审稿人，就是cross-reply
            if len(reviewer_signatures_in_thread) >= 2:
                is_cross_reply = True
                break

        if is_cross_reply:
            return "cross_reply"

        # 步骤3: 剩下的都是 normal
        return "normal"

    def get_reviewer_info(self, thread_reviews: List[Dict]) -> Dict:
        """从线程中提取首个非作者reviewer的信息"""
        for review in thread_reviews:
            signature = OpenReviewProcessor._get_signature(review)
            if 'Authors' not in signature:
                return {
                    'reviewer': signature,
                    'rating': self.extract_rating(review),
                    'confidence': self.extract_confidence(review)
                }
        return {'reviewer': 'Author', 'rating': '-1', 'confidence': '-1'}

    def create_dialogue_from_thread(self, thread_reviews: List[Dict]) -> List[Dict]:
        """从线程创建dialogue列表"""
        dialogue = []

        for review in thread_reviews:
            cdate_ms = review.get('cdate')
            time_str = ""  # 默认值
            if cdate_ms:
                # 将毫秒时间戳/1000得到秒，然后格式化
                time_str = datetime.fromtimestamp(cdate_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')

            dialogue.append({
                'time': time_str,
                'content': self.extract_review_content(review),
                'review_type': self.determine_review_type(review)
            })
        return dialogue

    def process_single_paper(self, data: Dict) -> Dict:
        """处理单篇论文的数据，格式化为最终输出"""
        paper_info = self.extract_paper_info(data)
        reviews = [review for review in data.get('reviews', [])
                   if not OpenReviewProcessor._get_signature(review).endswith('/Public_Comment')]
        review_threads = self.group_reviews_by_thread(reviews)
        processed_reviews = []
        for thread_id, thread_reviews in review_threads.items():
            reviewer_info = self.get_reviewer_info(thread_reviews)
            dialogue = self.create_dialogue_from_thread(thread_reviews)
            processed_reviews.append({
                'reviewer': reviewer_info['reviewer'],
                'rating': reviewer_info['rating'],
                'confidence': reviewer_info['confidence'],
                'dialogue': dialogue
            })
        processed_reviews.sort(key=lambda x: x.get('reviewer', ''))
        return {**paper_info, 'reviews': processed_reviews}

    def process_dataset(self, input_file: str, output_file: str):
            # --- 步骤 1: 初始化所有计数器 ---
            # 预处理前统计
            pre_counts = defaultdict(int)
            # 处理后（最终写入文件）的统计
            post_counts = defaultdict(int)
            # 被丢弃论文的统计
            count_dropped_desk_reject = 0
            count_dropped_withdraw_no_review = 0
            count_dropped_cross_reply = 0
            count_written = 0

            print(f"正在读取并处理文件: {input_file}...")

            all_papers = []
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_papers.append(json.loads(line))
            except FileNotFoundError:
                print(f"错误: 输入文件未找到 at path: {input_file}")
                return
            except json.JSONDecodeError:
                print(f"错误: 文件 {input_file} 包含无效的JSON。")
                return

            # --- 步骤 2: 在单个循环中处理所有逻辑 ---
            with jsonlines.open(output_file, 'w') as writer:
                for paper_data in all_papers:
                    # --- a. 预处理统计 ---
                    # 获取最原始的决定用于统计
                    original_decision = self.extract_decision(paper_data)
                    if 'desk reject' in original_decision.lower():
                        pre_counts['Desk Reject'] += 1
                    elif 'withdraw' in original_decision.lower():
                        pre_counts['Withdraw'] += 1
                    elif 'accept' in original_decision.lower():
                        pre_counts['Accept'] += 1
                    elif 'reject' in original_decision.lower():
                        pre_counts['Reject'] += 1
                    else:
                        pre_counts['Other/Pending'] += 1

                    # --- b. 应用过滤和转换规则 ---
                    # 获取最终决定 (此函数内置了 withdraw -> reject 的逻辑)
                    final_decision = self.get_final_decision(paper_data)

                    # 规则 1: 舍弃 Desk Reject
                    if 'desk reject' in final_decision.lower():
                        count_dropped_desk_reject += 1
                        continue

                    # 规则 2: 舍弃没有 review 的 Withdraw
                    if 'withdraw' in final_decision.lower():
                        count_dropped_withdraw_no_review += 1
                        continue

                    # 规则 3: 舍弃 Cross-Reply 类型的论文
                    raw_reviews = [r for r in paper_data.get('reviews', [])
                                   if not self._get_signature(r).endswith('/Public_Comment')]
                    category = self.classify_paper(raw_reviews)
                    if category == 'cross_reply':
                        count_dropped_cross_reply += 1
                        continue

                    # --- c. 写入数据并进行处理后统计 ---
                    # 如果论文通过所有过滤，则处理并写入
                    if final_decision in ['Reject', 'Accept (oral)', 'Accept (poster)', 'Accept (spotlight)']:
                        try:
                            processed_data = self.process_single_paper(paper_data)
                            writer.write(processed_data)
                            count_written += 1

                            # 根据最终写入的数据进行统计
                            if 'accept' in processed_data['paper_decision'].lower():
                                post_counts['Accept'] += 1
                            elif 'reject' in processed_data['paper_decision'].lower():
                                post_counts['Reject'] += 1

                        except Exception as e:
                            paper_id = paper_data.get('id', 'N/A')
                            print(f"处理论文时出错 (Paper ID: {paper_id}): {e}")

            # --- 步骤 3: 打印最终的统计报告 ---
            print("\n================== 处理完成 ==================")
            print(f"共读取 {len(all_papers)} 篇原始论文。")

            print("\n--- 预处理前数据统计 ---")
            print(f"原始 Accept 论文数量: {pre_counts['Accept']}")
            print(f"原始 Reject 论文数量: {pre_counts['Reject']}")
            print(f"原始 Withdraw 论文数量: {pre_counts['Withdraw']}")
            print(f"原始 Desk Reject 论文数量: {pre_counts['Desk Reject']}")
            print(f"其他或待定状态数量: {pre_counts['Other/Pending']}")

            print("\n--- 数据处理与过滤总结 ---")
            print(f"因 [Desk Reject] 被丢弃: {count_dropped_desk_reject} 条")
            print(f"因 [Withdrawal 且无Review] 被丢弃: {count_dropped_withdraw_no_review} 条")
            print(f"因 [Cross-Reply] 被丢弃: {count_dropped_cross_reply} 条")

            print("\n--- 预处理后数据统计 ---")
            print(f"最终被标记为 Accept 并写入文件的论文数量: {post_counts['Accept']}")
            print(f"最终被标记为 Reject 并写入文件的论文数量: {post_counts['Reject']}")

            print("\n----------------------------------------------")
            print(f"总计 {count_written} 条数据已成功保存到: {output_file}")
            print("==============================================")

def main():
    processor = OpenReviewProcessor()
    input_filename = '../../Datasets/raw_dataset/ICLR.cc_2024.jsonl'
    output_filename = '../../Datasets/formatted_dataset/ICLR.cc_2024_formatted.jsonl'

    print(f"开始处理文件: {input_filename}")

    processor.process_dataset(
        input_file=input_filename,
        output_file=output_filename
    )

if __name__ == "__main__":
    main()