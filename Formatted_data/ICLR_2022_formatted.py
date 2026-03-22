import json
import jsonlines
from typing import Dict, List, Any, Set
from collections import defaultdict
from datetime import datetime


class OpenReviewProcessor:
    def _get_signature(self, review: Dict) -> str:
        """安全地获取第一个签名"""
        signatures = review.get('signatures')
        return signatures[0] if signatures and isinstance(signatures, list) and len(signatures) > 0 else "Unknown"

    def extract_paper_info(self, data: Dict) -> Dict:
        """提取论文基本信息"""
        content = data.get('content', {})
        original_track = content.get('Please_choose_the_closest_area_that_your_submission_falls_into', '')
        paper_info = {
            'paper_title': content.get('title', ''),
            'paper_authors': content.get('authors', []),
            'paper_abstract': content.get('abstract', ''),
            'paper_keywords': content.get('keywords', []),
            'paper_tldr': content.get('TL;DR', ''),
            'paper_track': original_track.replace('_', ' ') if original_track else '',
            'paper_venue': content.get('venue', ''),
            'paper_decision': self.get_final_decision(data)
        }
        return paper_info

    def get_final_decision(self, data: Dict) -> str:
        original_decision = self.extract_decision(data)

        if 'withdraw' in original_decision.lower():
            reviews = data.get('reviews', [])
            has_review_comments = any(
                'Reviewer' in self._get_signature(r)
                for r in reviews
            )

            if has_review_comments:
                return 'Reject'
            else:
                return 'Withdraw'

        return original_decision

    def extract_decision(self, data: Dict) -> str:
        invitation_str = data.get('invitation', '')
        if 'Withdrawn_Submission' in invitation_str:
            return 'Withdraw'
        if 'Desk_Rejected_Submission' in invitation_str:
            return 'Desk Reject'

        # 优先级3: 在 reviews 列表中寻找 decision
        reviews = data.get('reviews', [])
        for review in reviews:
            review_content = review.get('content', {})
            review_invitation = review.get('invitation', '')
            if 'Desk_Reject' in review_invitation:
                return 'Desk Reject'

            if 'decision' in review_content:
                decision = review_content['decision']
                return decision if decision else 'Pending'

        return 'Pending'

    def filter_papers(self, papers_data: List[Dict]) -> List[Dict]:  # <-- 加上 self
        filtered_papers = []

        for paper in papers_data:
            decision = self.extract_decision(paper)
            if decision in ['Accept (Oral)', 'Accept (Spotlight)', 'Accept (Poster)', 'Reject']:
                filtered_papers.append(paper)

        return filtered_papers

    def determine_review_type(self, review: Dict) -> str:
        """确定review类型"""
        content = review.get('content', {})
        signature = self._get_signature(review)
        if 'Authors' in signature:
            return 'author_response'
        else:
            return 'reviewer_review'

    def extract_review_content(self, review: Dict) -> str:
        signature = self._get_signature(review)
        content = review.get('content', {})

        if 'Reviewer' in signature:
            parts = []
            if content.get('summary_of_the_paper'):
                parts.append(f"Summary of the paper: {content['summary_of_the_paper']}")
            if content.get('main_review'):
                parts.append(f"Main review: {content['main_review']}")
            if content.get('summary_of_the_review'):
                parts.append(f"Summary of the review: {content['summary_of_the_review']}")
            return "\n\n".join(parts)

        elif 'Authors' in signature:
            return content.get('comment', '')


        elif 'Program_Chairs' in signature or 'Area_Chair' in signature:
            if 'decision' in content and 'comment' in content:
                return content.get('comment', '')
            else:
                return content.get('comment', '')

        else:
            return content.get('comment', '')

    def extract_rating(self, review: Dict) -> str:
        content = review.get('content', {})
        recommendation = content.get('recommendation', '')
        if recommendation and ':' in recommendation:
            try:
                return recommendation.split(':')[0].strip()
            except:
                return '-1'
        return '-1'

    def extract_confidence(self, review: Dict) -> str:
        content = review.get('content', {})
        confidence_str = content.get('confidence', '')
        if confidence_str and ':' in confidence_str:
            try:
                return confidence_str.split(':')[0].strip()
            except:
                return '-1'
        return '-1'

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
            first_poster_sig = self._get_signature(thread_reviews[0])
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
                sig = self._get_signature(review)
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
            signature = self._get_signature(review)
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
                   if not self._get_signature(review).endswith('/Public_Comment')]
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
        """
        处理整个数据集，并在一次遍历中完成所有统计和过滤。
        """
        # --- 步骤 1: 初始化所有计数器 ---
        pre_counts = defaultdict(int)
        post_counts = defaultdict(int)
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
                # --- a. 预处理统计 (基于原始状态) ---
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
                # 定义此数据集可接受的最终决定
                decisions_to_keep = ['Accept (Oral)', 'Accept (Spotlight)', 'Accept (Poster)', 'Reject']
                if final_decision in decisions_to_keep:
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
    input_filename = '../../Datasets/raw_dataset/ICLR.cc_2022.jsonl'
        # 只需要一个输出文件名
    output_filename = '../../Datasets/formatted_dataset/ICLR.cc_2022_formatted.jsonl'

    print(f"开始处理文件: {input_filename}")

        # 调用最终版的 process_dataset 函数
    processor.process_dataset(
        input_file=input_filename,
        output_file=output_filename  # 传入合并后的文件名
        )

if __name__ == "__main__":
    main()