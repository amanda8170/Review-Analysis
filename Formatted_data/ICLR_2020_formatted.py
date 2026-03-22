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
            'paper_decision': self.extract_decision(data)
        }
        return paper_info

    def extract_decision(self, data: Dict) -> str:  # <-- 加上 self
        """从数据中提取论文决定"""
        reviews = data.get('reviews', [])

        for review in reviews:
            content = review.get('content', {})
            if 'decision' in content:
                return content['decision']
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
            return content.get('review', '')

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
        rating_str = content.get('rating', '')
        if rating_str and ':' in rating_str:
            try:
                return rating_str.split(':')[0].strip()
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
        处理整个数据集:
        1. 读取所有数据.
        2. 过滤掉 'Withdrawal' 和 'Desk Reject' 的论文.
        3. 分类剩下的论文，丢弃 'cross_reply' 类.
        4. 将 'normal' 和 'self_reply' 合并写入同一个文件.
        """
        # 步骤 1: 读取所有论文数据
        print("正在读取所有论文...")
        all_papers = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        all_papers.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"警告: 跳过一行无法解析的JSON数据。")
        print(f"读取完成，共 {len(all_papers)} 篇论文。")

        # 步骤 2: 过滤 'Withdrawal' 和 'Desk Reject'
        print("正在过滤论文...")
        filtered_papers = self.filter_papers(all_papers)
        initial_count = len(all_papers)
        filtered_count = len(filtered_papers)
        print(f"过滤完成，移除了 {initial_count - filtered_count} 篇论文，剩余 {filtered_count} 篇。")

        # 步骤 3: 遍历、分类并写入同一个文件
        print(f"正在分类论文并写入到合并文件: {output_file} (将丢弃 cross_reply)...")
        # 只打开一个输出文件
        with jsonlines.open(output_file, 'w') as writer:

            # 更新计数器
            count_written = 0
            count_dropped_cross_reply = 0

            for paper_data in filtered_papers:
                try:
                    # 获取分类
                    raw_reviews = [review for review in paper_data.get('reviews', [])
                                   if not self._get_signature(review).endswith('/Public_Comment')]
                    category = self.classify_paper(raw_reviews)

                    # 如果是 'normal' 或 'self_reply'，就处理并写入
                    if category == 'normal' or category == 'self_reply':
                        processed_data = self.process_single_paper(paper_data)
                        writer.write(processed_data)
                        count_written += 1
                    # 如果是 'cross_reply'，则计数并跳过
                    elif category == 'cross_reply':
                        count_dropped_cross_reply += 1
                        continue

                except Exception as e:
                    paper_id = paper_data.get('id', 'N/A')
                    print(f"处理论文时出错 (Paper ID: {paper_id}): {e}")

        print("\n处理完成!")
        print(f"共 {count_written} 条数据已保存到: {output_file}")
        print(f"被丢弃的 Cross-Reply 数据: {count_dropped_cross_reply} 条。")

def main():
    processor = OpenReviewProcessor()
    input_filename = '../../Datasets/raw_dataset/ICLR.cc_2020.jsonl'
        # 只需要一个输出文件名
    output_filename = '../../Datasets/formatted_dataset/ICLR.cc_2020_formatted.jsonl'

    print(f"开始处理文件: {input_filename}")

        # 调用最终版的 process_dataset 函数
    processor.process_dataset(
        input_file=input_filename,
        output_file=output_filename  # 传入合并后的文件名
        )

if __name__ == "__main__":
    main()