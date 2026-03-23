import json
import spacy
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import warnings
import time
from tqdm import tqdm

# 忽略spaCy的警告信息
warnings.filterwarnings("ignore", category=UserWarning)


class ReviewAspectScorer:
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)
        print(f"Successfully loaded spaCy model: {model_name}")

        self.aspects = {
            # 1. 论文结构与逻辑 (Paper Structure & Logic)
            "title_quality": ["heading", "title"],
            "abstract_completeness": ["abstract", "overview", "summary", "synopsis"],
            "abstract_clarity": ["abstract clarity", "summary clarity"],
            "introduction_background": ["background", "context", "introduction"],
            "literature_review": ["literature review", "prior art", "prior work", "related work", "state-of-the-art"],
            "problem_motivation": ["gap", "motivation", "problem statement", "rationale"],
            "contribution_statement": ["contribution", "contributions", "innovation", "innovations", "novelty"],
            "logical_flow": ["argument", "coherence", "flow", "narrative", "organization", "structure"],
            "conclusion_quality": ["conclusion", "final remarks", "summary"],

            # 2. 方法论与技术内容 (Methodology & Technical Content)
            "theoretical_foundation": ["formulation", "theoretical foundation", "theory"],
            "methodology_choice": ["algorithm", "approach", "architecture", "framework", "method", "methodology",
                                   "model", "technique"],
            "experimental_design": ["evaluation plan", "evaluation setup", "experiment setup", "experimental design",
                                    "procedure", "protocol", "setup"],
            "data_quality": ["benchmark", "corpus", "corpora", "data collection", "data quality", "dataset"],
            "statistical_analysis": ["statistical analysis", "statistics"],
            "reproducibility": ["artifact", "code availability", "replicability", "replicate", "reproducibility",
                                "reproduce"],

            # 3. 结果与评估 (Results & Evaluation)
            "results_presentation": ["evaluation results", "experiments", "findings", "outcomes", "results"],
            "figure_table_quality": ["chart", "diagram", "figure", "graph", "illustration", "plot", "table",
                                     "visualization"],
            "result_interpretation": ["analysis", "discussion", "discussion of results", "insight", "interpretation"],
            "baseline_comparison": ["baseline", "baselines", "benchmark", "comparison", "comparisons",
                                    "state-of-the-art comparison"],
            "evaluation_metrics": ["evaluation", "measure", "measures", "metric", "metrics", "performance metric"],
            "ablation_study": ["ablation", "ablation study", "component analysis"],

            # 4. 创新性与贡献 (Innovation & Contribution)
            "novelty_originality": ["inventiveness", "novelty", "originality"],
            "technical_soundness": ["correctness", "rigor", "soundness", "validity"],
            "practical_significance": ["application", "impact", "practical significance", "real-world application",
                                       "usefulness", "utility"],
            "theoretical_contribution": ["theoretical advance", "theoretical contribution"],

            # 5. 写作与呈现 (Writing & Presentation)
            "writing_quality": ["clarity", "grammar", "language", "presentation", "prose", "readability", "style",
                                "writing"],
            "technical_language": ["jargon", "technical terms", "terminology"],
            "citation_references": ["bibliography", "citation", "citations", "reference", "references"],

            # 6. 技术细节 (Technical Details)
            "implementation_details": ["code", "hyperparameters", "implementation", "parameters", "technical details"],
            "computational_complexity": ["complexity", "computational cost", "efficiency", "performance", "runtime",
                                         "speed"],
            "scalability": ["large-scale", "scale", "scalability", "scaling"],

            # 7. 实验评估 (Experimental Evaluation)
            "experimental_thoroughness": ["coverage", "depth", "extensiveness", "scope"],
            "dataset_appropriateness": ["data appropriateness", "data selection", "dataset choice",
                                        "dataset suitability"],
            "comparison_fairness": ["evaluation fairness", "fair comparison", "unbiased comparison"],
        }

        self.sentiment_words = {
            # --- 正面词 (值为 1) ---
            "excellent": 1, "outstanding": 1, "impressive": 1, "thorough": 1,
            "robust": 1, "comprehensive": 1, "significant": 1, "innovative": 1,
            "remarkable": 1, "exceptional": 1, "superb": 1, "groundbreaking": 1,
            "well-written": 1, "well-organized": 1, "well-motivated": 1,
            "good": 1, "clear": 1, "solid": 1, "effective": 1, "adequate": 1,
            "appropriate": 1, "convincing": 1, "detailed": 1, "sound": 1,
            "valid": 1, "useful": 1, "important": 1, "promising": 1,

            # --- 负面词 (值为 -1) ---
            "poor": -1, "inadequate": -1, "flawed": -1, "incorrect": -1,
            "insufficient": -1, "problematic": -1, "weak": -1, "unclear": -1,
            "terrible": -1, "awful": -1, "unconvincing": -1, "unjustified": -1,
            "questionable": -1, "disappointing": -1, "invalid": -1,
            "limited": -1, "missing": -1, "confusing": -1, "superficial": -1,
            "incomplete": -1, "biased": -1, "vague": -1, "ambiguous": -1,
            "marginal": -1, "trivial": -1
        }

        self.negation_words = {
            "not", "no", "never", "without", "lack", "cannot", "fail",
            "hardly", "rarely", "barely", "isn't", "doesn't", "won't"
        }

        self._preprocess_aspects()

    def _preprocess_aspects(self):
        """预处理方面关键词，获得lemma形式"""
        self.processed_aspects = {}
        for aspect, keywords in self.aspects.items():
            processed = []
            for keyword in keywords:
                doc = self.nlp(keyword)
                lemmas = [token.lemma_.lower() for token in doc if not token.is_stop]
                if lemmas:
                    processed.append(lemmas)
            self.processed_aspects[aspect] = processed

    def _match_sequence(self, text_tokens: List[str], pattern: List[str]) -> bool:
        """匹配token序列"""
        pattern_len = len(pattern)
        if pattern_len == 0: return False
        for i in range(len(text_tokens) - pattern_len + 1):
            if text_tokens[i:i + pattern_len] == pattern:
                return True
        return False

    def _analyze_sentiment_from_doc(self, doc, context_window: int = 3) -> tuple[int, int]:
        """（内部、高效）从spacy doc对象分析情感"""
        pos_count = 0
        neg_count = 0
        for i, token in enumerate(doc):
            word = token.lemma_.lower()
            if word in self.sentiment_words:
                weight = self.sentiment_words[word]
                is_negated = self._check_negation(doc, i, context_window)
                if is_negated:
                    weight = -weight
                if weight > 0:
                    pos_count += 1
                else:
                    neg_count += 1
        return pos_count, neg_count

    def _check_negation(self, doc, target_idx: int, window: int) -> bool:
        """（内部）检查否定"""
        target_token = doc[target_idx]
        if target_token.dep_ == "neg": return True
        for child in target_token.children:
            if child.dep_ == "neg": return True
        start = max(0, target_idx - window)
        for i in range(start, target_idx):
            if doc[i].text.lower() in self.negation_words:
                return True
        return False

    def _extract_aspects_from_doc(self, doc) -> Set[str]:
        """（内部、高效）从spacy doc对象提取方面"""
        text_lemmas = [token.lemma_.lower() for token in doc if not token.is_stop]
        text_lemma_set = set(text_lemmas)
        found_aspects = set()
        for aspect, keyword_lists in self.processed_aspects.items():
            for keyword_lemmas in keyword_lists:
                if len(keyword_lemmas) == 1:
                    if keyword_lemmas[0] in text_lemma_set:
                        found_aspects.add(aspect)
                        break
                elif self._match_sequence(text_lemmas, keyword_lemmas):
                    found_aspects.add(aspect)
                    break
        return found_aspects

    def _analyze_aspect_sentiment_from_doc(self, doc, aspect: str) -> tuple[int, int]:
        """（内部、高效）从spacy doc对象分析特定方面的情感"""
        relevant_sentences_docs = []
        aspect_keywords = self.processed_aspects.get(aspect, [])
        if not aspect_keywords: return 0, 0
        for sent in doc.sents:
            sent_lemmas = [token.lemma_.lower() for token in sent if not token.is_stop]
            for keyword_lemmas in aspect_keywords:
                if len(keyword_lemmas) == 1:
                    if keyword_lemmas[0] in sent_lemmas:
                        relevant_sentences_docs.append(sent)
                        break
                elif self._match_sequence(sent_lemmas, keyword_lemmas):
                    relevant_sentences_docs.append(sent)
                    break

        # 如果找不到相关句子，则分析整个文档对该方面的情感
        if not relevant_sentences_docs:
            return self._analyze_sentiment_from_doc(doc)

        total_pos, total_neg = 0, 0
        for sent_doc in relevant_sentences_docs:
            pos, neg = self._analyze_sentiment_from_doc(sent_doc)
            total_pos += pos
            total_neg += neg
        return total_pos, total_neg

    def process_review(self, dialogue: List[Dict]) -> Dict[str, Tuple[int, int]]:
        """（高效）处理单个评审的所有对话"""
        review_content = " ".join(
            entry.get("content", "")
            for entry in dialogue
            if entry.get("review_type") == "reviewer_review" and entry.get("content")
        )
        if not review_content.strip(): return {}

        # 核心优化：只调用一次nlp()
        doc = self.nlp(review_content)

        mentioned_aspects = self._extract_aspects_from_doc(doc)
        aspect_scores = {}
        for aspect in mentioned_aspects:
            pos, neg = self._analyze_aspect_sentiment_from_doc(doc, aspect)
            if pos > 0 or neg > 0:
                aspect_scores[aspect] = (pos, neg)
        return aspect_scores

    def process_paper(self, data: Dict) -> Dict:
        """处理单篇论文，输出与旧版报告兼容的数据结构"""
        reviewer_aspect_scores = {}
        for review in data.get("reviews", []):
            reviewer_id = review.get("reviewer", "")
            dialogue = review.get("dialogue", [])
            if reviewer_id and dialogue:
                scores = self.process_review(dialogue)
                if scores:
                    reviewer_aspect_scores[reviewer_id] = scores

        all_reviewers = list(reviewer_aspect_scores.keys())
        total_reviewers = len(all_reviewers)

        if total_reviewers == 0:
            return {
                'paper_title': data.get('paper_title', ''),
                'paper_decision': data.get('paper_decision', ''),
                'aspect_average_scores': {},
                'summary': {'participating_reviewers': 0}
            }

        all_mentioned_aspects = set()
        for scores_dict in reviewer_aspect_scores.values():
            all_mentioned_aspects.update(scores_dict.keys())

        final_scores_padded = defaultdict(list)
        for aspect in all_mentioned_aspects:
            for reviewer in all_reviewers:
                pos, neg = reviewer_aspect_scores.get(reviewer, {}).get(aspect, (0, 0))
                score = 1 if pos > neg else -1 if neg > pos else 0
                final_scores_padded[aspect].append(score)

        average_scores = {}
        for aspect, scores in final_scores_padded.items():
            avg = sum(scores) / total_reviewers
            average_scores[aspect] = {
                'average_score': round(avg, 3),
                'reviewer_count_mentioning': len([s for s in scores if s != 0]),
                'total_reviewers': total_reviewers,
                'individual_scores': scores
            }

        return {
            'paper_title': data.get('paper_title', ''),
            'paper_decision': data.get('paper_decision', ''),
            'aspect_average_scores': average_scores,
            'summary': {
                'total_aspects_mentioned': len(all_mentioned_aspects),
                'participating_reviewers': total_reviewers
            }
        }

    def process_file(self, file_path: str) -> List[Dict]:
        """处理JSONL文件"""
        results = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
        except Exception:
            total_lines = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            progress_bar = tqdm(f, total=total_lines, desc=f"Processing {file_path.split('/')[-1]}", unit="papers")
            for line in progress_bar:
                try:
                    data = json.loads(line.strip())
                    result = self.process_paper(data)
                    results.append(result)
                except (json.JSONDecodeError, Exception) as e:
                    progress_bar.write(f"Error processing a line in {file_path}: {e}")
                    continue
        return results

    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """生成与旧版格式兼容的汇总报告"""
        aspect_score_sums = defaultdict(list)
        aspect_paper_count = defaultdict(int)

        for result in results:
            for aspect, score_info in result['aspect_average_scores'].items():
                aspect_paper_count[aspect] += 1
                aspect_score_sums[aspect].extend(score_info['individual_scores'])

        overall_stats = {}
        for aspect, scores in aspect_score_sums.items():
            if scores:
                total_ratings = len(scores)
                overall_stats[aspect] = {
                    'average_score': round(sum(scores) / total_ratings, 3),
                    'positive_ratio': round(sum(1 for s in scores if s > 0) / total_ratings, 3),
                    'negative_ratio': round(sum(1 for s in scores if s < 0) / total_ratings, 3),
                    'paper_count': aspect_paper_count[aspect],
                    'total_ratings': total_ratings
                }

        all_aspects_sorted = sorted(
            overall_stats.items(),
            key=lambda item: item[1]['average_score'],
            reverse=True
        )

        return {
            'total_papers': len(results),
            'all_aspects_sorted': all_aspects_sorted
        }


def print_summary_report(summary: Dict, report_title: str, output_file=None):
    """打印与旧版格式兼容的报告"""

    def write(text):
        if output_file:
            output_file.write(text + '\n')
        else:
            print(text)

    write("\n" + "=" * 110)
    write(f"Overall Aspect Analysis Report for: {report_title}")
    write("=" * 110)
    write(f"Total papers analyzed: {summary['total_papers']}")
    write("\n--- All Aspects Sorted by Average Score ---")
    write(
        f"{'Aspect':<30} | {'Avg Score':<12} | {'Positive Ratio':<15} | {'Negative Ratio':<15} | {'Paper Count':<15} | {'Total Ratings':<15}")
    write("-" * 110)
    for aspect, stats in summary['all_aspects_sorted']:
        avg_score_str = f"{stats['average_score']:.3f}"
        pos_ratio_str = f"{stats['positive_ratio']:.1%}"
        neg_ratio_str = f"{stats['negative_ratio']:.1%}"
        paper_count_str = f"{stats['paper_count']}"
        total_ratings_str = f"{stats['total_ratings']}"
        write(
            f"{aspect:<30} | {avg_score_str:<12} | {pos_ratio_str:<15} | {neg_ratio_str:<15} | {paper_count_str:<15} | {total_ratings_str:<15}")
    write("-" * 110)


def main():
    start_time = time.time()

    input_files_map = {
        "Hard Samples": "../../../Datasets/hard example/hard_sample.jsonl",
        "Simple Accept": "../../../Datasets/hard example/simple_accept_sample.jsonl",
        "Simple Reject": "../../../Datasets/hard example/simple_reject_sample.jsonl",
        "Hard Accept": "../../../Datasets/hard example/hard_ac.jsonl",
        "Hard Reject": "../../../Datasets/hard example/hard_rej.jsonl"
    }
    output_file_path = "summary_report.txt"

    scorer = ReviewAspectScorer()

    all_results_by_file = {}
    print("Starting analysis...")

    for category, file_path in input_files_map.items():
        print(f"\nNow processing file: {category} ({file_path})...")
        try:
            results_for_file = scorer.process_file(file_path)
            all_results_by_file[file_path] = results_for_file
        except FileNotFoundError:
            print(f"\nError: Input file not found at {file_path}")
        except Exception as e:
            print(f"\nAn unexpected error occurred while processing {file_path}: {e}")

    if not all_results_by_file:
        print("\nNo papers were processed successfully. Exiting.")
        return

    try:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"\nReports will be saved to: {output_file_path}")

            # 生成合并的摘要报告
            print("\nGenerating combined summary report...")
            all_results = [item for sublist in all_results_by_file.values() for item in sublist]
            if all_results:
                combined_summary = scorer.generate_summary_report(all_results)
                print_summary_report(combined_summary, "Combined Report from All Files", output_file=output_file)

            # 生成每个文件的独立报告
            print("\nGenerating individual file reports...")
            for file_path, results_for_file in all_results_by_file.items():
                if not results_for_file:
                    continue
                file_name = file_path.split('/')[-1].split('\\')[-1]
                individual_summary = scorer.generate_summary_report(results_for_file)
                print_summary_report(individual_summary, f"Report for: {file_name}", output_file=output_file)

            print(f"\nAll reports have been successfully saved.")

    except IOError as e:
        print(f"\nError: Could not open output file {output_file_path}: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal analysis finished in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")


if __name__ == "__main__":
    main()