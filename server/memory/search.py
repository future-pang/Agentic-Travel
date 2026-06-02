import math
from typing import List, Dict, Any, Tuple

def tokenize_chinese(text: str) -> List[str]:
    """
    轻量级中英文分词（单字级 + 中文双字 Bigram），无需外部依赖（如 jieba）。
    非常适合峨眉山文旅场景中的景点名、特殊词汇（如雷洞坪、猴区、索道、恐高）的极速匹配。
    """
    if not text:
        return []
    
    text = text.lower()
    tokens = []
    
    # 1. 提取单字和连续英文字词/数字词作为基本 token
    current_word = []
    for char in text:
        if char.isalnum():
            if '\u4e00' <= char <= '\u9fff':
                # 中文字符：先结算并清理前面的英文字词
                if current_word:
                    tokens.append("".join(current_word))
                    current_word = []
                tokens.append(char)
            else:
                # 英文或数字字符：暂存
                current_word.append(char)
        else:
            # 标点符号或空格：结算英文字词
            if current_word:
                tokens.append("".join(current_word))
                current_word = []
                
    if current_word:
        tokens.append("".join(current_word))
        
    # 2. 生成相邻中文字符的双字 Bigram 词项，以大幅提高词组/景点名称的精准匹配（如 "金顶" -> ["金", "顶", "金顶"]）
    for i in range(len(text) - 1):
        c1, c2 = text[i], text[i+1]
        if '\u4e00' <= c1 <= '\u9fff' and '\u4e00' <= c2 <= '\u9fff':
            tokens.append(c1 + c2)
            
    return tokens

class BM25:
    """
    纯 Python 实现的轻量级 BM25 文本检索算法，用于旅程记忆的关键词检索。
    """
    def __init__(self, docs: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.doc_lens = []
        self.doc_freqs = []  # List[Dict[str, int]]
        self.df = {}  # Dict[str, int]
        
        for doc in docs:
            # 整合描述和正文进行联合匹配
            text = f"{doc.get('description', '')} {doc.get('content', '')}"
            tokens = tokenize_chinese(text)
            self.doc_lens.append(len(tokens))
            
            # 计算当前文档的词频 TF
            freqs = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            
            # 统计词项在多少个文档中出现 DF
            for token in freqs:
                self.df[token] = self.df.get(token, 0) + 1
                
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0

    def get_idf(self, word: str) -> float:
        """计算词项的逆文档频率 IDF。"""
        df_w = self.df.get(word, 0)
        # 使用平滑 BM25 IDF 表达式，防止词项全匹配时 IDF 产生负值
        return math.log((self.N - df_w + 0.5) / (df_w + 0.5) + 1.0)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算单个文档与查询的 BM25 相关性分数。"""
        score = 0.0
        doc_len = self.doc_lens[doc_idx]
        freqs = self.doc_freqs[doc_idx]
        
        for token in query_tokens:
            if token not in freqs:
                continue
            tf = freqs[token]
            idf = self.get_idf(token)
            
            # BM25 计算公式主项
            denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (tf * (self.k1 + 1.0)) / denom
            
        return score

    def rank(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        """根据查询内容对所有记忆进行打分和排序。"""
        query_tokens = tokenize_chinese(query)
        if not query_tokens or self.N == 0:
            return []
            
        scored_docs = []
        for i, doc in enumerate(self.docs):
            s = self.score(query_tokens, i)
            if s > 0.0:  # 仅保留至少有一个词匹配成功的记忆
                scored_docs.append((doc, s))
                
        # 按相关性分数降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
