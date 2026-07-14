

import math
from collections import Counter

class BM25:
    def __init__(self, corpus, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)

        self.tokenized_corpus = []
        self.doc_len = []
        self.term_freqs = []          # one Counter per doc
        self.doc_freq = Counter()     # word -> number of docs containing it

        for doc in corpus:
            tokens = self._tokenize(doc)
            self.tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))
            self.term_freqs.append(Counter(tokens))
            self.doc_freq.update(set(tokens))   # set() = count docs, not occurrences

        self.avgdl = sum(self.doc_len) / self.N

    def _tokenize(self, text):
        return [t for t in text.lower().split() if (t == t.strip(".,;:!?()\"'")) ]


    def score(self, query, doc_idx):
        tokens = self._tokenize(query)

        bracket = 1 - self.b + self.b * (self.doc_len[doc_idx] / self.avgdl)

        total = 0.0
        for token in tokens:
            tf = self.term_freqs[doc_idx][token]
            if tf == 0:
                continue

            df = self.doc_freq[token]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

            tf_part = (tf * (self.k1 + 1)) / (tf + self.k1 * bracket)

            total += idf * tf_part

        return total
    def search(self, query, k=5):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:k]
        
     

toy_corpus = [
    "apple revenue grew and apple revenue hit record apple highs",   # d0: "apple" 3x -> saturation test
    "revenue across the industry declined this year",                # d1
    "the company launched a new phone last week",                    # d2: shares no words with money queries
    "AAPL reported strong quarterly results",                        # d3: rare token, case test
    "revenue growth and revenue expansion drove revenue gains across every segment of the business this year", # d4: long doc -> length penalty test
    "banks reported declining results",                              # d5: short doc -> length bonus test
]   
bm = BM25(toy_corpus)
print(bm.search("apple revenue"))
print(bm.search("AAPL results"))
print(bm.search("phone"))                     # expect exactly 0.0