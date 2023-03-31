# -*- coding: UTF-8 -*-

import numpy as np
import spacy
import pickle
from collections import defaultdict

from spacy.tokens import Doc
from tqdm import tqdm
from transformers import BertTokenizer


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_trf')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


class Tokenizer4Bert:
    def __init__(self, pretrained_bert_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


def get_lexicon():
    pos_file = 'lexicon/positive-words.txt'
    neg_file = 'lexicon/negative-words.txt'
    lexicon = defaultdict(lambda: 'NEU')
    fin1 = open(pos_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fin2 = open(neg_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines1 = fin1.readlines()
    lines2 = fin2.readlines()
    fin1.close()
    fin2.close()
    for pos_word in lines1:
        lexicon[pos_word.strip()] = 'POS'
    for neg_word in lines2:
        lexicon[neg_word.strip()] = 'NEG'

    return lexicon


def position_weight_matrix(aspect_id, tokens):
    length = len(tokens) + 2
    position_weight = [0, 0]
    for i in range(aspect_id[0]):
        position_weight.append(1 - (aspect_id[0] - i) / len(tokens))
    for i in range(aspect_id[0], aspect_id[1] + 1):
        position_weight.append(1)
    for i in range(aspect_id[1] + 1, len(tokens)):
        position_weight.append(1 - (i - aspect_id[1]) / len(tokens))
    matrix = []

    for i in range(length):
        temp = position_weight[:]
        temp[i] = 1
        matrix.append(temp)

    return matrix


def node_weight_dependency_adj_matrix(text, lexicon, aspect_id):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words) + 2, len(words) + 2)).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        if lexicon[token.text] == 'POS':
            matrix[0][token.i + 2] = 1
            matrix[token.i + 2][0] = 1
        elif lexicon[token.text] == 'NEG':
            matrix[1][token.i + 2] = 1
            matrix[token.i + 2][1] = 1
        matrix[0][0] = 1
        matrix[1][1] = 1
        matrix[token.i + 2][token.i + 2] = 1  # self-loop
        for child in token.children:
            matrix[token.i + 2][child.i + 2] = 1
            matrix[child.i + 2][token.i + 2] = 1

    position_weight = position_weight_matrix(aspect_id, tokens)

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if j == i:
                continue
            if matrix[i][j] == 1:
                matrix[i][j] = 1 + position_weight[i][j]
    return matrix


def adj4bert(ori_adj, tokenizer, text_left, aspect, text_right, sent=True):
    left_tokens, term_tokens, right_tokens = [], [], []
    left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []
    if sent:
        sent_tokens = ['POS', 'NEG']
        sent_tok2ori_map = [0, 1]
        offset = 2
    else:
        sent_tokens = []
        sent_tok2ori_map = []
        offset = 0

    for ori_i, w in enumerate(text_left):
        for t in tokenizer.tokenize(w):
            left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
            left_tok2ori_map.append(ori_i + offset)  # * [0, 0, 1, 2, 2]
    offset += len(text_left)
    for ori_i, w in enumerate(aspect):
        for t in tokenizer.tokenize(w):
            term_tokens.append(t)
            term_tok2ori_map.append(ori_i + offset)
    offset += len(aspect)
    for ori_i, w in enumerate(text_right):
        for t in tokenizer.tokenize(w):
            right_tokens.append(t)
            right_tok2ori_map.append(ori_i + offset)

    bert_tokens = sent_tokens + left_tokens + term_tokens + right_tokens
    tok2ori_map = sent_tok2ori_map + left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
    truncate_tok_len = len(bert_tokens)
    tok_adj = np.zeros(
        (truncate_tok_len, truncate_tok_len), dtype='float32')
    for i in range(truncate_tok_len):
        for j in range(truncate_tok_len):
            tok_adj[i][j] = ori_adj[tok2ori_map[i]][tok2ori_map[j]]
    return tok_adj


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph1 = {}
    idx2graph2 = {}
    fout1 = open(filename + '.segcn.graph', 'wb')
    fout2 = open(filename + '.segcn_bert.graph', 'wb')

    sentiment_lexicon = get_lexicon()
    tokenizer = Tokenizer4Bert()
    for i in tqdm(range(0, len(lines), 3)):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = text_left + ' ' + aspect + ' ' + text_right
        aspect_double_id = []
        aspect_double_id.append(len(text_left.split()))
        aspect_double_id.append(len(text_left.split()) + len(aspect.split()) - 1)
        SPGCN_adj_matrix = node_weight_dependency_adj_matrix(text, sentiment_lexicon, aspect_double_id)
        SPGCN_BERT_adj_matrix = adj4bert(SPGCN_adj_matrix, tokenizer, text_left.split(), aspect.split(), text_right.split(), sent=True)
        idx2graph1[i] = SPGCN_adj_matrix
        idx2graph2[i] = SPGCN_BERT_adj_matrix
    pickle.dump(idx2graph1, fout1)
    pickle.dump(idx2graph2, fout2)
    fout1.close()
    fout2.close()
    print(filename + " done!")


if __name__ == '__main__':
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')
