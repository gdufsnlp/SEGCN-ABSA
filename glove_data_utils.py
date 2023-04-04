# -*- coding: UTF-8 -*-

import os
import pickle
import numpy as np
import spacy
import torch
from spacy.tokens import Doc
from tqdm import tqdm


def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = './preprocess/{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:  # 创建embedding
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        fname = './preprocess/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


def opinion_lexicon():
    pos_file = 'lexicon/positive-words.txt'
    neg_file = 'lexicon/negative-words.txt'
    lexicon = []
    fin1 = open(pos_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fin2 = open(neg_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines1 = fin1.readlines()
    lines2 = fin2.readlines()
    fin1.close()
    fin2.close()
    for pos_word in lines1:
        lexicon.append(pos_word.strip())
    for neg_word in lines2:
        lexicon.append(neg_word.strip())
    return lexicon


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


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:  # 添加pad unk
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

    def opinion_in_text(self, text, lexicon):
        words = text.split()
        all_tag = []
        for idx, word in enumerate(words):
            if word in lexicon:
                all_tag.append(1)
            else:
                all_tag.append(0)
        return all_tag

    def position_id(self, max_len):
        position_ids = {}
        position = (max_len - 1) * -1
        position_id = 1
        while position <= max_len - 1:
            position_ids[position] = position_id
            position_id += 1
            position += 1
        return position_ids

    def position_indexing(self, text_left, aspect, text_right, position_dict):
        left_indices = list(range(-len(text_left.split()), 0))
        aspect_indices = [0] * len(aspect.split())
        right_indices = list(range(1, len(text_right.split()) + 1))
        position = left_indices + aspect_indices + right_indices
        for i in range(len(position)):
            position[i] = position_dict[position[i]]
        return position


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        text_len = []
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text_len.append(len(text_raw.split()))
                text += text_raw + " "
        return text

    @staticmethod
    def __text_len__(fnames):
        text_len = []
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text_len.append(len(text_raw.split()))
        return max(text_len)

    @staticmethod
    def __read_data__(fname, model, tokenizer, max_len):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        if model == 'segcn_transfer':
            model = 'segcn'
        print("loading dataset {0} graph: {1}".format(model, fname + '.' + model + '.graph'))
        fin = open(fname + '.' + model + '.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()

        all_data = []
        lexicon = opinion_lexicon()
        for i in tqdm(range(0, len(lines), 3)):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)
            opinion_indices = tokenizer.opinion_in_text(text_left + " " + aspect + " " + text_right, lexicon)
            position_dict = tokenizer.position_id(max_len)
            position_indices = tokenizer.position_indexing(text_left, aspect, text_right, position_dict)
            polarity = int(polarity) + 1
            mask = opinion_indices
            dependency_graph = idx2graph[i]

            data = {
                'text_indices': text_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'position_indices': position_indices,
                'opinion_indices': opinion_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'mask': mask,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', model='segcn', embed_dim=300):
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },
        }
        print("preparing {0} dataset ...".format(dataset))
        print("reading {0} dataset path: \n{1}\n{2}".format(dataset, fname[dataset]['train'], fname[dataset]['test']))
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        max_len = ABSADatesetReader.__text_len__([fname[dataset]['train'], fname[dataset]['test']])
        if not os.path.exists('preprocess'):
            os.makedirs('preprocess')
        if os.path.exists('./preprocess/' + dataset + '_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open('./preprocess/' + dataset + '_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open('./preprocess/' + dataset + '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], model, tokenizer, max_len))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], model, tokenizer, max_len))
