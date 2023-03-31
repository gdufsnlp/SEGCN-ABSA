# -*- coding: UTF-8 -*-

import numpy as np
import pickle
import spacy
from spacy.tokens import Doc
from tqdm import tqdm

from transformers import BertTokenizer


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


class Tokenizer4Bert:
    def __init__(self, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def text_to_sequence(self, text):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def opinion_in_text(self, text, lexicon):
        opinion_index = []
        for idx, word in enumerate(text.split()):
            for t in self.tokenize(word):
                if word in lexicon:
                    opinion_index.append(1)
                else:
                    opinion_index.append(0)
        return opinion_index


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader_BERT:
    @staticmethod
    def __read_data__(fname, model, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        print("loading dataset {0} graph: {1}".format(model, fname + '.' + model + '.graph'))
        fin = open('./' + fname + '.' + model + '.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()

        all_data = []
        lexicon = opinion_lexicon()
        for i in tqdm(range(0, len(lines), 3)):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            text_len = np.sum(text_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            opinion_indices = tokenizer.opinion_in_text('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + ' [SEP]', lexicon)
            mask = opinion_indices
            polarity = int(polarity) + 1

            text_aspect_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            text_aspect_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            dependency_graph = idx2graph[i]

            data = {
                'text_aspect_bert_indices': text_aspect_bert_indices,
                'text_aspect_segments_indices': text_aspect_segments_indices,
                'text_indices': text_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'opinion_indices': opinion_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'mask': mask,
            }
            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', model='segcn_bert', pretrain=''):
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
        tokenizer = Tokenizer4Bert(pretrain)
        self.train_data = ABSADataset(ABSADatesetReader_BERT.__read_data__(fname[dataset]['train'], model, tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader_BERT.__read_data__(fname[dataset]['test'], model, tokenizer))