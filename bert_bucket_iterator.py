# -*- coding: UTF-8 -*-

import math
import random
import torch
import numpy as np


class BucketIterator_BERT(object):
    def __init__(self, data, batch_size, model, sort_key='text_aspect_bert_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size, model)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size, model):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size], model))
        return batches

    def pad_data(self, batch_data, model):
        batch_text_aspect_bert_indices = []
        batch_text_aspect_segments_indices = []
        batch_text_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_dependency_graph = []
        batch_polarity = []
        batch_opinion_indices = []
        batch_mask = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_aspect_bert_indices, text_aspect_segments_indices, \
            text_indices, aspect_indices, left_indices, dependency_graph, \
            polarity, opinion_indices, mask = \
                item['text_aspect_bert_indices'], item['text_aspect_segments_indices'], \
                item['text_indices'], item['aspect_indices'], item['left_indices'], item['dependency_graph'], \
                item['polarity'], item['opinion_indices'], item['mask']

            text_aspect_bert_indices_padding = [0] * (max_len - len(text_aspect_bert_indices))
            text_aspect_segments_indices_padding = [0] * (max_len - len(text_aspect_segments_indices))
            text_padding = [0] * (max_len - len(text_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            opinion_padding = [0] * (max_len - len(opinion_indices))

            batch_text_aspect_bert_indices.append(text_aspect_bert_indices + text_aspect_bert_indices_padding)
            batch_text_aspect_segments_indices.append(text_aspect_segments_indices + text_aspect_segments_indices_padding)
            batch_text_indices.append(text_indices + text_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)

            batch_dependency_graph.append(np.pad(dependency_graph, ((0, max_len - len(text_indices) - 1), (0, max_len - len(text_indices) - 1)), 'constant'))
            batch_opinion_indices.append(opinion_indices + opinion_padding)
            batch_mask.append(mask + opinion_padding)
        return {
            'text_aspect_bert_indices': torch.tensor(batch_text_aspect_bert_indices),
            'text_aspect_segments_indices': torch.tensor(batch_text_aspect_segments_indices),
            'text_indices': torch.tensor(batch_text_indices),
            'aspect_indices': torch.tensor(batch_aspect_indices),
            'left_indices': torch.tensor(batch_left_indices),
            'opinion_indices': torch.tensor(batch_opinion_indices),
            'polarity': torch.tensor(batch_polarity),
            'dependency_graph': torch.tensor(np.array(batch_dependency_graph)),
            'mask': torch.BoolTensor(batch_mask),
        }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
