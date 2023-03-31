# -*- coding: UTF-8 -*-

import math
import random
import torch
import numpy as np


class BucketIterator(object):
    def __init__(self, data, batch_size, model, sort_key='text_indices', shuffle=True, sort=True):
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
        batch_text_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_position_indices = []
        batch_opinion_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_mask = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_indices, aspect_indices, left_indices, position_indices, opinion_indices, polarity, dependency_graph, mask = \
                item['text_indices'], item['aspect_indices'], item['left_indices'], item['position_indices'], \
                item['opinion_indices'], item['polarity'], item['dependency_graph'], item['mask']

            text_padding = [0] * (max_len - len(text_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            opinion_padding = [0] * (max_len - len(opinion_indices))

            batch_text_indices.append(text_indices + text_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_position_indices.append(position_indices + text_padding)
            batch_opinion_indices.append(opinion_indices + opinion_padding)
            batch_polarity.append(polarity)
            batch_dependency_graph.append(np.pad(dependency_graph, ((0, max_len - len(text_indices)), (0, max_len - len(text_indices))), 'constant'))
            batch_mask.append(mask + opinion_padding)
        return {
            'text_indices': torch.tensor(batch_text_indices),
            'aspect_indices': torch.tensor(batch_aspect_indices),
            'left_indices': torch.tensor(batch_left_indices),
            'position_indices': torch.tensor(batch_position_indices),
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
