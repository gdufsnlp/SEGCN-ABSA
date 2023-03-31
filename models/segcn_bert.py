# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text.float(), self.w)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SEGCN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(SEGCN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.sentiment_embed = nn.Embedding(2, opt.bert_dim)
        self.ln = nn.LayerNorm(opt.bert_dim)
        self.gc1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.fc_opinion_predict = nn.Linear(opt.bert_dim, 1)
        self.fc = nn.Linear(2 * opt.bert_dim, opt.polarities_dim)

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask * x

    def forward(self, inputs):
        text_aspect_bert_indices, text_aspect_bert_segments_ids, \
        text_indices, aspect_indices, left_indices, adj = inputs
        batch = text_indices.shape[0]
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        output = self.bert(text_aspect_bert_indices, token_type_ids=text_aspect_bert_segments_ids)
        hidden = output.last_hidden_state
        text_out = self.dropout(F.relu(self.ln(hidden)))
        cls = self.dropout(output.pooler_output)
        output_op = self.fc_opinion_predict(hidden)
        output_op = torch.sigmoid(output_op).squeeze(2)
        sent_emb = self.sentiment_embed(torch.tensor([0, 1]).to(self.opt.device))
        sent_node = sent_emb.repeat(batch, 1, 1)
        word_node = text_out[:, 1:]
        x = torch.cat([sent_node, word_node], dim=1)
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        pos, neg, x = torch.split(x, [1, 1, word_node.shape[1]], dim=1)
        x = self.mask(x, aspect_double_idx)
        ra = torch.sum(x, dim=1) / aspect_len.unsqueeze(-1)
        r = torch.cat([cls, ra], dim=-1)
        output = self.fc(r)

        return output, output_op
