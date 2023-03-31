# -*- coding: utf-8 -*-


import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM


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


class SEGCN_TRANSFER(nn.Module):
    def __init__(self, embedding_matrix, sentemb_matrix, opt):
        super(SEGCN_TRANSFER, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.position_embed = nn.Embedding(200, opt.position_embed_dim, padding_idx=0)
        self.sentiment_embed = nn.Embedding.from_pretrained(torch.tensor(sentemb_matrix, dtype=torch.float), freeze=False)
        self.text_lstm = DynamicLSTM(opt.embed_dim + opt.position_embed_dim, opt.hidden_dim, num_layers=1,
                                     batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(2 * opt.hidden_dim)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)

        self.fc_opinion_predict = nn.Linear(2 * opt.hidden_dim, 1)
        self.fc = nn.Linear(4 * opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(opt.dropout)

    def position_weight(self, batch_size, seq_len, aspect_double_idx, text_len, aspect_len):
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight

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
        text_indices, aspect_indices, left_indices, position_indices, adj = inputs
        batch = text_indices.shape[0]
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        position = self.position_embed(position_indices)
        input = torch.cat([text, position], dim=-1)
        input = self.text_embed_dropout(input)
        text_out, (_, _) = self.text_lstm(input, text_len)
        text_out = F.relu(self.ln(text_out))
        output_op = self.fc_opinion_predict(text_out)
        output_op = torch.sigmoid(output_op).squeeze(2)
        sent_emb = self.sentiment_embed(torch.tensor([0, 1]).to(self.opt.device))
        sent_node = sent_emb.repeat(batch, 1, 1)
        x = torch.cat([sent_node, text_out], dim=1)
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        pos, neg, x = torch.split(x, [1, 1, text_indices.size()[1]], dim=1)
        x = self.mask(x, aspect_double_idx)
        graph_aspect = torch.sum(x, dim=1) / aspect_len.unsqueeze(-1)
        Wp = self.position_weight(text_out.shape[0], text_out.shape[1], aspect_double_idx, text_len, aspect_len)
        text_out = Wp * text_out
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        rs = torch.matmul(alpha, text_out).squeeze(1)
        r = torch.cat([graph_aspect, rs], dim=-1)
        output = self.fc(r)

        return output, output_op
