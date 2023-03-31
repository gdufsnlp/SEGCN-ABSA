# -*- coding: UTF-8 -*-

import os
import math
import argparse
import random
import time

import numpy
import torch
from tqdm import tqdm

from glove_bucket_iterator import BucketIterator
from bert_bucket_iterator import BucketIterator_BERT
from sklearn import metrics
from glove_data_utils import ABSADatesetReader
from bert_data_utils import ABSADatesetReader_BERT
from models import SEGCN, SEGCN_TRANSFER, SEGCN_BERT
from transformers import BertModel
import torch.nn as nn


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            absa_dataset = ABSADatesetReader_BERT(dataset=opt.dataset, model=opt.model_name, pretrain=opt.pretrained_bert_name)
            self.train_data_loader = BucketIterator_BERT(data=absa_dataset.train_data, batch_size=opt.batch_size, model=opt.model_name, shuffle=True)
            self.test_data_loader = BucketIterator_BERT(data=absa_dataset.test_data, batch_size=opt.batch_size, model=opt.model_name, shuffle=False)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            absa_dataset = ABSADatesetReader(dataset=opt.dataset, model=opt.model_name, embed_dim=opt.embed_dim)
            self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, model=opt.model_name, shuffle=True)
            self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, model=opt.model_name, shuffle=False)
            if opt.transfer == 'none':
                self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
            elif opt.transfer == '14-15':
                assert opt.dataset == 'rest15' and opt.model_name == 'segcn_transfer'
                params = torch.load('state_dict/segcn_rest14/acc_75.18_f1_61.36.pkl')
                sentemb_matrix = params['sentiment_embed.weight'].cpu().numpy()
                print('load rest14 sentiment nodes')
                self.model = opt.model_class(absa_dataset.embedding_matrix, sentemb_matrix, opt).to(opt.device)
            elif opt.transfer == '14-16':
                assert opt.dataset == 'rest16' and opt.model_name == 'segcn_transfer'
                params = torch.load('state_dict/segcn_rest14/acc_xx.xx_f1_xx.xx.pkl')
                sentemb_matrix = params['sentiment_embed.weight'].cpu().numpy()
                print('load rest14 sentiment nodes')
                self.model = opt.model_class(absa_dataset.embedding_matrix, sentemb_matrix, opt).to(opt.device)
            elif opt.transfer == '15-16':
                assert opt.dataset == 'rest16' and opt.model_name == 'segcn_transfer'
                params = torch.load('state_dict/segcn_rest15/acc_xx.xx_f1_xx.xx.pkl')
                sentemb_matrix = params['sentiment_embed.weight'].cpu().numpy()
                print('load rest15 sentiment nodes')
                self.model = opt.model_class(absa_dataset.embedding_matrix, sentemb_matrix, opt).to(opt.device)

        self._print_args()
        self.global_acc = 0.
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child, (name, _) in zip(self.model.children(), self.model.named_parameters()):
            if self.opt.transfer != 'none':
                if name == "sentiment_embed.weight":  # skip sentiment nodes
                    continue
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0
        if not os.path.exists('/state_dict/' + self.opt.model_name + '_' + self.opt.dataset):
            os.makedirs('/state_dict/' + self.opt.model_name + '_' + self.opt.dataset)
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total, n_correct_opinion_predict, n_total_opinion_predict = 0, 0, 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = [sample_batched[col].to(self.opt.device) for col in self.opt.outputs_cols]
                outputs, outputs_opinion_predict = self.model(inputs)

                loss1 = criterion[0](outputs, targets[0])
                loss2 = criterion[1](outputs_opinion_predict, targets[1].float())
                mask = targets[2]
                loss2 = loss2.masked_select(mask).mean()
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets[0]).sum().item()
                n_total += len(outputs)
                train_acc = n_correct / n_total
                op_pred = outputs_opinion_predict.masked_select(mask)
                op_pred = torch.where(op_pred > 0.5, 1, 0)
                n_correct_opinion_predict += (op_pred == targets[1].masked_select(mask)).sum().item()
                n_total_opinion_predict += len(outputs_opinion_predict.masked_select(mask))
                train_acc_opinion_predict = n_correct_opinion_predict / n_total_opinion_predict

                test_acc, test_f1 = self._evaluate_acc_f1()
                if test_acc > max_test_acc:
                    increase_flag = True
                    max_test_acc = test_acc
                    max_test_f1 = test_f1

                    if self.opt.save and test_f1 >= self.global_f1:
                        self.global_f1 = test_f1
                        if not os.path.exists('state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '/'):
                            os.makedirs('state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '/')
                        torch.save(self.model.state_dict(), 'state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '/acc_' + str(round(test_acc * 100, 2)) + '_f1_' + str(round(test_f1 * 100, 2)) + '.pkl')
                        print('>>> best model saved')
                    elif self.opt.save and test_acc >= self.global_acc:
                        self.global_acc = test_acc
                        if not os.path.exists('state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '/acc_' + str(round(test_acc * 100, 2)) + '_f1_' + str(round(test_f1 * 100, 2)) + '.pkl'):
                            torch.save(self.model.state_dict(), 'state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '/acc_' + str(round(test_acc * 100, 2)) + '_f1_' + str(round(test_f1 * 100, 2)) + '.pkl')
                            print('>>> best model saved')
                if global_step % self.opt.log_step == 0:
                    print('loss: {:.4f}, train_acc: {:.4f}, train_op_acc:{:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, train_acc_opinion_predict, test_acc, test_f1))
            print(
                'current best model:\n test_acc: {:.4f}, test_f1: {:.4f}'.format(max_test_acc, max_test_f1))
            if not increase_flag:
                continue_not_increase += 1
                if continue_not_increase >= 5:  # early stop
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = [t_sample_batched[col].to(self.opt.device) for col in self.opt.outputs_cols]
                t_outputs, t_outputs_opinion_predict = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets[0]).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets[0]
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets[0]), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return test_acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = [nn.CrossEntropyLoss(), nn.BCELoss()]
        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/' + self.opt.model_name + '_' + self.opt.dataset + '_val.txt', 'a', encoding='utf-8')
        f_out.write(time.strftime("%Y-%m-%d %H:%M:%S %A", time.localtime()) + '\n')
        for arg in vars(self.opt):
            f_out.write("{0}:{1}, ".format(arg, getattr(self.opt, arg)))
        f_out.write('\n')

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        for i in tqdm(range(self.opt.repeats)):
            print('repeat: ', (i + 1))
            f_out.write('repeat: ' + str(i + 1) + '\t')
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            start_time = time.time()
            max_test_acc, max_test_f1 = self._train(criterion, optimizer)
            total_time = time.time() - start_time
            f_out.write('max_test_acc: {0}, max_test_f1: {1}, train time: {2}s\n'.format(max_test_acc, max_test_f1, str(round(total_time, 2))))
            print('max_test_acc: {0}    max_test_f1: {1}    train time:{2}s'.format(max_test_acc, max_test_f1, str(round(total_time, 2))))

            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            print('#' * 100)
        f_out.write("max_test_acc_avg: {0}, max_test_f1_avg: {1}\n".format(max_test_acc_avg / self.opt.repeats, max_test_f1_avg / self.opt.repeats))
        f_out.close()


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='segcn', type=str, help='segcn, segcn_transfer, segcn_bert')
    parser.add_argument('--dataset', default='rest14', type=str, help='twitter, lap14, rest14,  rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='GloVe: 1e-3, BERT: 2e-5')
    parser.add_argument('--l2reg', default=1e-3, type=float, help='try 1e-3, 1e-4, 1e-5')
    parser.add_argument('--dropout', default=0.3, type=float, help='try 0.3, 0.5, 0.7')
    parser.add_argument('--repeats', default=2, type=int, help='10')
    parser.add_argument('--num_epoch', default=1, type=int, help='GloVe: 100, BERT:20')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32')
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--position_embed_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--transfer', default='none', type=str, help='none, 14-15, 14-16, 15-16')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default=None, type=str)

    opt = parser.parse_args()

    model_classes = {
        'segcn': SEGCN,
        'segcn_transfer': SEGCN_TRANSFER,
        'segcn_bert': SEGCN_BERT,
    }
    input_colses = {
        'segcn': ['text_indices', 'aspect_indices', 'left_indices', 'position_indices', 'dependency_graph'],
        'segcn_transfer': ['text_indices', 'aspect_indices', 'left_indices', 'position_indices', 'dependency_graph'],
        'segcn_bert': ['text_aspect_bert_indices', 'text_aspect_segments_indices', 'text_indices', 'aspect_indices', 'left_indices', 'dependency_graph']
    }
    output_colses = {
        'segcn': ['polarity', 'opinion_indices', 'mask'],
        'segcn_transfer': ['polarity', 'opinion_indices', 'mask'],
        'segcn_bert': ['polarity', 'opinion_indices', 'mask'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.outputs_cols = output_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
