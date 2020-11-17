# -*- coding: utf-8 -*-#

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.autograd import Variable
import torch.nn.functional as F

import os
import time
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

# Utils functions copied from Slot-gated model, origin url:
# 	https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py
from utils import miulab


def multilabel2one_hot(labels, nums):
    res = [0.] * nums
    if len(labels) == 0:
        return res
    if isinstance(labels[0], list):
        for label in labels[0]:
            res[label] = 1.
        return res
    for label in labels:
        res[label] = 1.
    return res


def instance2onehot(func, num_intent, data):
    res = []
    for intents in func(data):
        res.append(multilabel2one_hot(list(intents), num_intent))
    return np.array(res)


def normalize_adj(mx):
    """
    Row-normalize matrix  D^{-1}A
    torch.diag_embed: https://github.com/pytorch/pytorch/pull/12447
    """
    mx = mx.float()
    rowsum = mx.sum(2)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag_embed(r_inv, 0)
    mx = r_mat_inv.matmul(mx)
    return mx


class Processor(object):

    def __init__(self, dataset, model, args):
        self.__dataset = dataset
        self.__model = model
        self.args = args
        self.__batch_size = args.batch_size
        self.__load_dir = args.load_dir

        if args.gpu:
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.__criterion = nn.NLLLoss()
        self.__criterion_intent = nn.BCEWithLogitsLoss()
        self.__optimizer = optim.Adam(
            self.__model.parameters(), lr=self.__dataset.learning_rate, weight_decay=self.__dataset.l2_penalty
        )

        if self.__load_dir:
            if self.args.gpu:
                print("MODEL {} LOADED".format(str(self.__load_dir)))
                self.__model = torch.load(os.path.join(self.__load_dir, 'model/model.pkl'))
            else:
                print("MODEL {} LOADED".format(str(self.__load_dir)))
                self.__model = torch.load(os.path.join(self.__load_dir, 'model/model.pkl'),
                                          map_location=torch.device('cpu'))

    def train(self):
        best_dev_sent = 0.0
        best_epoch = 0
        no_improve = 0
        dataloader = self.__dataset.batch_delivery('train')
        for epoch in range(0, self.__dataset.num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0
            time_start = time.time()
            self.__model.train()

            for text_batch, slot_batch, intent_batch in tqdm(dataloader, ncols=50):
                padded_text, [sorted_slot, sorted_intent], seq_lens = self.__dataset.add_padding(
                    text_batch, [(slot_batch, True), (intent_batch, False)])
                sorted_intent = [multilabel2one_hot(intents, len(self.__dataset.intent_alphabet)) for intents in
                                 sorted_intent]
                text_var = torch.LongTensor(padded_text)
                slot_var = torch.LongTensor(sorted_slot)
                intent_var = torch.Tensor(sorted_intent)
                max_len = np.max(seq_lens)

                if self.args.gpu:
                    text_var = text_var.cuda()
                    slot_var = slot_var.cuda()
                    intent_var = intent_var.cuda()

                random_slot, random_intent = random.random(), random.random()
                if random_slot < self.__dataset.slot_forcing_rate:
                    slot_out, intent_out = self.__model(text_var, seq_lens, forced_slot=slot_var)
                else:
                    slot_out, intent_out = self.__model(text_var, seq_lens)

                slot_var = torch.cat([slot_var[i][:seq_lens[i]] for i in range(0, len(seq_lens))], dim=0)
                slot_loss = self.__criterion(slot_out, slot_var)
                intent_loss = self.__criterion_intent(intent_out, intent_var)
                batch_loss = slot_loss + intent_loss

                self.__optimizer.zero_grad()
                batch_loss.backward()
                self.__optimizer.step()

                try:
                    total_slot_loss += slot_loss.cpu().item()
                    total_intent_loss += intent_loss.cpu().item()
                except AttributeError:
                    total_slot_loss += slot_loss.cpu().data.numpy()[0]
                    total_intent_loss += intent_loss.cpu().data.numpy()[0]

            time_con = time.time() - time_start
            print(
                '[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, intent data is {:2.6f}, cost ' \
                'about {:2.6} seconds.'.format(epoch, total_slot_loss, total_intent_loss, time_con))

            change, time_start = False, time.time()
            dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score, dev_sent_acc_score = self.estimate(
                if_dev=True,
                test_batch=self.__batch_size,
                args=self.args)

            if dev_sent_acc_score > best_dev_sent:
                no_improve = 0
                best_epoch = epoch
                best_dev_sent = dev_sent_acc_score
                test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc = self.estimate(
                    if_dev=False, test_batch=self.__batch_size, args=self.args)

                print('\nTest result: epoch: {}, slot f1 score: {:.6f}, intent f1 score: {:.6f}, intent acc score:'
                      ' {:.6f}, semantic accuracy score: {:.6f}.'.
                      format(epoch, test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc))

                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                torch.save(self.__model, os.path.join(model_save_dir, "model.pkl"))
                torch.save(self.__dataset, os.path.join(model_save_dir, 'dataset.pkl'))

                time_con = time.time() - time_start
                print('[Epoch {:2d}]: In validation process, the slot f1 score is {:2.6f}, ' \
                      'the intent f1 score is {:2.6f}, the intent acc score is {:2.6f}, the semantic acc is {:.2f}, cost about {:2.6f} seconds.\n'.format(
                    epoch, dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score,
                    dev_sent_acc_score, time_con))
            else:
                no_improve += 1

            if self.args.early_stop == True:
                if no_improve > self.args.patience:
                    print('early stop at epoch {}'.format(epoch))
                    break
        print('Best epoch is {}'.format(best_epoch))
        return best_epoch

    def estimate(self, if_dev, args, test_batch=100):
        """
        Estimate the performance of model on dev or test dataset.
        """

        if if_dev:
            ss, pred_slot, real_slot, pred_intent, real_intent = self.prediction(
                self.__model, self.__dataset, "dev", test_batch, args)
        else:
            ss, pred_slot, real_slot, pred_intent, real_intent = self.prediction(
                self.__model, self.__dataset, "test", test_batch, args)

        num_intent = len(self.__dataset.intent_alphabet)
        slot_f1_score = miulab.computeF1Score(ss, real_slot, pred_slot, args)[0]
        intent_f1_score = f1_score(
            instance2onehot(self.__dataset.intent_alphabet.get_index, num_intent, real_intent),
            instance2onehot(self.__dataset.intent_alphabet.get_index, num_intent, pred_intent),
            average='macro')
        intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
        print("slot f1: {}, intent f1: {}, intent acc: {}, exact acc: {}".format(slot_f1_score, intent_f1_score,
                                                                                 intent_acc_score, sent_acc))
        # Write those sample both have intent and slot errors.
        with open(os.path.join(args.save_dir, 'error.txt'), 'w', encoding="utf8") as fw:
            for p_slot_list, r_slot_list, p_intent_list, r_intent in \
                    zip(pred_slot, real_slot, pred_intent, real_intent):
                fw.write(','.join(p_intent_list) + '\t' + ','.join(r_intent) + '\n')
                for w, r_slot, in zip(p_slot_list, r_slot_list):
                    fw.write(w + '\t' + r_slot + '\t''\n')
                fw.write('\n\n')

        return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc

    @staticmethod
    def validate(model_path, dataset, batch_size, num_intent, args):
        """
        validation will write mistaken samples to files and make scores.
        """

        if args.gpu:
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))

        ss, pred_slot, real_slot, pred_intent, real_intent = Processor.prediction(
            model, dataset, "test", batch_size, args)

        # To make sure the directory for save error prediction.
        mistake_dir = os.path.join(dataset.save_dir, "error")
        if not os.path.exists(mistake_dir):
            os.mkdir(mistake_dir)

        slot_f1_score = miulab.computeF1Score(ss, real_slot, pred_slot, args)[0]
        intent_f1_score = f1_score(instance2onehot(dataset.intent_alphabet.get_index, num_intent, real_intent),
                                   instance2onehot(dataset.intent_alphabet.get_index, num_intent, pred_intent),
                                   average='macro')
        intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
        print("slot f1: {}, intent f1: {}, intent acc: {}, exact acc: {}".format(slot_f1_score, intent_f1_score,
                                                                                 intent_acc_score, sent_acc))
        # Write those sample both have intent and slot errors.

        with open(os.path.join(args.save_dir, 'error.txt'), 'w', encoding="utf8") as fw:
            for p_slot_list, r_slot_list, p_intent_list, r_intent in \
                    zip(pred_slot, real_slot, pred_intent, real_intent):
                fw.write(','.join(p_intent_list) + '\t' + ','.join(r_intent) + '\n')
                for w, r_slot, in zip(p_slot_list, r_slot_list):
                    fw.write(w + '\t' + r_slot + '\t''\n')
                fw.write('\n\n')
        # with open(os.path.join(args.save_dir, 'slot_right.txt'), 'w', encoding="utf8") as fw:
        #     for p_slot_list, r_slot_list, tokens in \
        #             zip(pred_slot, real_slot, ss):
        #         if p_slot_list != r_slot_list:
        #             continue
        #         fw.write(' '.join(tokens) + '\n' + ' '.join(r_slot_list) + '\n' + ' '.join(p_slot_list) + '\n' + '\n\n')

        return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size, args):
        model.eval()

        if mode == "dev":
            dataloader = dataset.batch_delivery('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []
        all_token = []
        for text_batch, slot_batch, intent_batch in tqdm(dataloader, ncols=50):
            padded_text, [sorted_slot, sorted_intent], seq_lens = dataset.add_padding(
                text_batch, [(slot_batch, False), (intent_batch, False)],
                digital=False
            )
            real_slot.extend(sorted_slot)
            all_token.extend([pt[:seq_lens[idx]] for idx, pt in enumerate(padded_text)])
            for intents in list(Evaluator.expand_list(sorted_intent)):
                if '#' in intents:
                    real_intent.append(intents.split('#'))
                else:
                    real_intent.append([intents])

            digit_text = dataset.word_alphabet.get_index(padded_text)
            var_text = torch.LongTensor(digit_text)
            max_len = np.max(seq_lens)
            if args.gpu:
                var_text = var_text.cuda()
            slot_idx, intent_idx = model(var_text, seq_lens, n_predicts=1)
            nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]
            pred_slot.extend(dataset.slot_alphabet.get_instance(nested_slot))
            intent_idx_ = [[] for i in range(len(digit_text))]
            for item in intent_idx:
                intent_idx_[item[0]].append(item[1])
            intent_idx = intent_idx_
            pred_intent.extend(dataset.intent_alphabet.get_instance(intent_idx))
        if 'MixSNIPS' in args.data_dir or 'MixATIS' in args.data_dir or 'DSTC' in args.data_dir:
            [p_intent.sort() for p_intent in pred_intent]
        with open(os.path.join(args.save_dir, 'token.txt'), "w", encoding="utf8") as writer:
            idx = 0
            for line, slots, rss in zip(all_token, pred_slot, real_slot):
                for c, sl, rsl in zip(line, slots, rss):
                    writer.writelines(
                        str(sl == rsl) + " " + c + " " + sl + " " + rsl + "\n")
                idx = idx + len(line)
                writer.writelines("\n")

        return all_token, pred_slot, real_slot, pred_intent, real_intent


class Evaluator(object):

    @staticmethod
    def intent_acc(pred_intent, real_intent):
        total_count, correct_count = 0.0, 0.0
        for p_intent, r_intent in zip(pred_intent, real_intent):

            if p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """
        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def f1_score_intents(pred_array, real_array):
        pred_array = pred_array.transpose()
        real_array = real_array.transpose()
        P, R, F1 = 0, 0, 0
        for i in range(pred_array.shape[0]):
            TP, FP, FN = 0, 0, 0
            for j in range(pred_array.shape[1]):
                if (pred_array[i][j] + real_array[i][j]) == 2:
                    TP += 1
                elif real_array[i][j] == 1 and pred_array[i][j] == 0:
                    FN += 1
                elif pred_array[i][j] == 1 and real_array[i][j] == 0:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 += 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
            P += precision
            R += recall
        P /= pred_array.shape[0]
        R /= pred_array.shape[0]
        F1 /= pred_array.shape[0]
        return F1

    @staticmethod
    def f1_score(pred_list, real_list):
        """
        Get F1 score measured by predictions and ground-trues.
        """

        tp, fp, fn = 0.0, 0.0, 0.0
        for i in range(len(pred_list)):
            seg = set()
            result = [elem.strip() for elem in pred_list[i]]
            target = [elem.strip() for elem in real_list[i]]

            j = 0
            while j < len(target):
                cur = target[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(target):
                        str_ = target[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    seg.add((cur, j, k - 1))
                    j = k - 1
                j = j + 1

            tp_ = 0
            j = 0
            while j < len(result):
                cur = result[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(result):
                        str_ = result[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    if (cur, j, k - 1) in seg:
                        tp_ += 1
                    else:
                        fp += 1
                    j = k - 1
                j = j + 1

            fn += len(seg) - tp_
            tp += tp_

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        return 2 * p * r / (p + r) if p + r != 0 else 0

    """
    Max frequency prediction. 
    """

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def exp_decay_predict(sample, decay_rate=0.8):
        predict = []
        for items in sample:
            item_dict = {}
            curr_weight = 1.0
            for item in items[::-1]:
                item_dict[item] = item_dict.get(item, 0) + curr_weight
                curr_weight *= decay_rate
            predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items


class JointBertProcessor(object):

    def __init__(self, dataset, model, args):
        self.__dataset = dataset
        self.__model = model
        self.__batch_size = args.batch_size
        self.args = args
        self.__load_dir = args.load_dir

        if torch.cuda.is_available():
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.__criterion = nn.NLLLoss()
        self.__criterion_2 = nn.BCEWithLogitsLoss()

        self.params = list(filter(lambda p: p.requires_grad, self.__model.parameters()))
        self.__optimizer = optim.Adam(self.params, lr=self.__dataset.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=0)  # self.__dataset.l2_penalty) # (beta1, beta2)
        if self.__load_dir:
            if self.args.gpu:
                print("MODEL {} LOADED".format(str(self.__load_dir)))
                self.__model = torch.load(os.path.join(self.__load_dir, 'model/model.pkl'))
            else:
                print("MODEL {} LOADED".format(str(self.__load_dir)))
                self.__model = torch.load(os.path.join(self.__load_dir, 'model/model.pkl'),
                                          map_location=torch.device('cpu'))

    def train(self):

        # test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc = self.estimate(if_dev=False, args=self.args, test_batch=self.__batch_size)
        best_dev_slot = 0.0
        best_dev_intent = 0.0
        best_dev_sent = 0.0
        best_epoch = 0
        no_improve = 0
        dataloader = self.__dataset.batch_delivery('train')
        for epoch in range(0, self.__dataset.num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0

            time_start = time.time()
            self.__model.train()

            for text_batch, slot_batch, intent_batch in tqdm(dataloader, ncols=50):
                padded_text, [sorted_slot, sorted_intent], seq_lens = self.__dataset.add_padding(
                    text_batch, [(slot_batch, False), (intent_batch, False)], tokenizer=self.__dataset.tokenizer
                )

                sorted_intent = [multilabel2one_hot(intents, len(self.__dataset.intent_alphabet)) for intents in
                                 sorted_intent]

                padded_text['tokens'] = torch.LongTensor(padded_text['tokens'])
                padded_text['selects'] = torch.LongTensor(padded_text['selects'])
                padded_text['mask'] = torch.LongTensor(padded_text['mask'])
                text_var = padded_text
                slot_var = Variable(torch.LongTensor(list(Evaluator.expand_list(sorted_slot))))
                intent_var = Variable(torch.Tensor(sorted_intent))
                if self.args.single_intent:
                    length = intent_var.size()[1]
                    intent_var = intent_var.flatten()
                    index = torch.gt(intent_var, 0).nonzero()
                    intent_var = index % length
                    intent_var.reshape(-1)
                    intent_var = torch.squeeze(intent_var)

                # print(text_var.size())
                # print(slot_var.size())
                # print(intent_var.size())

                if torch.cuda.is_available():
                    text_var['tokens'] = text_var['tokens'].cuda()
                    text_var['selects'] = text_var['selects'].cuda()
                    text_var['mask'] = text_var['mask'].cuda()

                    slot_var = slot_var.cuda()
                    intent_var = intent_var.cuda()

                slot_out, intent_out = self.__model(text_var, seq_lens)

                # print(slot_out.size(), intent_out.size())

                slot_loss = self.__criterion(slot_out, slot_var)
                if self.args.single_intent:
                    intent_loss = self.__criterion(intent_out, intent_var.reshape(-1))
                else:
                    intent_loss = self.__criterion_2(intent_out, intent_var)

                batch_loss = self.args.lambda_slot * slot_loss + self.args.lambda_intent * intent_loss

                self.__optimizer.zero_grad()
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.params, 1)

                self.__optimizer.step()

                try:
                    total_slot_loss += slot_loss.cpu().item()
                    total_intent_loss += intent_loss.cpu().item()
                except AttributeError:
                    total_slot_loss += slot_loss.cpu().data.numpy()[0]
                    total_intent_loss += intent_loss.cpu().data.numpy()[0]

            time_con = time.time() - time_start
            print('[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, intent data is {:2.6f}, cost ' \
                  'about {:2.6} seconds.'.format(epoch, total_slot_loss, total_intent_loss, time_con))

            change, time_start = False, time.time()
            dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score, dev_sent_acc_score = self.estimate(if_dev=True, args=self.args,test_batch=self.__batch_size)
            if dev_sent_acc_score > best_dev_sent:
                no_improve = 0
                best_epoch = epoch
                best_dev_sent = dev_sent_acc_score
                test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc = self.estimate(if_dev=False, args=self.args, test_batch=self.__batch_size)

                print('\nTest result: epoch: {}, slot f1 score: {:.6f}, intent f1 score: {:.6f}, intent acc score:'
                      ' {:.6f}, semantic accuracy score: {:.6f}.'.
                      format(epoch, test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc))

                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                torch.save(self.__model, os.path.join(model_save_dir, "model.pkl"))
                torch.save(self.__dataset, os.path.join(model_save_dir, 'dataset.pkl'))

                time_con = time.time() - time_start
                print('[Epoch {:2d}]: In validation process, the slot f1 score is {:2.6f}, ' \
                      'the intent f1 score is {:2.6f}, the intent acc score is {:2.6f}, the semantic acc is {:.2f}, cost about {:2.6f} seconds.\n'.format(
                    epoch, dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score,
                    dev_sent_acc_score, time_con))
            else:
                no_improve += 1

            if self.args.early_stop == True:
                if no_improve > self.args.patience:
                    print('early stop at epoch {}'.format(epoch))
                    break
        print('Best epoch is {}'.format(best_epoch))
        return best_epoch

    def estimate(self, if_dev, args ,test_batch=100):
        """
        Estimate the performance of model on dev or test dataset.
        """

        if if_dev:
            ss, pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction(
                self.__model, self.__dataset, "dev", test_batch, args
            )
        else:
            ss , pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction(
                self.__model, self.__dataset, "test", test_batch, args
            )
        num_intent = len(self.__dataset.intent_alphabet)

        # print("num intent{}".format(num_intent))
        # if args.single_intent:
        #     slot_f1_score = miulab.computeF1ScoreSingleIntent(real_slot, pred_slot)[0]
        # else:
        slot_f1_score = miulab.computeF1Score(ss, real_slot, pred_slot, args)[0]
        intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        if self.args.single_intent:
            intent_f1_score = intent_acc_score
        else:
            intent_f1_score = f1_score(
                instance2onehot(self.__dataset.intent_alphabet.get_index, num_intent, real_intent),
                instance2onehot(self.__dataset.intent_alphabet.get_index, num_intent, pred_intent),
                average='macro')
        # intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
        print("slot f1: {}, intent f1: {}, intent acc: {}, exact acc: {}".format(slot_f1_score, intent_f1_score,
                                                                                 intent_acc_score, sent_acc))
        return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc

    @staticmethod
    def validate(model_path, dataset, batch_size, num_intent ,args, use_mask=False):
        """
        validation will write mistaken samples to files and make scores.
        """
        if args.gpu:
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        # dataset = torch.load(dataset_path)

        # Get the sentence list in test dataset.

        ss, pred_slot, real_slot, exp_pred_intent, real_intent, pred_intent = JointBertProcessor.prediction(
            model, dataset, "test", batch_size, args, use_mask=use_mask
        )
        slot_f1_score = miulab.computeF1Score(ss, real_slot, pred_slot, args)[0]
        intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        if args.single_intent:
            intent_f1_score = intent_acc_score
        else:
            intent_f1_score = f1_score(instance2onehot(dataset.intent_alphabet.get_index, num_intent, real_intent),
                                       instance2onehot(dataset.intent_alphabet.get_index, num_intent, pred_intent),
                                       average='macro')
        # intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
        print("slot f1: {}, intent f1: {}, intent acc: {}, exact acc: {}".format(slot_f1_score, intent_f1_score,
                                                                                 intent_acc_score, sent_acc))
        # sentences, select_index = zip(*pad_texts)
        # new_sent_list = []
        # for i in range(len(sentences)):
        #     sent, index_p = sentences[i], select_index[i]
        #     #    print("######select_index",sent,select_index)
        #     new_sent_list.append([sent[index] for index in index_p])
        #
        # sent_list = new_sent_list
        # # To make sure the directory for save error prediction.
        # mistake_dir = os.path.join(dataset.save_dir, "error")
        # if not os.path.exists(mistake_dir):
        #     os.mkdir(mistake_dir)
        #
        # # print(mistake_dir)
        # if use_mask == True:
        #     slot_file_path = os.path.join(mistake_dir, "slots_mask.txt")
        #     intent_file_path = os.path.join(mistake_dir, "intents_mask.txt")
        #     both_file_path = os.path.join(mistake_dir, "both_mask.txt")
        # else:
        #     slot_file_path = os.path.join(mistake_dir, "slots.txt")
        #     intent_file_path = os.path.join(mistake_dir, "intents.txt")
        #     both_file_path = os.path.join(mistake_dir, "both.txt")
        #
        # # Write those sample with mistaken slot prediction.
        # with open(slot_file_path, 'w') as fw:
        #     for w_list, r_slot_list, p_slot_list, p_intent, r_intent in zip(sent_list, real_slot, pred_slot,
        #                                                                     exp_pred_intent, real_intent):
        #         # print("#########################",r_slot_list != p_slot_list)
        #         if r_slot_list != p_slot_list:
        #             for w, r, p in zip(w_list, r_slot_list, p_slot_list):
        #                 fw.write(w + '\t' + r + '\t' + p + '\n')
        #             fw.write('\n')
        #         fw.write(r_intent + '\t' + p_intent + '\n\n')
        #
        # # Write those sample with mistaken intent prediction.
        # with open(intent_file_path, 'w') as fw:
        #     for w_list, p_intent, r_intent in zip(sent_list, exp_pred_intent, real_intent):
        #         if p_intent != r_intent:
        #             if p_intent != r_intent:
        #                 for w in w_list:
        #                     fw.write(w + '\n')
        #             fw.write(r_intent + '\t' + p_intent + '\n\n')
        #
        # # Write those sample both have intent and slot errors.
        # with open(both_file_path, 'w') as fw:
        #     for w_list, r_slot_list, p_slot_list, p_intent, r_intent in \
        #             zip(sent_list, real_slot, pred_slot, exp_pred_intent, real_intent):
        #
        #         if r_slot_list != p_slot_list or r_intent != p_intent:
        #             for w, r_slot, p_slot in zip(w_list, r_slot_list, p_slot_list):
        #                 fw.write(w + '\t' + r_slot + '\t' + p_slot + '\n')
        #             fw.write(r_intent + '\t' + p_intent + '\n\n')
        #
        # slot_f1 = miulab.computeF1Score(pred_slot, real_slot)[0]
        # intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        # sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size, args, use_mask=False):
        model.eval()

        if mode == "dev":
            dataloader = dataset.batch_delivery('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []
        pad_texts = []
        padded_text_token = []
        padded_text_selects = []
        for text_batch, slot_batch, intent_batch in tqdm(dataloader, ncols=50):
            # print(text_batch[0])
            # print(slot_batch[0])
            # print(intent_batch[0])
            padded_text, [sorted_slot, sorted_intent], seq_lens = dataset.add_padding(
                text_batch, [(slot_batch, False), (intent_batch, False)], tokenizer=dataset.tokenizer, digital=False
            )

            # print(padded_text['tokens'][0], padded_text['selects'][0], padded_text['mask'][0])
            # print(seq_lens[0])

            padded_text_token += padded_text['tokens']
            length_select = len(padded_text['tokens'][0])

            # print("#######tokenssssss",len(padded_text['tokens']),padded_text['tokens'])
            selects = []
            n = 0
            for i in range(len(padded_text['tokens'])):
                term = []
                while n < len(padded_text['selects']) and padded_text['selects'][n] < (i + 1) * length_select:
                    term.append(padded_text['selects'][n] - (i * length_select))
                    n += 1

                selects.append(term)

            padded_text_selects += selects

            real_slot.extend(sorted_slot)
            for intents in list(Evaluator.expand_list(sorted_intent)):
                if '#' in intents:
                    real_intent.append(intents.split('#'))
                else:
                    real_intent.append([intents])

            padded_text['tokens'] = torch.LongTensor(dataset.word_alphabet.get_index(padded_text['tokens']))
            padded_text['selects'] = torch.LongTensor(padded_text['selects'])
            padded_text['mask'] = torch.LongTensor(padded_text['mask'])

            var_text = padded_text
            # digit_text = dataset.word_alphabet.get_index(padded_text)
            # var_text = Variable(torch.LongTensor(digit_text))
            if args.gpu:
                # var_text = var_text.cuda()
                var_text['tokens'] = var_text['tokens'].cuda()
                var_text['selects'] = var_text['selects'].cuda()
                var_text['mask'] = var_text['mask'].cuda()

            slot_idx, intent_idx = model(var_text, seq_lens, n_predicts=1)

            # print("slot_idx{} length {}".format(slot_idx, len(slot_idx)))
            # print("intent_idx{} length{}".format(intent_idx, len(intent_idx)))
            # print(slot_idx)
            # print(len(intent_idx))
            # print(intent_idx)

            if args.single_intent:
                for nested_intent_ in intent_idx:
                    pred_intent.append(dataset.intent_alphabet.get_instance(list(nested_intent_)))
            else:
                intent_idx_ = [[] for i in range(len(padded_text['tokens']))]
                for item in intent_idx:
                    intent_idx_[item[0]].append(item[1])
                intent_idx = intent_idx_
                pred_intent.extend(dataset.intent_alphabet.get_instance(intent_idx))

            nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]
            pred_slot.extend(dataset.slot_alphabet.get_instance(nested_slot))

        if 'MixSNIPS' in args.data_dir or 'MixATIS' in args.data_dir or 'DSTC' in args.data_dir:
            [p_intent.sort() for p_intent in pred_intent]

        pad_texts = zip(padded_text_token, padded_text_selects)
        # print(real_intent[:3])
        # print(pred_intent[:3])
        # print(real_slot[:3])
        # print(pred_slot[:3])

        #TODO 修改返回的句子
        # print(padded_text_token[:3])
        # print(padded_text_selects[:3])
        """
        [['[CLS]', 'i', 'would', 'like', 'to', 'book', 'a', 'highly', 'rated', 'brass', '##erie', 'with', 'so', '##u', '##v', '##lak', '##i', 'neighboring', 'la', 'next', 'week', ',', 'what', 'is', 'the', 'forecast', 'for', 'in', '1', 'second', 'at', 'monte', 'ser', '##eno', 'for', 'freezing', 'temps', 'and', 'then', 'play', 'me', 'a', 'top', '-', 'ten', 'song', 'by', 'phil', 'och', '##s', 'on', 'groove', 'shark', '[SEP]'], ['[CLS]', 'please', 'add', 'jen', '##cy', 'anthony', 'to', 'my', 'play', '##list', 'this', 'is', 'mozart', ',', 'book', 'me', 'a', 'reservation', 'for', 'eight', 'for', 'the', 'top', '-', 'rated', 'bakery', 'eleven', 'hours', 'from', 'now', 'in', 'mango', 'and', 'then', 'can', 'i', 'get', 'the', 'movie', 'times', 'for', 'fox', 'theatres', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[CLS]', 'add', 'the', 'song', 'to', 'the', 'sounds', '##cape', '##s', 'for', 'gaming', 'play', '##list', ',', 'i', 'need', 'a', 'table', 'in', 'uruguay', 'in', '213', 'days', 'when', 'it', 's', 'chill', '##ier', 'and', 'the', 'book', 'history', 'by', 'contract', 'is', 'rated', 'five', 'stars', 'in', 'my', 'opinion', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']]
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 47, 48, 50, 51, 52], [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], [1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
        """
        tokens = []
        for padded_text, padded_text_select in zip(padded_text_token, padded_text_selects):
            token = []
            for i in range(len(padded_text_select) - 1):
                word = "".join(padded_text[padded_text_select[i]:padded_text_select[i + 1]]).replace("##", "")
                token.append(word)
            if padded_text[padded_text_select[-1]] is not "[SEP]":
                word = padded_text[padded_text_select[-1]]
                token.append(word)
            tokens.append(token)

        with open(os.path.join(args.save_dir, 'token.txt'), "w", encoding="utf8") as writer:
            for line, slots, rss, intent, rintent in zip(tokens, pred_slot, real_slot, pred_intent, real_intent):
                writer.writelines(str(intent == rintent) + "\t"+",".join(intent) + "\t" + ",".join(rintent) + "\n")
                for c, sl, rsl in zip(line, slots, rss):
                    writer.writelines(
                        str(sl == rsl) + "\t" + c + "\t" + sl + "\t" + rsl + "\n")
                writer.writelines("\n")
        return tokens, pred_slot, real_slot, pred_intent, real_intent, pred_intent


class CPosModelBertProcessor(object):
    def __init__(self, dataset, model, args):
        self.__dataset = dataset
        self.__model = model
        self.__batch_size = args.batch_size
        self.args = args
        self.__load_dir = args.load_dir

        if torch.cuda.is_available():
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.__criterion = nn.NLLLoss()
        self.__criterion_2 = nn.BCEWithLogitsLoss()

        self.params = list(filter(lambda p: p.requires_grad, self.__model.parameters()))
        self.__optimizer = optim.Adam(self.params, lr=self.__dataset.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=0)  # self.__dataset.l2_penalty) # (beta1, beta2)
        if self.__load_dir:
            if self.args.gpu:
                print("MODEL {} LOADED".format(str(self.__load_dir)))
                self.__model = torch.load(os.path.join(self.__load_dir, 'model/model.pkl'))
            else:
                print("MODEL {} LOADED".format(str(self.__load_dir)))
                self.__model = torch.load(os.path.join(self.__load_dir, 'model/model.pkl'),
                                          map_location=torch.device('cpu'))

    def train(self):

        # test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc = self.estimate(if_dev=False, args=self.args, test_batch=self.__batch_size)
        best_dev_slot = 0.0
        best_dev_intent = 0.0
        best_dev_sent = 0.0
        best_epoch = 0
        no_improve = 0

        dataloader = self.__dataset.batch_delivery('train')
        for epoch in range(0, self.__dataset.num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0

            time_start = time.time()
            self.__model.train()

            for text_batch, slot_batch, intent_batch in tqdm(dataloader, ncols=50):
                padded_text, [sorted_slot, sorted_intent], seq_lens = self.__dataset.add_padding(
                    text_batch, [(slot_batch, False), (intent_batch, False)], tokenizer=self.__dataset.tokenizer
                )

                sorted_intent = [multilabel2one_hot(intents, len(self.__dataset.intent_alphabet)) for intents in
                                 sorted_intent]

                padded_text['tokens'] = torch.LongTensor(padded_text['tokens'])
                padded_text['selects'] = torch.LongTensor(padded_text['selects'])
                padded_text['mask'] = torch.LongTensor(padded_text['mask'])
                text_var = padded_text
                slot_var = Variable(torch.LongTensor(list(Evaluator.expand_list(sorted_slot))))
                intent_var = Variable(torch.Tensor(sorted_intent))
                bio_var = slot_var.eq(0).to(dtype=torch.long)
                if self.args.single_intent:
                    length = intent_var.size()[1]
                    intent_var = intent_var.flatten()
                    index = torch.gt(intent_var, 0).nonzero()
                    intent_var = index % length
                    intent_var.reshape(-1)
                    intent_var = torch.squeeze(intent_var)

                # print(text_var.size())
                # print(slot_var.size())
                # print(intent_var.size())

                if torch.cuda.is_available():
                    text_var['tokens'] = text_var['tokens'].cuda()
                    text_var['selects'] = text_var['selects'].cuda()
                    text_var['mask'] = text_var['mask'].cuda()

                    slot_var = slot_var.cuda()
                    intent_var = intent_var.cuda()
                    bio_var = bio_var.cuda()

                random_intent = random.random()
                if random_intent < self.__dataset.intent_forcing_rate:
                    slot_out, bio_out, intent_out = self.__model(
                        text_var, seq_lens, forced_bio=bio_var, forced_slot=slot_var
                    )
                else:
                    slot_out, bio_out, intent_out = self.__model(text_var, seq_lens)

                # print(slot_out.size(), intent_out.size())

                slot_loss = self.__criterion(slot_out, slot_var)
                bio_loss = self.__criterion(bio_out, bio_var)
                # intent_loss = self.__criterion_2(intent_out, intent_var)
                if self.args.single_intent:
                    intent_loss = self.__criterion(intent_out, intent_var)
                else:
                    intent_loss = self.__criterion_2(intent_out, intent_var)

                batch_loss = self.args.lambda_slot * slot_loss + self.args.lambda_intent * intent_loss + bio_loss
                self.__optimizer.zero_grad()
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.params, 1)

                self.__optimizer.step()

                try:
                    total_slot_loss += slot_loss.cpu().item()
                    total_intent_loss += intent_loss.cpu().item()
                except AttributeError:
                    total_slot_loss += slot_loss.cpu().data.numpy()[0]
                    total_intent_loss += intent_loss.cpu().data.numpy()[0]

            time_con = time.time() - time_start
            print('[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, intent data is {:2.6f}, cost ' \
                  'about {:2.6} seconds.'.format(epoch, total_slot_loss, total_intent_loss, time_con))

            change, time_start = False, time.time()
            dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score, dev_sent_acc_score = self.estimate(if_dev=True, args=self.args,test_batch=self.__batch_size)
            if dev_sent_acc_score > best_dev_sent:
                no_improve = 0
                best_epoch = epoch
                best_dev_sent = dev_sent_acc_score
                test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc = self.estimate(if_dev=False, args=self.args, test_batch=self.__batch_size)

                print('\nTest result: epoch: {}, slot f1 score: {:.6f}, intent f1 score: {:.6f}, intent acc score:'
                      ' {:.6f}, semantic accuracy score: {:.6f}.'.
                      format(epoch, test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc))

                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                torch.save(self.__model, os.path.join(model_save_dir, "model.pkl"))
                torch.save(self.__dataset, os.path.join(model_save_dir, 'dataset.pkl'))

                time_con = time.time() - time_start
                print('[Epoch {:2d}]: In validation process, the slot f1 score is {:2.6f}, ' \
                      'the intent f1 score is {:2.6f}, the intent acc score is {:2.6f}, the semantic acc is {:.2f}, cost about {:2.6f} seconds.\n'.format(
                    epoch, dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score,
                    dev_sent_acc_score, time_con))
            else:
                no_improve += 1
            if self.args.early_stop:
                if no_improve > self.args.patience:
                    print('early stop at epoch {}'.format(epoch))
                    break
        print('Best epoch is {}'.format(best_epoch))
        return best_epoch

    def estimate(self, if_dev, args ,test_batch=100):
        """
        Estimate the performance of model on dev or test dataset.
        """

        if if_dev:
            ss, pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction(
                self.__model, self.__dataset, "dev", test_batch, args
            )
        else:
            ss , pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction(
                self.__model, self.__dataset, "test", test_batch, args
            )
        num_intent = len(self.__dataset.intent_alphabet)

        # print("num intent{}".format(num_intent))

        slot_f1_score = miulab.computeF1Score(ss, real_slot, pred_slot, args)[0]
        intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        if self.args.single_intent:
            intent_f1_score = intent_acc_score
        else:
            intent_f1_score = f1_score(
                instance2onehot(self.__dataset.intent_alphabet.get_index, num_intent, real_intent),
                instance2onehot(self.__dataset.intent_alphabet.get_index, num_intent, pred_intent),
                average='macro')
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
        print("slot f1: {}, intent f1: {}, intent acc: {}, exact acc: {}".format(slot_f1_score, intent_f1_score,
                                                                                 intent_acc_score, sent_acc))
        return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc

    @staticmethod
    def validate(model_path, dataset, batch_size, num_intent ,args, use_mask=False):
        """
        validation will write mistaken samples to files and make scores.
        """
        if args.gpu:
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        # dataset = torch.load(dataset_path)

        # Get the sentence list in test dataset.

        ss, pred_slot, real_slot, exp_pred_intent, real_intent, pred_intent = JointBertProcessor.prediction(
            model, dataset, "test", batch_size, args, use_mask=use_mask
        )
        slot_f1_score = miulab.computeF1Score(ss, real_slot, pred_slot, args)[0]
        intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        if args.single_intent:
            intent_f1_score = intent_acc_score
        else:
            intent_f1_score = f1_score(instance2onehot(dataset.intent_alphabet.get_index, num_intent, real_intent),
                                       instance2onehot(dataset.intent_alphabet.get_index, num_intent, pred_intent),
                                       average='macro')
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
        print("slot f1: {}, intent f1: {}, intent acc: {}, exact acc: {}".format(slot_f1_score, intent_f1_score,
                                                                                 intent_acc_score, sent_acc))

        return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size, args, use_mask=False):
        model.eval()

        if mode == "dev":
            dataloader = dataset.batch_delivery('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []
        pad_texts = []
        padded_text_token = []
        padded_text_selects = []
        for text_batch, slot_batch, intent_batch in tqdm(dataloader, ncols=50):
            # print(text_batch[0])
            # print(slot_batch[0])
            # print(intent_batch[0])
            padded_text, [sorted_slot, sorted_intent], seq_lens = dataset.add_padding(
                text_batch, [(slot_batch, False), (intent_batch, False)], tokenizer=dataset.tokenizer, digital=False
            )

            # print(padded_text['tokens'][0], padded_text['selects'][0], padded_text['mask'][0])
            # print(seq_lens[0])

            padded_text_token += padded_text['tokens']
            length_select = len(padded_text['tokens'][0])

            # print("#######tokenssssss",len(padded_text['tokens']),padded_text['tokens'])
            selects = []
            n = 0
            for i in range(len(padded_text['tokens'])):
                term = []
                while n < len(padded_text['selects']) and padded_text['selects'][n] < (i + 1) * length_select:
                    term.append(padded_text['selects'][n] - (i * length_select))
                    n += 1

                selects.append(term)

            padded_text_selects += selects

            real_slot.extend(sorted_slot)
            for intents in list(Evaluator.expand_list(sorted_intent)):
                if '#' in intents:
                    real_intent.append(intents.split('#'))
                else:
                    real_intent.append([intents])

            padded_text['tokens'] = torch.LongTensor(dataset.word_alphabet.get_index(padded_text['tokens']))
            padded_text['selects'] = torch.LongTensor(padded_text['selects'])
            padded_text['mask'] = torch.LongTensor(padded_text['mask'])

            var_text = padded_text
            # digit_text = dataset.word_alphabet.get_index(padded_text)
            # var_text = Variable(torch.LongTensor(digit_text))
            if args.gpu:
                # var_text = var_text.cuda()
                var_text['tokens'] = var_text['tokens'].cuda()
                var_text['selects'] = var_text['selects'].cuda()
                var_text['mask'] = var_text['mask'].cuda()

            slot_idx, intent_idx = model(var_text, seq_lens, n_predicts=1)

            # print("slot_idx{} length {}".format(slot_idx, len(slot_idx)))
            # print("intent_idx{} length{}".format(intent_idx, len(intent_idx)))

            if args.single_intent:
                for nested_intent_ in intent_idx:
                    pred_intent.append(dataset.intent_alphabet.get_instance(list(nested_intent_)))
            else:
                intent_idx_ = [[] for i in range(len(padded_text['tokens']))]
                for item in intent_idx:
                    intent_idx_[item[0]].append(item[1])
                intent_idx = intent_idx_
                pred_intent.extend(dataset.intent_alphabet.get_instance(intent_idx))

            nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]
            pred_slot.extend(dataset.slot_alphabet.get_instance(nested_slot))

        if 'MixSNIPS' in args.data_dir or 'MixATIS' in args.data_dir or 'DSTC' in args.data_dir:
            [p_intent.sort() for p_intent in pred_intent]

        pad_texts = zip(padded_text_token, padded_text_selects)
        # print(real_intent[:3])
        # print(pred_intent[:3])
        # print(real_slot[:3])
        # print(pred_slot[:3])

        #TODO 修改返回的句子
        # print(padded_text_token[:3])
        # print(padded_text_selects[:3])
        """
        [['[CLS]', 'i', 'would', 'like', 'to', 'book', 'a', 'highly', 'rated', 'brass', '##erie', 'with', 'so', '##u', '##v', '##lak', '##i', 'neighboring', 'la', 'next', 'week', ',', 'what', 'is', 'the', 'forecast', 'for', 'in', '1', 'second', 'at', 'monte', 'ser', '##eno', 'for', 'freezing', 'temps', 'and', 'then', 'play', 'me', 'a', 'top', '-', 'ten', 'song', 'by', 'phil', 'och', '##s', 'on', 'groove', 'shark', '[SEP]'], ['[CLS]', 'please', 'add', 'jen', '##cy', 'anthony', 'to', 'my', 'play', '##list', 'this', 'is', 'mozart', ',', 'book', 'me', 'a', 'reservation', 'for', 'eight', 'for', 'the', 'top', '-', 'rated', 'bakery', 'eleven', 'hours', 'from', 'now', 'in', 'mango', 'and', 'then', 'can', 'i', 'get', 'the', 'movie', 'times', 'for', 'fox', 'theatres', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[CLS]', 'add', 'the', 'song', 'to', 'the', 'sounds', '##cape', '##s', 'for', 'gaming', 'play', '##list', ',', 'i', 'need', 'a', 'table', 'in', 'uruguay', 'in', '213', 'days', 'when', 'it', 's', 'chill', '##ier', 'and', 'the', 'book', 'history', 'by', 'contract', 'is', 'rated', 'five', 'stars', 'in', 'my', 'opinion', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']]
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 47, 48, 50, 51, 52], [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], [1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
        """
        tokens = []
        for padded_text, padded_text_select in zip(padded_text_token, padded_text_selects):
            token = []
            for i in range(len(padded_text_select) - 1):
                word = "".join(padded_text[padded_text_select[i]:padded_text_select[i + 1]]).replace("##", "")
                token.append(word)
            if padded_text[padded_text_select[-1]] is not "[SEP]":
                word = padded_text[padded_text_select[-1]]
                token.append(word)
            tokens.append(token)

        with open(os.path.join(args.save_dir, 'token.txt'), "w", encoding="utf8") as writer:
            for line, slots, rss, intent, rintent in zip(tokens, pred_slot, real_slot, pred_intent, real_intent):
                writer.writelines(str(intent == rintent) + "\t"+",".join(intent) + "\t" + ",".join(rintent) + "\n")
                for c, sl, rsl in zip(line, slots, rss):
                    writer.writelines(
                        str(sl == rsl) + "\t" + c + "\t" + sl + "\t" + rsl + "\n")
                writer.writelines("\n")
        return tokens, pred_slot, real_slot, pred_intent, real_intent, pred_intent