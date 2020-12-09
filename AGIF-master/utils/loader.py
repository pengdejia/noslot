# -*- coding: utf-8 -*-#

import os
import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict
from ordered_set import OrderedSet

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import BertTokenizer


class Alphabet(object):
    """
    Storage and serialization a set of elements.
    """

    def __init__(self, name, if_use_pad, if_use_unk):

        self.__name = name
        self.__if_use_pad = if_use_pad
        self.__if_use_unk = if_use_unk

        self.__index2instance = OrderedSet()
        self.__instance2index = OrderedDict()

        # Counter Object record the frequency
        # of element occurs in raw text.
        self.__counter = Counter()

        if if_use_pad:
            self.__sign_pad = "<PAD>"
            self.add_instance(self.__sign_pad)
        if if_use_unk:
            self.__sign_unk = "<UNK>"
            self.add_instance(self.__sign_unk)

    @property
    def name(self):
        return self.__name

    def add_instance(self, instance, multi_intent=False):
        """ Add instances to alphabet.

        1, We support any iterative data structure which
        contains elements of str type.

        2, We will count added instances that will influence
        the serialization of unknown instance.

        :param instance: is given instance or a list of it.
        """

        if isinstance(instance, (list, tuple)):
            for element in instance:
                self.add_instance(element, multi_intent=multi_intent)
            return

        # We only support elements of str type.
        assert isinstance(instance, str)
        if multi_intent and '#' in instance:
            for element in instance.split('#'):
                self.add_instance(element, multi_intent=multi_intent)
            return
        # count the frequency of instances.
        self.__counter[instance] += 1

        if instance not in self.__index2instance:
            self.__instance2index[instance] = len(self.__index2instance)
            self.__index2instance.append(instance)

    def get_index(self, instance, multi_intent=False):
        """ Serialize given instance and return.

        For unknown words, the return index of alphabet
        depends on variable self.__use_unk:

            1, If True, then return the index of "<UNK>";
            2, If False, then return the index of the
            element that hold max frequency in training data.

        :param instance: is given instance or a list of it.
        :return: is the serialization of query instance.
        """

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem, multi_intent=multi_intent) for elem in instance]

        assert isinstance(instance, str)
        if multi_intent and '#' in instance:
            return [self.get_index(element, multi_intent=multi_intent) for element in instance.split('#')]

        try:
            return self.__instance2index[instance]
        except KeyError:
            if self.__if_use_unk:
                return self.__instance2index[self.__sign_unk]
            else:
                max_freq_item = self.__counter.most_common(1)[0][0]
                return self.__instance2index[max_freq_item]

    def get_instance(self, index):
        """ Get corresponding instance of query index.

        if index is invalid, then throws exception.

        :param index: is query index, possibly iterable.
        :return: is corresponding instance.
        """

        if isinstance(index, list):
            return [self.get_instance(elem) for elem in index]

        return self.__index2instance[index]

    def save_content(self, dir_path):
        """ Save the content of alphabet to files.

        There are two kinds of saved files:
            1, The first is a list file, elements are
            sorted by the frequency of occurrence.

            2, The second is a dictionary file, elements
            are sorted by it serialized index.

        :param dir_path: is the directory path to save object.
        """

        # Check if dir_path exists.
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        list_path = os.path.join(dir_path, self.__name + "_list.txt")
        with open(list_path, 'w', encoding="utf8") as fw:
            for element, frequency in self.__counter.most_common():
                fw.write(element + '\t' + str(frequency) + '\n')

        dict_path = os.path.join(dir_path, self.__name + "_dict.txt")
        with open(dict_path, 'w', encoding="utf8") as fw:
            for index, element in enumerate(self.__index2instance):
                fw.write(element + '\t' + str(index) + '\n')

    def __len__(self):
        return len(self.__index2instance)

    def __str__(self):
        return 'Alphabet {} contains about {} words: \n\t{}'.format(self.name, len(self), self.__index2instance)


class TorchDataset(Dataset):
    """
    Helper class implementing torch.utils.data.Dataset to
    instantiate DataLoader which deliveries data batch.
    """

    def __init__(self, text, slot, intent):
        self.__text = list(text)
        self.__slot = slot
        self.__intent = intent

    def __getitem__(self, index):
        return self.__text[index], self.__slot[index], self.__intent[index]

    def __len__(self):
        # Pre-check to avoid bug.
        assert len(self.__text) == len(self.__slot)
        assert len(self.__text) == len(self.__intent)
        # print(len(self.__text))
        # print(len(self.__intent))
        # print(len(self.__slot))
        return len(self.__text)


class DatasetManager(object):

    def __init__(self, args):

        # Instantiate alphabet objects.
        self.__word_alphabet = Alphabet('word', if_use_pad=True, if_use_unk=True)
        self.__slot_alphabet = Alphabet('slot', if_use_pad=False, if_use_unk=False)
        self.__intent_alphabet = Alphabet('intent', if_use_pad=False, if_use_unk=False)

        # Record the raw text of dataset.
        self.__text_word_data = {}
        self.__text_slot_data = {}
        self.__text_intent_data = {}

        # Record the serialization of dataset.
        self.__digit_word_data = {}
        self.__digit_slot_data = {}
        self.__digit_intent_data = {}

        self.__args = args

    @property
    def test_sentence(self):
        return deepcopy(self.__text_word_data['test'])

    @property
    def word_alphabet(self):
        return deepcopy(self.__word_alphabet)

    @property
    def slot_alphabet(self):
        return deepcopy(self.__slot_alphabet)

    @property
    def intent_alphabet(self):
        return deepcopy(self.__intent_alphabet)

    @property
    def num_epoch(self):
        return self.__args.num_epoch

    @property
    def batch_size(self):
        return self.__args.batch_size

    @property
    def learning_rate(self):
        return self.__args.learning_rate

    @property
    def l2_penalty(self):
        return self.__args.l2_penalty

    @property
    def save_dir(self):
        return self.__args.save_dir

    @property
    def slot_forcing_rate(self):
        return self.__args.slot_forcing_rate

    def show_summary(self):
        """
        :return: show summary of dataset, training parameters.
        """

        print("Training parameters are listed as follows:\n")

        print('\tnumber of train sample:                    {};'.format(len(self.__text_word_data['train'])))
        print('\tnumber of dev sample:                      {};'.format(len(self.__text_word_data['dev'])))
        print('\tnumber of test sample:                     {};'.format(len(self.__text_word_data['test'])))
        print('\tnumber of epoch:						    {};'.format(self.num_epoch))
        print('\tbatch size:							    {};'.format(self.batch_size))
        print('\tlearning rate:							    {};'.format(self.learning_rate))
        print('\trandom seed:							    {};'.format(self.__args.random_state))
        print('\trate of l2 penalty:					    {};'.format(self.l2_penalty))
        print('\trate of dropout in network:                {};'.format(self.__args.dropout_rate))
        print('\tteacher forcing rate(slot)		    		{};'.format(self.slot_forcing_rate))

        print("\nEnd of parameters show. Save dir: {}.\n\n".format(self.save_dir))

    def quick_build(self):
        """
        Convenient function to instantiate a dataset object.
        """

        train_path = os.path.join(self.__args.data_dir, 'train.txt')
        dev_path = os.path.join(self.__args.data_dir, 'dev.txt')
        test_path = os.path.join(self.__args.data_dir, 'test.txt')

        self.add_file(train_path, 'train', if_train_file=True)
        self.add_file(dev_path, 'dev', if_train_file=False)
        self.add_file(test_path, 'test', if_train_file=False)

        # Check if save path exists.
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        alphabet_dir = os.path.join(self.__args.save_dir, "alphabet")
        self.__word_alphabet.save_content(alphabet_dir)
        self.__slot_alphabet.save_content(alphabet_dir)
        self.__intent_alphabet.save_content(alphabet_dir)

    def get_dataset(self, data_name, is_digital):
        """ Get dataset of given unique name.

        :param data_name: is name of stored dataset.
        :param is_digital: make sure if want serialized data.
        :return: the required dataset.
        """

        if is_digital:
            return self.__digit_word_data[data_name], \
                   self.__digit_slot_data[data_name], \
                   self.__digit_intent_data[data_name]
        else:
            return self.__text_word_data[data_name], \
                   self.__text_slot_data[data_name], \
                   self.__text_intent_data[data_name]

    def add_file(self, file_path, data_name, if_train_file):
        text, slot, intent = self.__read_file(file_path)

        if if_train_file:
            self.__word_alphabet.add_instance(text)
            self.__slot_alphabet.add_instance(slot)
            self.__intent_alphabet.add_instance(intent, multi_intent=(self.__args.single_intent == False))

        # Record the raw text of dataset.
        self.__text_word_data[data_name] = text
        self.__text_slot_data[data_name] = slot
        self.__text_intent_data[data_name] = intent

        # Serialize raw text and stored it.
        self.__digit_word_data[data_name] = self.__word_alphabet.get_index(text)
        if if_train_file:
            self.__digit_slot_data[data_name] = self.__slot_alphabet.get_index(slot)
            self.__digit_intent_data[data_name] = self.__intent_alphabet.get_index(intent, multi_intent=(self.__args.single_intent == False))

    @staticmethod
    def __read_file(file_path):
        """ Read data file of given path.

        :param file_path: path of data file.
        :return: list of sentence, list of slot and list of intent.
        """

        texts, slots, intents = [], [], []
        text, slot = [], []

        with open(file_path, 'r', encoding="utf8") as fr:
            for line in fr.readlines():
                items = line.strip().split()

                if len(items) == 1:
                    texts.append(text)
                    slots.append(slot)
                    if "/" not in items[0]:
                        intents.append(items)
                    else:
                        new = items[0].split("/")
                        intents.append([new[1]])

                    # clear buffer lists.
                    text, slot = [], []

                elif len(items) == 2:
                    text.append(items[0].strip())
                    slot.append(items[1].strip())

        return texts, slots, intents

    def batch_delivery(self, data_name, batch_size=None, is_digital=True, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size

        if is_digital:
            text = self.__digit_word_data[data_name]
            slot = self.__digit_slot_data[data_name]
            intent = self.__digit_intent_data[data_name]
        else:
            text = self.__text_word_data[data_name]
            slot = self.__text_slot_data[data_name]
            intent = self.__text_intent_data[data_name]
        dataset = TorchDataset(text, slot, intent)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.__collate_fn)

    @staticmethod
    def add_padding(texts, items=None, digital=True):
        len_list = [len(text) for text in texts]
        max_len = max(len_list)

        # Get sorted index of len_list.
        sorted_index = np.argsort(len_list)[::-1]

        trans_texts, seq_lens, trans_items = [], [], None
        if items is not None:
            trans_items = [[] for _ in range(0, len(items))]

        for index in sorted_index:
            seq_lens.append(deepcopy(len_list[index]))
            trans_texts.append(deepcopy(texts[index]))
            if digital:
                trans_texts[-1].extend([0] * (max_len - len_list[index]))
            else:
                trans_texts[-1].extend(['<PAD>'] * (max_len - len_list[index]))

            # This required specific if padding after sorting.
            if items is not None:
                for item, (o_item, required) in zip(trans_items, items):
                    item.append(deepcopy(o_item[index]))
                    if required:
                        if digital:
                            item[-1].extend([0] * (max_len - len_list[index]))
                        else:
                            item[-1].extend(['<PAD>'] * (max_len - len_list[index]))

        if items is not None:
            return trans_texts, trans_items, seq_lens
        else:
            return trans_texts, seq_lens

    @staticmethod
    def __collate_fn(batch):
        """
        helper function to instantiate a DataLoader Object.
        """

        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch


class BertAlphabet():
    def __init__(self, tokenizer, name, if_use_pad, if_use_unk):
        super(BertAlphabet, self).__init__()
        self.__tokenizer = tokenizer

    @property
    def name(self):
        return self.__name

    def add_instance(self, instance):
        pass

    def get_index(self, instance):

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem) for elem in instance]
        assert isinstance(instance, str)
        return self.__tokenizer._convert_token_to_id(instance)

    def get_instance(self, index):

        if isinstance(index, list):
            return [self.get_instance(elem) for elem in index]

        return self.__tokenizer._convert_id_to_token(index)

    def save_content(self, dir_path):
        pass

    def __len__(self):
        return len(self.__tokenizer.vocab)

    def __str__(self):
        return 'Bert Alphabet'


class BertDatasetManager(object):

    def __init__(self, args):
        # print(args.bert_model)
        print(os.listdir(args.bert_model))
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        # Instantiate alphabet objects.
        self.__word_alphabet = BertAlphabet(self.tokenizer, 'word', if_use_pad=True, if_use_unk=True)
        self.__slot_alphabet = Alphabet('slot', if_use_pad=False, if_use_unk=False)
        self.__intent_alphabet = Alphabet('intent', if_use_pad=False, if_use_unk=False)
        self.__domain_alphabet = Alphabet('domain', if_use_pad=False, if_use_unk=False)
        # self.__slot_alphabet = Alphabet('slot', if_use_pad=False, if_use_unk=True)
        # self.__intent_alphabet = Alphabet('intent', if_use_pad=False, if_use_unk=True)
        # self.__domain_alphabet = Alphabet('domain', if_use_pad=False, if_use_unk=True)

        self.__split_domain = args.split_domain

        # Record the raw text of dataset.
        self.__text_word_data = {}
        self.__text_slot_data = {}
        self.__text_intent_data = {}
        self.__text_domain_data = {}

        # Record the serialization of dataset.
        self.__digit_word_data = {}
        self.__digit_slot_data = {}
        self.__digit_intent_data = {}
        self.__digit_domain_data = {}

        self.__args = args

        # domain_dict_path = os.path.join(args.data_dir, "domain_tensor.pkl")
        # self.domain_tensor = pickle.load(open(domain_dict_path, 'rb'))

    @property
    def test_sentence(self):
        return deepcopy(self.__text_word_data['test'])

    @property
    def word_alphabet(self):
        return deepcopy(self.__word_alphabet)

    @property
    def slot_alphabet(self):
        return deepcopy(self.__slot_alphabet)

    @property
    def intent_alphabet(self):
        return deepcopy(self.__intent_alphabet)

    @property
    def domain_alphabet(self):
        return deepcopy(self.__domain_alphabet)

    @property
    def num_epoch(self):
        return self.__args.num_epoch

    @property
    def batch_size(self):
        return self.__args.batch_size

    @property
    def learning_rate(self):
        return self.__args.learning_rate

    @property
    def l2_penalty(self):
        return self.__args.l2_penalty

    @property
    def save_dir(self):
        return self.__args.save_dir

    @property
    def intent_forcing_rate(self):
        return self.__args.intent_forcing_rate

    @property
    def slot_forcing_rate(self):
        return self.__args.slot_forcing_rate

    def show_summary(self):
        """
        :return: show summary of dataset, training parameters.
        """

        print("Training parameters are listed as follows:\n")

        # print('\tnumber of train sample:                    {};'.format(len(self.__text_word_data['train'])))
        # print('\tnumber of dev sample:                      {};'.format(len(self.__text_word_data['dev'])))
        # print('\tnumber of test sample:                     {};'.format(len(self.__text_word_data['test'])))
        print('\tnumber of epoch:						    {};'.format(self.num_epoch))
        print('\tbatch size:							    {};'.format(self.batch_size))
        print('\tlearning rate:							    {};'.format(self.learning_rate))
        print('\trandom seed:							    {};'.format(self.__args.random_state))
        print('\trate of l2 penalty:					    {};'.format(self.l2_penalty))
        print('\trate of dropout in network:                {};'.format(self.__args.dropout_rate))
        print('\tteacher forcing rate(slot)		    		{};'.format(self.slot_forcing_rate))
        print('\tteacher forcing rate(intent):		    	{};'.format(self.intent_forcing_rate))

        print("\nEnd of parameters show. Save dir: {}.\n\n".format(self.save_dir))

    def quick_build(self):
        """
        Convenient function to instantiate a dataset object.
        """

        train_path = os.path.join(self.__args.data_dir, 'train.txt')
        dev_path = os.path.join(self.__args.data_dir, 'dev.txt')
        test_path = os.path.join(self.__args.data_dir, 'test.txt')

        self.add_file(train_path, 'train', if_train_file=True)
        self.add_file(dev_path, 'dev', if_train_file=False)
        self.add_file(test_path, 'test', if_train_file=False)

        # Check if save path exists.
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        alphabet_dir = os.path.join(self.__args.save_dir, "alphabet")
        # self.__word_alphabet.save_content(alphabet_dir)
        self.__slot_alphabet.save_content(alphabet_dir)
        self.__intent_alphabet.save_content(alphabet_dir)
        self.__domain_alphabet.save_content(alphabet_dir)

    def get_dataset(self, data_name, is_digital):
        """ Get dataset of given unique name.

        :param data_name: is name of stored dataset.
        :param is_digital: make sure if want serialized data.
        :return: the required dataset.
        """
        if self.__split_domain:
            if is_digital:
                return self.__digit_word_data[data_name], \
                       self.__digit_slot_data[data_name], \
                       self.__digit_intent_data[data_name], \
                       self.__digit_domain_data[data_name]
            else:
                return self.__text_word_data[data_name], \
                       self.__text_slot_data[data_name], \
                       self.__text_intent_data[data_name], \
                       self.__text_domain_data[data_name]
        else:
            if is_digital:
                return self.__digit_word_data[data_name], \
                       self.__digit_slot_data[data_name], \
                       self.__digit_intent_data[data_name]
            else:
                return self.__text_word_data[data_name], \
                       self.__text_slot_data[data_name], \
                       self.__text_intent_data[data_name]

    def add_file(self, file_path, data_name, if_train_file):
        text, slot, intent, domain = self.__read_file(file_path, split_domain=self.__split_domain)

        if if_train_file:
            self.__word_alphabet.add_instance(text)
            self.__slot_alphabet.add_instance(slot)
            self.__intent_alphabet.add_instance(intent, multi_intent=(self.__args.single_intent == False))
            self.__domain_alphabet.add_instance(domain)

        seq_len = [len(words) for words in text]
        # Record the raw text of dataset.
        total_texts_tokens = []
        total_select_index = []
        for words in text:
            cut_words = []
            select_index = []
            for word in words:
                select_index.append(len(cut_words) + 1)
                cut_words += self.tokenizer.tokenize(word)
            cut_words = ['[CLS]'] + cut_words + ['[SEP]']
            total_select_index.append(select_index)
            total_texts_tokens.append(cut_words)

        self.__text_word_data[data_name] = list(zip(total_texts_tokens, total_select_index, seq_len))

        self.__text_slot_data[data_name] = slot
        self.__text_intent_data[data_name] = intent
        self.__text_domain_data[data_name] = domain

        # Serialize raw text and stored it.
        # print("text",text)
        self.__digit_word_data[data_name] = zip(
            [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in total_texts_tokens], total_select_index,
            seq_len)  # self.__word_alphabet.get_index(text)

        if if_train_file:
            self.__digit_slot_data[data_name] = self.__slot_alphabet.get_index(slot)
            self.__digit_intent_data[data_name] = self.__intent_alphabet.get_index(intent, multi_intent=(self.__args.single_intent == False))
            self.__digit_domain_data[data_name] = self.__domain_alphabet.get_index(domain)

    @staticmethod
    def __read_file(file_path, split_domain=False):
        """ Read data file of given path.

        :param file_path: path of data file.
        :param split_domain:    False: texts/slots/intents(mix domain and intent)   or    True : texts/slots/intents/domains
        :return: list of sentence, list of slot and list of intent.
        """

        texts, slots, intents, domains = [], [], [], []
        text, slot = [], []

        with open(file_path, 'r') as fr:
            for line in fr.readlines():
                items = line.strip().split()

                if len(items) == 1:
                    texts.append(text)
                    slots.append(slot)
                    data = items[0].split("@")
                    if split_domain:
                        intents.append([data[1]])
                        domains.append([data[0]])
                    else:
                        intents.append(items)

                    # clear buffer lists.
                    text, slot = [], []

                elif len(items) == 2:
                    text.append(items[0].strip())
                    slot.append(items[1].strip())

        return texts, slots, intents, domains

    def batch_delivery(self, data_name, batch_size=None, is_digital=True, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size

        if is_digital:
            text = self.__digit_word_data[data_name]
            # print("##texts",text)
            slot = self.__digit_slot_data[data_name]
            intent = self.__digit_intent_data[data_name]
            domain = self.__digit_domain_data[data_name]
        else:
            text = self.__text_word_data[data_name]
            slot = self.__text_slot_data[data_name]
            intent = self.__text_intent_data[data_name]
            domain = self.__text_domain_data[data_name]

        if self.__split_domain:

            dataset = TorchDataset(text, slot, intent, domain)
        else:
            dataset = TorchDataset(text, slot, intent)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.__collate_fn)

    @staticmethod
    def add_padding(texts, items=None, digital=True, tokenizer=None):

        def expand_list(nested_list):
            for item in nested_list:
                if isinstance(item, (list, tuple)):
                    for sub_item in expand_list(item):
                        yield sub_item
                else:
                    yield item

        # print("##texts",texts)
        texts_tokens, select_indexs, seq_len = zip(*texts)
        sorted_index = np.argsort(seq_len)[::-1]

        max_len = max([len(tokens) for tokens in texts_tokens])

        tran_texts_tokens, tran_select_indexs, Masks, tran_seq_len = [], [], [], []

        len_list = [len(text) for text in texts_tokens]
        trans_texts, trans_items = dict(), None

        if items is not None:
            trans_items = [[] for _ in range(0, len(items))]

        for index in sorted_index:
            tran_texts_tokens.append(deepcopy(texts_tokens[index]))
            tran_select_indexs.append(deepcopy(select_indexs[index]))
            tran_seq_len.append(deepcopy(seq_len[index]))
            Masks.append([1 for i in range(len(tran_texts_tokens[-1]))])
            Masks[-1].extend([1] * (max_len - len_list[index]))
            if digital:
                tran_texts_tokens[-1].extend([0] * (max_len - len_list[index]))
            else:
                tran_texts_tokens[-1].extend(['[PAD]'] * (max_len - len_list[index]))

            if items is not None:
                for item, (o_item, required) in zip(trans_items, items):
                    item.append(deepcopy(o_item[index]))
                    if required:
                        if digital:
                            item[-1].extend([0] * (max_len - seq_len[index]))
                        else:
                            item[-1].extend(['[PAD]'] * (max_len - seq_len[index]))

        tran_select_indexs = [[i + idx * max_len for i in selected_index] for idx, selected_index in
                              enumerate(tran_select_indexs)]
        trans_texts = {'tokens': tran_texts_tokens, 'selects': list(expand_list(tran_select_indexs)), 'mask': Masks}
        if items is not None:
            return trans_texts, trans_items, tran_seq_len
        else:
            return trans_texts, tran_seq_len

        # # Get sorted index of len_list.
        # sorted_index = np.argsort(len_list)[::-1]
        # # sorted_index = [i for i in range(len(len_list))]

        # trans_texts, seq_lens, trans_items = [], [], None
        # if items is not None:
        #     trans_items = [[] for _ in range(0, len(items))]

        # for index in sorted_index:
        #     seq_lens.append(deepcopy(len_list[index]))
        #     trans_texts.append(deepcopy(texts[index]))

        #     if digital:
        #         trans_texts[-1].extend([0] * (max_len - len_list[index]))
        #     else:
        #         trans_texts[-1].extend(['[PAD]'] * (max_len - len_list[index]))
        #     # print("@#@ ",trans_texts[-1])
        #     # This required specific if padding after sorting.
        #     if items is not None:
        #         for item, (o_item, required) in zip(trans_items, items):
        #             item.append(deepcopy(o_item[index]))
        #             if required:
        #                 if digital:
        #                     item[-1].extend([0] * (max_len - len_list[index]+1))
        #                 else:
        #                     item[-1].extend(['[PAD]'] * (max_len - len_list[index]+1))

        # # seq_lens = [i+2 for i in seq_lens]
        # if items is not None:
        #     return trans_texts, trans_items, seq_lens
        # else:
        #     return trans_texts, seq_lens

    @staticmethod
    def __collate_fn(batch):
        """
        helper function to instantiate a DataLoader Object.
        """

        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch