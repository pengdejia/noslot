# -*- coding: utf-8 -*-#

import os, json, random
import numpy as np
import torch
from models.module import ModelManager
from models.module_new import JointBert, CPosModelBert, CPosModelBertWithOutNoSlot, CPosModelBertIntent
from utils.loader import DatasetManager, BertDatasetManager
from utils.process import Processor, JointBertProcessor, CPosModelBertProcessor
from utils.config import *

if __name__ == "__main__":

    # Save training and model parameters.
    if not os.path.exists(args.save_dir):
        os.system("mkdir -p " + args.save_dir)

    log_path = os.path.join(args.save_dir, "param.json")
    with open(log_path, "w", encoding="utf8") as fw:
        fw.write(json.dumps(args.__dict__, indent=True))

    # Fix the random seed of package random.
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Fix the random seed of Pytorch when using GPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Fix the random seed of Pytorch when using CPU.
    torch.manual_seed(args.random_state)
    torch.random.manual_seed(args.random_state)

    # Instantiate a dataset object.
    # dataset = DatasetManager(args)

    dataset = BertDatasetManager(args)
    dataset.quick_build()
    dataset.show_summary()
    best_epoch = 0
    if args.method == "jointbert":
        model = JointBert(
                args, len(dataset.word_alphabet),
                len(dataset.slot_alphabet),
                len(dataset.intent_alphabet)
        )
        model.show_summary()
        print(model)
        processor = JointBertProcessor(dataset, model, args)
        if args.do_train:
            best_epoch = processor.train()

        result = JointBertProcessor.validate(
            os.path.join(args.save_dir, "model/model.pkl"),
            dataset,
            args.batch_size, len(dataset.intent_alphabet), args=args)
        print('\nAccepted performance: ' + str(result) + " at test dataset;\n")
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        with open(os.path.join(args.log_dir, args.log_name), 'w') as fw:
            fw.write(str(best_epoch) + ',' + str(result))

    elif args.method == "border":
        model = CPosModelBert(
            args, len(dataset.word_alphabet),
            len(dataset.slot_alphabet),
            len(dataset.intent_alphabet))
        print(model)
        model.show_summary()
        processor = CPosModelBertProcessor(dataset, model, args)
        if args.do_train:
            best_epoch = processor.train()
        result = CPosModelBertProcessor.validate(
            os.path.join(args.save_dir, "model/model.pkl"),
            dataset,
            args.batch_size, len(dataset.intent_alphabet), args=args)
        print('\nAccepted performance: ' + str(result) + " at test dataset;\n")
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        with open(os.path.join(args.log_dir, args.log_name), 'w') as fw:
            fw.write(str(best_epoch) + ',' + str(result))

    elif args.method == "borderWithoutNoSlot":
        model = CPosModelBertWithOutNoSlot(
            args, len(dataset.word_alphabet),
            len(dataset.slot_alphabet),
            len(dataset.intent_alphabet)
        )
        model.show_summary()
        print(model)
        processor = JointBertProcessor(dataset, model, args)
        if args.do_train:
            best_epoch = processor.train()

        result = JointBertProcessor.validate(
            os.path.join(args.save_dir, "model/model.pkl"),
            dataset,
            args.batch_size, len(dataset.intent_alphabet), args=args)
        print('\nAccepted performance: ' + str(result) + " at test dataset;\n")
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        with open(os.path.join(args.log_dir, args.log_name), 'w') as fw:
            fw.write(str(best_epoch) + ',' + str(result))

    elif args.method == "borderIntent":
        model = CPosModelBertIntent(
            args, len(dataset.word_alphabet),
            len(dataset.slot_alphabet),
            len(dataset.intent_alphabet))
        print(model)
        model.show_summary()
        processor = CPosModelBertProcessor(dataset, model, args)
        if args.do_train:
            best_epoch = processor.train()
        result = CPosModelBertProcessor.validate(
            os.path.join(args.save_dir, "model/model.pkl"),
            dataset,
            args.batch_size, len(dataset.intent_alphabet), args=args)
        print('\nAccepted performance: ' + str(result) + " at test dataset;\n")
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        with open(os.path.join(args.log_dir, args.log_name), 'w') as fw:
            fw.write(str(best_epoch) + ',' + str(result))

    elif args.method == "AGIF":
        model = ModelManager(
            args, len(dataset.word_alphabet),
            len(dataset.slot_alphabet),
            len(dataset.intent_alphabet)
        )
        model.show_summary()

        # To train and evaluate the models.
        process = Processor(dataset, model, args)
        best_epoch = process.train()
        result = Processor.validate(
            os.path.join(args.save_dir, "model/model.pkl"),
            dataset,
            args.batch_size, len(dataset.intent_alphabet), args=args)
        print('\nAccepted performance: ' + str(result) + " at test dataset;\n")
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        with open(os.path.join(args.log_dir, args.log_name), 'w') as fw:
            fw.write(str(best_epoch) + ',' + str(result))
