# -*- coding: utf-8 -*-#
import argparse
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description='A Based on Bert no slot Framework for Joint Multiple Intent Detection and Slot Filling')

parser.add_argument("--do_train", "-dt", action="store_true", default=False)
parser.add_argument('--method', '-md', type=str, default='border')
#数据是单意图还是多意图
parser.add_argument('--single_intent', '-si', action="store_true", default=True)
# Dataset and Other Parameters
parser.add_argument('--data_dir', '-dd', help='dataset file path', type=str, default='./data/SNIPS')
parser.add_argument('--save_dir', '-sd', type=str, default='./save/total/bert_noslot/SNIPS')
parser.add_argument('--load_dir', '-ld', type=str, default=None)
parser.add_argument('--log_dir', '-lod', type=str, default='./log/total/bert_noslot/SNIPS')
parser.add_argument('--log_name', '-ln', type=str, default='log.txt')
parser.add_argument("--random_state", '-rs', help='random seed', type=int, default=11)
parser.add_argument('--gpu', '-g', action='store_true', help='use gpu', required=False, default=True)

# Training parameters.
parser.add_argument('--num_epoch', '-ne', type=int, default=30)
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=5e-5)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.1)
parser.add_argument('--intent_forcing_rate', '-ifr', type=float, default=0.95)
parser.add_argument("--differentiable", "-d", action="store_true", default=False)
parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.95)
parser.add_argument('--lambda_intent', '-lambda1', type=int, default=1)
parser.add_argument('--lambda_slot', '-lambda2', type=int, default=2)


parser.add_argument('--threshold', '-thr', type=float, default=0.5)
parser.add_argument('--bert_model', '-bm', type=str, default="/home2/djpeng/bert-base-uncased")
parser.add_argument("--split_domain", "-sdo", action="store_true", default=False)
parser.add_argument("--do_lower_case", "-dlc", action="store_true", default=False)
parser.add_argument('--early_stop', action='store_true', default=True)
parser.add_argument('--patience', '-pa', type=int, default=5)


# Model parameters.
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=32)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
parser.add_argument('--bio_embedding_dim', '-ied', type=int, default=2)
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=128)
parser.add_argument('--bio_decoder_hidden_dim', '-idhd', type=int, default=2)
parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)

parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)


# parser.add_argument("--differentiable", "-d", action="store_true", default=False)


args = parser.parse_args()
args.gpu = args.gpu and torch.cuda.is_available()
print(str(vars(args)))
