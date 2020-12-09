from transformers import BertForSequenceClassification, AdamW, BertTokenizer, BertModel
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from models.module_new import QKVAttention

# path = "/home/djpeng/AGIF/AGIF-master/bert_base_uncased/"
# tokenizer = BertTokenizer.from_pretrained(path)
# model = BertModel.from_pretrained(path)
# print(model)

# padded_text_token = [['[CLS]', 'i', 'would', 'like', 'to', 'book', 'a', 'highly', 'rated', 'brass', '##erie', 'with', 'so', '##u', '##v',
#   '##lak', '##i', 'neighboring', 'la', 'next', 'week', ',', 'what', 'is', 'the', 'forecast', 'for', 'in', '1', 'second',
#   'at', 'monte', 'ser', '##eno', 'for', 'freezing', 'temps', 'and', 'then', 'play', 'me', 'a', 'top', '-', 'ten',
#   'song', 'by', 'phil', 'och', '##s', 'on', 'groove', 'shark', '[SEP]'],
#  ['[CLS]', 'please', 'add', 'jen', '##cy', 'anthony', 'to', 'my', 'play', '##list', 'this', 'is', 'mozart', ',', 'book',
#   'me', 'a', 'reservation', 'for', 'eight', 'for', 'the', 'top', '-', 'rated', 'bakery', 'eleven', 'hours', 'from',
#   'now', 'in', 'mango', 'and', 'then', 'can', 'i', 'get', 'the', 'movie', 'times', 'for', 'fox', 'theatres', '[SEP]',
#   '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
#  ['[CLS]', 'add', 'the', 'song', 'to', 'the', 'sounds', '##cape', '##s', 'for', 'gaming', 'play', '##list', ',', 'i',
#   'need', 'a', 'table', 'in', 'uruguay', 'in', '213', 'days', 'when', 'it', 's', 'chill', '##ier', 'and', 'the', 'book',
#   'history', 'by', 'contract', 'is', 'rated', 'five', 'stars', 'in', 'my', 'opinion', '[SEP]', '[PAD]', '[PAD]',
#   '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']]
# padded_text_selects = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38,
#   39, 40, 41, 42, 45, 46, 47, 48, 50, 51, 52],
#  [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
#   36, 37, 38, 39, 40, 41, 42],
#  [1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35,
#   36, 37, 38, 39, 40]]
#
# def get_tokens():
#     tokens = []
#     for padded_text, padded_text_select in zip(padded_text_token, padded_text_selects):
#         # print(len(padded_text))
#         # print(padded_text)
#         # print(len(padded_text_select))
#         # print(padded_text_select)
#         token = []
#         for i in range(len(padded_text_select) - 1):
#             word = "".join(padded_text[padded_text_select[i]:padded_text_select[i+1]]).replace("##", "")
#             token.append(word)
#         if padded_text[padded_text_select[-1]] is not "[SEP]":
#             word = padded_text[padded_text_select[-1]]
#             token.append(word)
#         print("\n".join(token))
#         tokens.append(token)


if __name__ == '__main__':
    # get_tokens()
    print("fuck you")
    top_k = 3
    arr = np.array([1, 3, 2, 4, 5])
    top_k_idx = arr.argsort()[::-1][0:top_k]
    print(top_k_idx)
    # pred_bio = torch.rand([3, 2])
    # pred_slot = torch.rand([3, 5])
    # print(pred_bio.size())
    # BI_slot = pred_bio[:, 1].unsqueeze(dim=1)
    # print(BI_slot.size())
    # print(pred_bio)
    # print(BI_slot)
    # BI_slot = BI_slot.expand([pred_slot.size()[0], pred_slot.size()[1] - 2])
    # print(BI_slot.size())
    # pred_bio = torch.cat([pred_bio, BI_slot], dim=1)
    # print(pred_bio.size())
    # print(pred_bio)
    # pred_slot = torch.mul(pred_slot, pred_bio)
    # print(pred_slot.size())
    # bio_index = torch.LongTensor([0, 1, 1, 0])
    # bio_index = bio_index.unsqueeze(dim=1)
    # print(bio_index.size())
    # compare_tensor = torch.zeros_like(bio_index)
    # bio_index = torch.le(bio_index, compare_tensor).to(dtype=torch.long)
    # print(bio_index.size())
    # print(bio_index)

    # print(str([1, 2, 3] == [1, 2, 3]))
    # slot_var = torch.LongTensor([1, 2, 3, 4, 0])
    # bio_var = slot_var.eq(0).to(dtype=torch.long)
    # print(bio_var.type())
    # print(bio_var)
    # intent_var = torch.LongTensor([[0, 2, 1], [1, 0, 0]])
    # batch_size, length = intent_var.size()
    # intent_list = intent_var.flatten().numpy().tolist()
    # target_list = []
    # for index in range(batch_size):
    #     temp_intent_list = intent_list[index *length: (index+1)*length]
    #     print(temp_intent_list)
    #     for i in range(len(temp_intent_list)):
    #         if temp_intent_list[i] > 0:
    #             target_list.append(i)
    #             break
    # target_tensor = torch.LongTensor(target_list)
    # print(target_tensor)

    # print(index.reshape(-1))
    # mask = torch.gt(intent_var, 0)
    # print(mask)
    # print(intent_var[mask])
    # print(intent_var.index_select(dim=0))
    # torch.index_select()
    # qkv_attention = QKVAttention(768, 768, 768, 768, 768, dropout_rate=0.1)
    # query = torch.rand([64, 1, 768])
    # print(query.size())
    # key = torch.rand([64, 20, 768])
    # print(key.size())
    # target_tensor = qkv_attention(query, key, key)
    # print(target_tensor.size())
    # feed_BIO = torch.FloatTensor([[0],[1], [1]])
    # print(feed_BIO.size())
    # hiddens = torch.rand([3, 6])
    # print(hiddens)
    # print(hiddens.size())
    # no_slot_index = feed_BIO.expand_as(hiddens)
    # print(no_slot_index)
    # print(no_slot_index.size())
    # no_slot_tensor = torch.mul(hiddens, no_slot_index)
    # print(no_slot_tensor)
    # print(no_slot_tensor.size())
    # params = list(filter(lambda p: p.requires_grad, model.parameters()))