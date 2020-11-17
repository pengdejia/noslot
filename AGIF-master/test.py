from transformers import BertForSequenceClassification, AdamW, BertTokenizer, BertModel
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# path = "/home/djpeng/AGIF/AGIF-master/bert_base_uncased/"
# tokenizer = BertTokenizer.from_pretrained(path)
# model = BertModel.from_pretrained(path)
# print(model)

padded_text_token = [['[CLS]', 'i', 'would', 'like', 'to', 'book', 'a', 'highly', 'rated', 'brass', '##erie', 'with', 'so', '##u', '##v',
  '##lak', '##i', 'neighboring', 'la', 'next', 'week', ',', 'what', 'is', 'the', 'forecast', 'for', 'in', '1', 'second',
  'at', 'monte', 'ser', '##eno', 'for', 'freezing', 'temps', 'and', 'then', 'play', 'me', 'a', 'top', '-', 'ten',
  'song', 'by', 'phil', 'och', '##s', 'on', 'groove', 'shark', '[SEP]'],
 ['[CLS]', 'please', 'add', 'jen', '##cy', 'anthony', 'to', 'my', 'play', '##list', 'this', 'is', 'mozart', ',', 'book',
  'me', 'a', 'reservation', 'for', 'eight', 'for', 'the', 'top', '-', 'rated', 'bakery', 'eleven', 'hours', 'from',
  'now', 'in', 'mango', 'and', 'then', 'can', 'i', 'get', 'the', 'movie', 'times', 'for', 'fox', 'theatres', '[SEP]',
  '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
 ['[CLS]', 'add', 'the', 'song', 'to', 'the', 'sounds', '##cape', '##s', 'for', 'gaming', 'play', '##list', ',', 'i',
  'need', 'a', 'table', 'in', 'uruguay', 'in', '213', 'days', 'when', 'it', 's', 'chill', '##ier', 'and', 'the', 'book',
  'history', 'by', 'contract', 'is', 'rated', 'five', 'stars', 'in', 'my', 'opinion', '[SEP]', '[PAD]', '[PAD]',
  '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']]
padded_text_selects = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38,
  39, 40, 41, 42, 45, 46, 47, 48, 50, 51, 52],
 [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
  36, 37, 38, 39, 40, 41, 42],
 [1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35,
  36, 37, 38, 39, 40]]

def get_tokens():
    tokens = []
    for padded_text, padded_text_select in zip(padded_text_token, padded_text_selects):
        # print(len(padded_text))
        # print(padded_text)
        # print(len(padded_text_select))
        # print(padded_text_select)
        token = []
        for i in range(len(padded_text_select) - 1):
            word = "".join(padded_text[padded_text_select[i]:padded_text_select[i+1]]).replace("##", "")
            token.append(word)
        if padded_text[padded_text_select[-1]] is not "[SEP]":
            word = padded_text[padded_text_select[-1]]
            token.append(word)
        print("\n".join(token))
        tokens.append(token)


if __name__ == '__main__':
    # get_tokens()
    print("fuck you")
    print(str([1, 2, 3] == [1, 2, 3]))