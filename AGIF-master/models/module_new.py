import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import BertTokenizer, BertPreTrainedModel, BertModel


class Transformers(nn.Module):

    def __init__(self, config):
        """Initialize model."""
        super(Transformers, self).__init__()
        self.__bert_model = BertModel.from_pretrained(config)
        self.hidden_size = self.__bert_model.config.hidden_size

    def forward(self, texts, seq_lens):
        tokens, selects, attention_mask = texts['tokens'], texts['selects'], texts['mask']

        token_type_ids = torch.zeros_like(tokens)

        transformer_hiddens, cls_hidden = self.__bert_model(tokens, token_type_ids=token_type_ids,
                                                            attention_mask=attention_mask)

        hiddens = transformer_hiddens.view(-1, self.hidden_size).index_select(0, selects)
        # select_tokens = tokens.view(-1,1).index_select(0, selects)

        # hiddens = None
        # for i in range(len(seq_lens)):
        #     end = seq_lens[i]
        #     if hiddens is None:
        #         hiddens = transformer_hiddens[i,1:end-1,:]
        #     else:
        #         hiddens = torch.cat([hiddens,transformer_hiddens[i,1:end-1,:]],dim=0)

        return hiddens, cls_hidden  # ,select_tokens


class JointBert(nn.Module):  ####临时修改版

    def __init__(self, args, num_word, num_slot, num_intent):
        super(JointBert, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        self.__transformer = Transformers(args.bert_model)

        self.__tramsformer_hidden_dim = self.__transformer.hidden_size

        self.__dropout = nn.Dropout(self.__args.dropout_rate)


        self.__intent_Linear = nn.Linear(self.__tramsformer_hidden_dim, self.__num_intent)
        self.__slot_Linear = nn.Linear(self.__tramsformer_hidden_dim, self.__num_slot)

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def forward(self, text, seq_lens, n_predicts=None, forced_slot=None, forced_bio=None):
        # word_tensor, _ = self.__embedding(text)
        # lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        # attention_hiddens = self.__attention(word_tensor, seq_lens)
        # hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)

        hiddens, cls_hidden = self.__transformer(text, seq_lens)
        pred_intent = self.__intent_Linear(self.__dropout(cls_hidden))
        pred_slot = self.__slot_Linear(self.__dropout(hiddens))
        # pred_BIO = self.__bio_decoder(
        #     hiddens, seq_lens,
        #     forced_input=forced_bio
        # )
        # # pred_BIO = self.__border_Linear(self.__dropout(hiddens))
        #
        # _, feed_BIO = pred_BIO.topk(1)
        #
        # if forced_bio is not None:
        #     feed_BIO = forced_bio.unsqueeze(1).float()
        #
        # border_hidden = torch.mul(hiddens, feed_BIO)
        #
        # new_hiddens = self.__attention(hiddens, border_hidden, seq_lens)
        # # new_hiddens = self.__attention(hiddens,hiddens,seq_lens)  #self_attention
        #
        # total_hidden = torch.cat([new_hiddens, hiddens], dim=-1)
        #
        # # total_hidden = hiddens
        # pred_slot = self.__slot_decoder(
        #     total_hidden, seq_lens,
        #     forced_input=forced_slot
        # )
        # pred_slot =  self.__slot_Linear(self.__dropout(total_hidden))

        # print("##########")
        # intent_index = (torch.sigmoid(pred_intent) > self.__args.threshold).nonzero()
        if n_predicts is None:
            if self.__args.single_intent:
                return F.log_softmax(pred_slot, dim=1),  F.log_softmax(pred_intent, dim=-1)
            else:
                return F.log_softmax(pred_slot, dim=1),  pred_intent
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            # 如果是单意图直接返回top1
            if self.__args.single_intent:
                _, intent_index = pred_intent.topk(n_predicts, dim=1)
            else:
                intent_index = (torch.sigmoid(pred_intent) > self.__args.threshold).nonzero()

            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()


class CPosModelBert(nn.Module):  ####临时修改版

    def __init__(self, args, num_word, num_slot, num_intent):
        super(CPosModelBert, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        self.__transformer = Transformers(args.bert_model)

        self.__tramsformer_hidden_dim = self.__transformer.hidden_size

        self.__dropout = nn.Dropout(self.__args.dropout_rate)

        # Initialize an self-attention layer.
        self.__attention = AttentionQ(
            self.__tramsformer_hidden_dim,
            self.__tramsformer_hidden_dim,
            self.__args.attention_hidden_dim,
            self.__tramsformer_hidden_dim,
            self.__args.dropout_rate
        )

        # Initialize an Decoder object for bio.
        self.__bio_decoder = LSTMDecoder(
            self.__tramsformer_hidden_dim,
            self.__args.bio_decoder_hidden_dim,
            2,
            self.__args.dropout_rate,
            embedding_dim=self.__args.bio_embedding_dim
        )

        self.__intent_Linear = nn.Linear(self.__tramsformer_hidden_dim, self.__num_intent)
        # self.__slot_Linear = nn.Linear(self.__tramsformer_hidden_dim,self.__num_slot)
        self.__slot_Linear = nn.Linear(2 * self.__tramsformer_hidden_dim, self.__num_slot)
        self.__border_Linear = nn.Linear(self.__tramsformer_hidden_dim, 2)
        self.__slot_decoder = LSTMDecoder(
            2 * self.__tramsformer_hidden_dim,
            self.__args.slot_decoder_hidden_dim,
            num_slot,
            self.__args.dropout_rate,
            embedding_dim=self.__args.slot_embedding_dim
        )

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def forward(self, text, seq_lens, n_predicts=None, forced_slot=None, forced_bio=None):
        """

        :param text:
        :param seq_lens:
        :param n_predicts:
        :param forced_slot:
        :param forced_bio:
        :return:
        """

        """
        without bert, bilstm plus self attention,
        """
        # word_tensor, _ = self.__embedding(text)
        # lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        # attention_hiddens = self.__attention(word_tensor, seq_lens)
        # hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)

        hiddens, cls_hidden = self.__transformer(text, seq_lens)
        pred_intent = self.__intent_Linear(self.__dropout(cls_hidden))

        pred_BIO = self.__bio_decoder(
            hiddens, seq_lens,
            forced_input=forced_bio
        )
        # pred_BIO = self.__border_Linear(self.__dropout(hiddens))

        _, feed_BIO = pred_BIO.topk(1)

        if forced_bio is not None:
            feed_BIO = forced_bio.unsqueeze(1).float()
        # print("类型：")
        # print(hiddens.type(), feed_BIO.type())
        # print("大小：")
        # print(hiddens.size(), feed_BIO.size())
        border_hidden = torch.mul(hiddens, feed_BIO.to(torch.float))

        new_hiddens = self.__attention(hiddens, border_hidden, seq_lens)
        # new_hiddens = self.__attention(hiddens,hiddens,seq_lens)  #self_attention

        total_hidden = torch.cat([new_hiddens, hiddens], dim=-1)

        # total_hidden = hiddens
        pred_slot = self.__slot_decoder(
            total_hidden, seq_lens,
            forced_input=forced_slot
        )
        # pred_slot =  self.__slot_Linear(self.__dropout(total_hidden))

        # print("##########")
        if n_predicts is None:
            if self.__args.single_intent:
                return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_BIO, dim=-1), F.log_softmax(pred_intent, dim=-1)
            else:
                return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_BIO, dim=-1), pred_intent
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            #如果是单意图直接返回top1
            if self.__args.single_intent:
                _, intent_index = pred_intent.topk(n_predicts, dim=1)
            else:
                intent_index = (torch.sigmoid(pred_intent) > self.__args.threshold).nonzero()

            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()



class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        # print("##############input_query",input_query.shape,input_key.shape,input_value.shape)
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ), dim=-1) / math.sqrt(self.__hidden_dim)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class AttentionQ(nn.Module):

    def __init__(self, input_dim,kv_dim ,hidden_dim, output_dim, dropout_rate):
        super(AttentionQ, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__kv_dim = kv_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__kv_dim, self.__kv_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def recover(self,input,seq_lens):
        start = 0
        batch_size,max_size = len(seq_lens),seq_lens[0]
        input_ori = torch.zeros(batch_size,max_size,input.size(-1))
        if torch.cuda.is_available():
            input_ori = input_ori.cuda()


        for i in range(len(seq_lens)):
            end = start + seq_lens[i]
            input_ori[i,:seq_lens[i]] = input[start:end]
            start = end
        return input_ori


    def forward(self, input_x,input_y, seq_lens):

        input_x = self.recover(input_x,seq_lens)
        input_y = self.recover(input_y,seq_lens)


        dropout_x = self.__dropout_layer(input_x)
        dropout_y = self.__dropout_layer(input_y)

        attention_x = self.__attention_layer(
            dropout_x, dropout_y, dropout_y
        )

        flat_x = torch.cat(
            [attention_x[i][:seq_lens[i], :] for
             i in range(0, len(seq_lens))], dim=0
        )
        return flat_x


class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        :param embedding_dim: if it's not None, the input and output are relevant.
        :param extra_dim: if it's not None, the decoder receives information tensors.
        """

        super(LSTMDecoder, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)

            self.__init_tensor = nn.Parameter(
                torch.randn(1, self.__embedding_dim),
                requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.
        if self.__extra_dim is not None and self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim + self.__embedding_dim
        elif self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        elif self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__embedding_dim
        else:
            lstm_input_dim = self.__input_dim
        self.lstm_input_dim = lstm_input_dim  ##################
        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1
        )
        self.__linear_layer = nn.Linear(
            self.__hidden_dim,
            self.__output_dim
        )

    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :param forced_input: is truth values of label, provided by teacher forcing.
        :param extra_input: comes from another decoder as information tensor.
        :return: is distribution of prediction labels.
        """

        # Concatenate information tensor if possible.
        if extra_input is not None:

            extra_input = extra_input.float()
            # print("@#@#",extra_input.shape,encoded_hiddens.shape,extra_input)
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=-1)

        else:
            input_tensor = encoded_hiddens

        output_tensor_list, sent_start_pos = [], 0
        if self.__embedding_dim is None or forced_input is not None:

            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]

                # Segment input hidden tensors.
                # print("input_tensor",input_tensor.shape,sent_start_pos,sent_end_pos)
                seg_hiddens = input_tensor[sent_start_pos: sent_end_pos, :]

                if self.__embedding_dim is not None and forced_input is not None:
                    if seq_lens[sent_i] > 1:
                        seg_forced_input = forced_input[sent_start_pos: sent_end_pos]
                        seg_forced_tensor = self.__embedding_layer(seg_forced_input).view(seq_lens[sent_i], -1)
                        seg_prev_tensor = torch.cat([self.__init_tensor, seg_forced_tensor[:-1, :]], dim=0)

                    else:
                        seg_prev_tensor = self.__init_tensor

                    # Concatenate forced target tensor.
                    # print("@!seg_hiddens",seg_hiddens.shape,seg_prev_tensor.shape)
                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor], dim=1)
                else:
                    combined_input = seg_hiddens
                dropout_input = self.__dropout_layer(combined_input)
                # print("###########combined_input",dropout_input.shape,seq_lens[sent_i],self.__input_dim ,self.__extra_dim,self.__embedding_dim)
                # print("dropout_input.view(1, seq_lens[sent_i], -1)",self.lstm_input_dim,seg_hiddens.shape,combined_input.shape,dropout_input.view(1, seq_lens[sent_i], -1).shape)
                lstm_out, _ = self.__lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))
                # linear_out = lstm_out.view(seq_lens[sent_i], -1)
                linear_out = self.__linear_layer(lstm_out.view(seq_lens[sent_i], -1))
                # print("#################lstm_out",lstm_out.shape,linear_out.shape)
                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
        else:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.__init_tensor

                # It's necessary to remember h and c state
                # when output prediction every single step.
                last_h, last_c = None, None

                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = input_tensor[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.__dropout_layer(combined_input).view(1, 1, -1)

                    if last_h is None and last_c is None:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                    else:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))
                    # lstm_out = lstm_out.view(1, -1)
                    lstm_out = self.__linear_layer(lstm_out.view(1, -1))

                    output_tensor_list.append(lstm_out)
                    _, index = lstm_out.topk(1, dim=1)
                    prev_tensor = self.__embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos

        return torch.cat(output_tensor_list, dim=0)