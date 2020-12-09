"""
@Author		:           Xiaoduo,Zhou
@StartTime	:           2019/11/10
@Filename	:           make_tensor.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/11/01
"""
import os
import pickle
from collections import defaultdict
import numpy as np

class Mask(object):
    def __init__(self,args):
        self._args = args


    def load_mask(self):
        domain_mask = {}
        dir_path = self._args.data_dir
        if  os.path.exists(dir_path):
            path = os.path.joint(dir_path,"domain_mask.pkl")
            if os.path.exists(path):
                f = open(path,'rb')
                domain_mask = pickle.load(f)
            else:
                self.make_mask(self.load_file(dir_path+"/train.txt"))

    def load_file(self,path):
        lines = []
        with open(path,'r',encoding='utf-8') as f:
            for line in  f.readlines():
                lines.append(line.strip())
        return lines

    def display_key(self,mask_dict):
        count = 0
        for key in mask_dict.keys():
            count += len(mask_dict[key])
            print(key,len(mask_dict[key]))
            list_value = set()
            for value in mask_dict[key]:
                value = value.replace("B-","").replace("I-","")
                list_value.add(value)
            print(list(list_value))
        print("total slots :",count)

    def make_mask(self,path,dict_path,pad_intent=None,pad_slot=None):
        path1 = path + "/train.txt"
        path2 = path + "/dev.txt"
        path3 = path + "/test.txt"

        slot_list_path = dict_path + "/slot_dict.txt"
        intent_list_path = dict_path + "/domain_dict.txt"

        save_tensor_path = path + "/domain_tensor.pkl"
        lines = mask.load_file(path1) + mask.load_file(path2) #+ mask.load_file(path3)
        slot_list = mask.load_file(slot_list_path)
        slots = [slot.split("	")[0] for slot in slot_list]
        
        big_slot_dict = dict()
        for slot in slots:   
            if '.' in slot :
                slot = slot.replace("B-","").replace("I-","")
                group = slot.split(".")
                if group[0] in big_slot_dict.keys():
                    big_slot_dict[group[0]].append(group[0]+"."+group[1] )
                else:
                    big_slot_dict[group[0]] = [group[0]+"."+group[1]]

        for key in big_slot_dict.keys():
            big_slot_dict[key] = list(set(big_slot_dict[key]))
        

        # intent_list = mask.load_file(intent_list_path)
        # intents =  [intent.split("	")[0] for intent in intent_list]

        # print("intents : ",intents)
        print("slots :",slots)
        # if pad_slot:
        #     slots = pad_slot + slots

        # if pad_intent:
        #     intents = pad_intent +intents

        # print("intents",intents)
        # print("slots",slots)
        # intent_idx = {intent:i for i,intent in enumerate(intents)}
        slot_dict ={slot:i for i,slot in enumerate(slots)}
        sen_slots = set()
        mask_dict = defaultdict(set)

        tensor_dict = {}
        for line in lines:
            line = line.strip()
            if line :
                Tags = line.split(" ")
                if(len(Tags) > 1):
                    sen_slots.add(Tags[-1])
                else:
                    
                    if '#' in line:
                        mutil_intents = line.split("#")
                        for intent in mutil_intents:
                            mask_dict[line] = mask_dict[line] | sen_slots
                            sen_slots = set()
                    else:
                        mask_dict[line] =  mask_dict[line]  |  sen_slots
                        sen_slots = set()

                    
                    # ###smp
                    # line = line.split("@")[0]
                    # mask_dict[line] =  mask_dict[line]  |  sen_slots
                    # sen_slots = set()
                    
        if pad_intent:
            for intent in pad_intent:
                mask_dict[intent] = slots

        for key in mask_dict.keys():

            # tensor = 100*np.ones(len(slots))
            tensor = np.zeros(len(slots))
            # tensor = np.ones(len(slots))

            expand_slot = []
            for slot in list(mask_dict[key]):
                if 'B-' in slot or 'I-' in slot :
                    new_slot = slot.replace("B-","").replace("I-","")

                    if "." in new_slot:
                        group_key = new_slot.split(".")[0]
                        group_slots = big_slot_dict[group_key]
                        for gslot in group_slots:
                            expand_slot.append('B-'+gslot)
                            expand_slot.append('I-'+gslot)

                    # if 'fromloc' in new_slot or 'toloc' in new_slot :
                    #     new_slot  = new_slot.replace("fromloc","").replace("toloc","")

                    #     for group_key in ['fromloc','toloc']:
                    #         group_slots = big_slot_dict[group_key]
                    #         for gslot in group_slots:
                    #             expand_slot.append('B-'+gslot)
                    #             expand_slot.append('I-'+gslot)
                    else:
                        expand_slot.append('B-'+new_slot)
                        expand_slot.append('I-'+new_slot)
                   
                    
            expand_slot = list(set(expand_slot))

            mask_dict[key] = expand_slot + ['O']
            # if pad_slot:
            #     mask_dict[key] = list(mask_dict[key]) + pad_slot
            
            #print(mask_dict[key])


            list_id = [] # [slot_dict[slot] for slot in mask_dict[key]]
            for slot in mask_dict[key] :
                if slot in slot_dict.keys():
                    list_id.append(slot_dict[slot])
            list_id = sorted(list_id)
            list_id = np.array(list_id)
            #print(list_id )
            #print("#"*100)

            tensor[list_id] = 1
            if '#' not in key:
                # tensor_dict[intent_idx[key]] = tensor
                tensor_dict[key] = tensor

        self.display_key(mask_dict)

        f = open(save_tensor_path,'wb')
        pickle.dump(tensor_dict,f)

        # for key in big_slot_dict.keys():
        #     print(key)
        #     print(big_slot_dict[key])
        f = open(save_tensor_path, 'rb')
        mask_tensor = pickle.load(f)
        print(mask_tensor)

        # for key in tensor_dict.keys():
        #     print(key)
        #     print(tensor_dict[key])
        #     print("--"*20)
    
    def my_get_intent_slot(self, lines):
        texts, slots, intents = [], [], []
        text, slot = [], []

        for line in lines:
            items = line.strip().split()
            if len(items) == 1:
                texts.append(text)
                slots.append(slot)
                intents.append(items)

                # clear buffer lists.
                text, slot = [], []

            elif len(items) == 2:
                text.append(items[0].strip())
                slot.append(items[1].strip())

        return texts, slots, intents

    def my_make_mask(self, path, dict_path, pad_intent=None, pad_slot=None):
        path1 = path + "/train.txt"
        path2 = path + "/dev.txt"
        path3 = path + "/test.txt"

        slot_list_path = dict_path + "/slot_dict.txt"
        intent_list_path = dict_path + "/intent_dict.txt"

        save_tensor_path = path + "/domain_tensor.pkl"

        slot_list = mask.load_file(slot_list_path)
        slots = [slot.split("\t")[0] for slot in slot_list]
        if isinstance(pad_slot, list):
            slots += pad_slot
        intent_list = mask.load_file(intent_list_path)
        intents = [intent.split("\t")[0] for intent in intent_list]
        if isinstance(pad_intent, list):
            intents += pad_intent
        slots_dict = {}
        for slot in slots:
            if slot not in slots_dict:
                slots_dict[slot] = len(slots_dict)

        for k, v in slots_dict.items():
            print(k, v)

        intent_dict = {}
        for no, intent in enumerate(intents):
            if intent not in intent_dict:
                intent_dict[intent] = np.array([0] * len(slots))
        print(intents)
        for k, v in intent_dict.items():
            print(k, v)
        lines = mask.load_file(path1) + mask.load_file(path2)  # + mask.load_file(path3)
        texts, slots, intents = mask.my_get_intent_slot(lines)
        dict_intent_slots = {}
        for slot_list, intent_list in zip(slots, intents):
            intent = intent_list[0]
            if intent not in dict_intent_slots:
                dict_intent_slots[intent] = []
            for slot in slot_list:
                if slot not in dict_intent_slots[intent]:
                    dict_intent_slots[intent].append(slot)
        for k, v in dict_intent_slots.items():
            print(k, v)
        for k, v in intent_dict.items():
            temp_slot_list = dict_intent_slots[k]
            for temp_slot in temp_slot_list:
                v[slots_dict[temp_slot]] = 1
        for k, v in intent_dict.items():
            print(k, v)
        with open(save_tensor_path, 'wb') as f:
            pickle.dump(intent_dict, f)
        print(list(intent_dict.keys()))
    def make_smp_mask(self,path,dict_path,pad_intent=None,pad_slot=None):
        path1 = path + "/train.txt"
        path2 = path + "/dev.txt"
        path3 = path + "/test.txt"

        slot_list_path = dict_path + "/slot_dict.txt"
        intent_list_path = dict_path + "/domain_dict.txt"

        save_tensor_path = path + "/domain_tensor.pkl"
        lines = mask.load_file(path1) + mask.load_file(path2) + mask.load_file(path3)
        slot_list = mask.load_file(slot_list_path)
        slots = [slot.split("	")[0] for slot in slot_list]
        
        big_slot_dict = dict()
        for slot in slots:   
            if '.' in slot :
                slot = slot.replace("B-","").replace("I-","")
                group = slot.split(".")
                if group[0] in big_slot_dict.keys():
                    big_slot_dict[group[0]].append(group[0]+"."+group[1] )
                else:
                    big_slot_dict[group[0]] = [group[0]+"."+group[1]]

        for key in big_slot_dict.keys():
            big_slot_dict[key] = list(set(big_slot_dict[key]))
        

        # intent_list = mask.load_file(intent_list_path)
        # intents =  [intent.split("	")[0] for intent in intent_list]

        # print("intents : ",intents)
        print("slots :",slots)
        # if pad_slot:
        #     slots = pad_slot + slots

        # if pad_intent:
        #     intents = pad_intent +intents

        # print("intents",intents)
        # print("slots",slots)
        # intent_idx = {intent:i for i,intent in enumerate(intents)}
        slot_dict ={slot:i for i,slot in enumerate(slots)}
        sen_slots = set()
        mask_dict = defaultdict(set)

        tensor_dict = {}
        for line in lines:
            line = line.strip()
            if line :
                Tags = line.split(" ")
                if(len(Tags) > 1):
                    sen_slots.add(Tags[-1])
                else:
                    '''
                    if '#' in line:
                        mutil_intents = line.split("#")
                        for intent in mutil_intents:
                            mask_dict[line] = mask_dict[line] | sen_slots
                            sen_slots = set()
                    else:
                        mask_dict[line] =  mask_dict[line]  |  sen_slots
                        sen_slots = set()

                    '''
                    ###smp
                    line = line.split("@")[0]
                    mask_dict[line] =  mask_dict[line]  |  sen_slots
                    sen_slots = set()
                    
        if pad_intent:
            for intent in pad_intent:
                mask_dict[intent] = slots

        for key in mask_dict.keys():

            # tensor = 100*np.ones(len(slots))
            tensor = np.zeros(len(slots))
            # tensor = np.ones(len(slots))

            expand_slot = []
            for slot in list(mask_dict[key]):
                if 'B-' in slot or 'I-' in slot :
                    new_slot = slot.replace("B-","").replace("I-","")

                    if "." in new_slot:
                        group_key = new_slot.split(".")[0]
                        group_slots = big_slot_dict[group_key]
                        for gslot in group_slots:
                            expand_slot.append('B-'+gslot)
                            expand_slot.append('I-'+gslot)

                    # if 'fromloc' in new_slot or 'toloc' in new_slot :
                    #     new_slot  = new_slot.replace("fromloc","").replace("toloc","")

                    #     for group_key in ['fromloc','toloc']:
                    #         group_slots = big_slot_dict[group_key]
                    #         for gslot in group_slots:
                    #             expand_slot.append('B-'+gslot)
                    #             expand_slot.append('I-'+gslot)
                    else:
                        expand_slot.append('B-'+new_slot)
                        expand_slot.append('I-'+new_slot)
                   
                    
            expand_slot = list(set(expand_slot))

            mask_dict[key] = expand_slot + ['O']
            # if pad_slot:
            #     mask_dict[key] = list(mask_dict[key]) + pad_slot
            
            #print(mask_dict[key])


            list_id = [] # [slot_dict[slot] for slot in mask_dict[key]]
            for slot in mask_dict[key] :
                if slot in slot_dict.keys():
                    list_id.append(slot_dict[slot])
            list_id = sorted(list_id)
            list_id = np.array(list_id)
            #print(list_id )
            #print("#"*100)

            tensor[list_id] = 1
            if '#' not in key:
                # tensor_dict[intent_idx[key]] = tensor
                tensor_dict[key] = tensor

        self.display_key(mask_dict)

        f = open(save_tensor_path,'wb')
        pickle.dump(tensor_dict,f)

        # for key in big_slot_dict.keys():
        #     print(key)
        #     print(big_slot_dict[key])
        f = open(save_tensor_path, 'rb')
        mask_tensor = pickle.load(f)
        print(mask_tensor)

        # for key in tensor_dict.keys():
        #     print(key)
        #     print(tensor_dict[key])
        #     print("--"*20)
       


import argparse
parser = argparse.ArgumentParser()

# # Training parameters.
# parser.add_argument('--data_dir', '-dd', type=str, default='data/smp')
# parser.add_argument('--dict_dir', '-dtd', type=str, default='save/smp_stackbert_ori/alphabet')

parser.add_argument('--data_dir', '-dd', type=str, default='data/SNIPS')
parser.add_argument('--dict_dir', '-dtd', type=str, default='save/total/border/SNIPS_uncased1/alphabet')

# parser.add_argument('--data_dir', '-dd', type=str, default='data/atis')
# parser.add_argument('--dict_dir', '-dtd', type=str, default='save/atis_border_ori_1/alphabet')

# parser.add_argument('--data_dir', '-dd', type=str, default='data/snip_dataset')
# parser.add_argument('--dict_dir', '-dtd', type=str, default='save/snips_incremental_jointbert_ori/alphabet')

args = parser.parse_args()
mask = Mask(args)

# base = os.path.join(os.path.dirname(__file__), '..')
base = ".."
path = base + "/" + args.data_dir
dict_path = base + "/" + args.dict_dir
if 'smp' in dict_path:
    mask.make_smp_mask(path,dict_path)#,pad_intent=['<pad>','<unk>'],pad_slot=['<pad>','<unk>'])
else:
    mask.my_make_mask(path,dict_path)#,pad_intent=['<pad>','<unk>'],pad_slot=['<pad>','<unk>'])
