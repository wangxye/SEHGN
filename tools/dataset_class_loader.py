import random

from torch.utils.data import Dataset
import sys
import os
import json

curPath = os.path.abspath(os.path.dirname('__file__'))
# rootPath = os.path.split(curPath)[0]
rootPath = curPath
sys.path.append(rootPath)
import torch
from tools.utils import tokenize, draw_counter_pic
from torchtext.data import Field
from torchtext.vocab import Vectors
from random import randint, choice
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import copy
from collections import defaultdict


class MashupDataset(Dataset):
    def __init__(self, all_api=False, api_threshold=1647):
        super().__init__()
        with open(rootPath + '/data/mashup_name.json', 'r') as f:
            self.name = json.load(f)
        with open(rootPath + '/data/mashup_description.json', 'r') as f:
            self.description = json.load(f)
        with open(rootPath + '/data/mashup_category.json', 'r') as f:
            self.category = json.load(f)
        with open(rootPath + '/data/mashup_used_api.json', 'r') as f:
            self.used_api = json.load(f)
        with open(rootPath + '/data/category_list.json', 'r') as f:
            category_list = json.load(f)

        with open(rootPath + '/data/api_category.json', 'r') as f:
            api_category = json.load(f)
        with open(rootPath + '/data/api_description.json', 'r') as f:
            api_description = json.load(f)
        with open(rootPath + '/data/api_name.json', 'r') as f:
            api_list_all = json.load(f)

        print("api list all:{0}".format(len(api_list_all)))
        self.api_category = []
        self.api_description = []

        self.api2category = {}
        self.api2description = {}
        for api in api_list_all:
            self.api2category[api] = api_category[api_list_all.index(api)]
            self.api2description[api] = api_description[api_list_all.index(api)]

        self.API_THRESHOLD = api_threshold
        if all_api:
            with open(rootPath + '/data/used_api_list.json', 'r') as f:
                api_list = json.load(f)

            for i in api_list_all:
                # print(i)
                if len(api_list) < self.API_THRESHOLD and i not in api_list:
                    api_list.append(i)
                # else:
                #     break
            print("{0}==>{1}".format(len(api_list), self.API_THRESHOLD))
        else:
            with open(rootPath + '/data/used_api_list.json', 'r') as f:
                api_list = json.load(f)

        category2api = {}
        for api in api_list:
            self.api_category.append(self.api2category[api])
            self.api_description.append(self.api2description[api])

            for cat in self.api2category[api]:
                if cat not in category2api:
                    category2api[cat] = []

                category2api[cat].append(api)

        self.num_api = len(api_list)
        self.num_mashup = len(self.used_api)
        self.num_category = len(category_list)
        self.category_mlb = MultiLabelBinarizer()
        self.category_mlb.fit([category_list])
        self.used_api_mlb = MultiLabelBinarizer()
        self.used_api_mlb.fit([api_list])

        self.des_lens = []
        self.category_token = []

        self.mashup_api = {}
        self.mashup_api_matrix = np.zeros((self.num_mashup, self.num_api), dtype='float32')

        self.api_api_compatibility_matrix = np.zeros((self.num_api, self.num_api), dtype='float32')
        self.api_api_affinity_matrix = np.zeros((self.num_api, self.num_api), dtype='float32')

        mashup_api_link = 0
        for i in range(len(self.used_api)):
            self.mashup_api.setdefault(i, {})
            api_list = self.used_api_mlb.transform([self.used_api[i]])
            api_ids = np.where(api_list[0] == 1)

            for j in api_ids[0]:
                mashup_api_link += 1
                self.mashup_api[i][j] = 1
                self.mashup_api_matrix[i][j] = 1

            for i in api_ids[0]:
                for j in api_ids[0]:
                    self.api_api_compatibility_matrix[i][j] += 1
                    self.api_api_compatibility_matrix[j][i] += 1

        for cat in category2api.keys():
            api_list = self.used_api_mlb.transform([category2api[cat]])
            api_ids = np.where(api_list[0] == 1)
            for i in api_ids[0]:
                for j in api_ids[0]:
                    if i != j:
                        self.api_api_affinity_matrix[i][j] += 1
                        self.api_api_affinity_matrix[j][i] += 1

        print("Number of composition links between APIs and Mashups: {0}".format(mashup_api_link))
        print("with category in api-api graph params...")
        self.categ = copy.deepcopy(self.category)
        # self.des = []
        self.des = copy.deepcopy(self.description)
        for des in self.description:
            self.des_lens.append(len(des) if len(des) < 50 else 50)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        description = self.description[index]
        category_tensor = torch.tensor(self.category_mlb.transform([self.category[index]]), dtype=torch.long).squeeze()
        used_api_tensor = torch.tensor(self.used_api_mlb.transform([self.used_api[index]]), dtype=torch.long).squeeze()
        des_len = torch.tensor(self.des_lens[index])
        category_token = torch.LongTensor(self.category_token[index])
        # category_token = self.category_token[index]
        return torch.tensor(index).long(), torch.tensor(
            description).long(), category_tensor, used_api_tensor, des_len, category_token, \
            # self.des[index]


class ApiDataset(Dataset):
    def __init__(self, all_api=False, api_threshold=1647):
        super().__init__()
        with open(rootPath + '/data/api_name.json', 'r') as f:
            name = json.load(f)
        with open(rootPath + '/data/api_description.json', 'r') as f:
            description = json.load(f)
        with open(rootPath + '/data/api_category.json', 'r') as f:
            category = json.load(f)
        with open(rootPath + '/data/category_list.json', 'r') as f:
            category_list = json.load(f)
        with open(rootPath + '/data/mashup_name.json', 'r') as f:
            self.mashup = json.load(f)
        with open(rootPath + '/data/used_api_list.json', 'r') as f:
            used_api_list = json.load(f)

        self.API_THRESHOLD = api_threshold

        if all_api:
            '''
            self.name = name
            self.description = description
            self.category = category
            self.used_api = []
            for api in self.name:
                self.used_api.append([api])
            '''
            self.name = used_api_list
            self.description = []
            self.category = []
            self.used_api = []

            if len(self.name) < self.API_THRESHOLD:
                for api in name:
                    if len(self.name) < self.API_THRESHOLD and api not in self.name:
                        self.name.append(api)

            print("{0}==>{1}".format(len(self.name), self.API_THRESHOLD))

            for api in self.name:
                self.description.append(description[name.index(api)])
                self.category.append(category[name.index(api)])
                self.used_api.append([api])

        else:
            self.name = used_api_list
            self.description = []
            self.category = []
            self.used_api = []
            for api in self.name:
                self.description.append(description[name.index(api)])
                self.category.append(category[name.index(api)])
                self.used_api.append([api])

        self.num_category = len(category_list)
        self.num_api = len(used_api_list)
        self.category_mlb = MultiLabelBinarizer()
        self.category_mlb.fit([category_list])
        self.used_api_mlb = MultiLabelBinarizer()
        self.used_api_mlb.fit([used_api_list])
        self.des_lens = []
        self.category_token = []

        self.categ = copy.deepcopy(self.category)
        self.des = copy.deepcopy(self.description)
        # self.des = []

        for des in self.description:
            self.des_lens.append(len(des) if len(des) < 50 else 50)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        description = self.description[index]
        category_tensor = torch.tensor(self.category_mlb.transform([self.category[index]]), dtype=torch.long).squeeze()
        used_api_tensor = torch.tensor(self.used_api_mlb.transform([self.name[index]]), dtype=torch.long).squeeze()
        des_len = torch.tensor(self.des_lens[index])
        category_token = torch.LongTensor(self.category_token[index])

        return torch.tensor(index).long(), \
               torch.tensor(description).long(), \
               category_tensor, \
               used_api_tensor, \
               des_len, category_token, \
            # self.des[index]

class TextDataset:
    def __init__(self, mul=1, is_random=False):
        cache = '.vec_cache'
        if not os.path.exists(cache):
            os.mkdir(cache)
        USED_APIS_LENS = 1647
        Multiple = mul
        self.API_THRESHOLD = USED_APIS_LENS * Multiple
        self.mashup_ds = MashupDataset(all_api=True, api_threshold=self.API_THRESHOLD)
        self.api_ds = ApiDataset(all_api=True, api_threshold=self.API_THRESHOLD)
        self.max_vocab_size = 10000
        self.max_doc_len = 50
        self.vectors = Vectors(name=rootPath + '/tools/glove/glove.6B.200d.txt', cache=cache)
        self.field = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=self.max_doc_len)

        self.field.build_vocab(self.mashup_ds.description, self.api_ds.description, vectors=self.vectors, min_freq=1,
                               max_size=self.max_vocab_size)

        self.random_seed = 2020
        self.num_category = self.mashup_ds.num_category
        self.num_mashup = len(self.mashup_ds)
        # self.num_api = len(self.api_ds)
        self.num_api = self.mashup_ds.num_api
        print(self.num_api)
        self.vocab_size = len(self.field.vocab)
        self.embed = self.field.vocab.vectors
        self.embed_dim = self.vectors.dim
        self.des_lens = []
        self.word2id(is_random)
        self.tag2feature()

    def word2id(self, is_random):

        counter = defaultdict(int)
        for i, des in enumerate(self.mashup_ds.description):
            tokens = [self.field.vocab.stoi[x] for x in des]

            counter[len(tokens)] += 1

            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.mashup_ds.description[i] = tokens

        sorted_dict = dict(sorted(counter.items()))
        print("mashup_ds_len:{0}".format(sorted_dict))
        draw_counter_pic(sorted_dict, 'Sorted dict visualization of mashup')

        counter = defaultdict(int)
        for i, des in enumerate(self.api_ds.description):
            tokens = [self.field.vocab.stoi[x] for x in des]

            counter[len(tokens)] += 1

            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.api_ds.description[i] = tokens

        sorted_dict = dict(sorted(counter.items()))
        print("api_ds_len:{0}".format(sorted_dict))
        draw_counter_pic(sorted_dict, 'Sorted dict visualization of api')


    def tag2feature(self):
        for i, category in enumerate(self.mashup_ds.category):
            tokens = [self.field.vocab.stoi[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.mashup_ds.category_token.append(tokens)

        for i, category in enumerate(self.api_ds.category):
            tokens = [self.field.vocab.stoi[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.api_ds.category_token.append(tokens)


if __name__ == '__main__':
    ds = TextDataset()
