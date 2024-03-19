# coding=utf-8
import torch
import datetime
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer("[\s,'\.]", gaps=True)
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
from string import punctuation
import re
import math
import time
from datetime import timedelta
from tqdm import tqdm
import numpy as np
import random
from itertools import combinations
import torch.nn as nn

random.seed(2020)
from random import shuffle


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速


def get_indices(dataset):
    data_idx = list(range(len(dataset)))
    shuffle(data_idx)
    split_num = int(len(dataset) / 10)
    val_idx = data_idx[:split_num]
    test_idx = data_idx[split_num:split_num * 2]
    train_idx = data_idx[split_num * 2:]

    # sample_lt = []
    #
    # for i in test_idx:
    #     for api in np.where(dataset[i][3] == 1):
    #         print(dataset[i])
    #     continue

    return train_idx, val_idx, test_idx


def random_init(tensor, in_dim, out_dim):
    thresh = math.sqrt(6.0 / (in_dim + out_dim))
    if tensor is not None:
        try:
            tensor.data.uniform_(-thresh, thresh)
        except:
            nn.init.uniform_(tensor, a=-thresh, b=thresh)


from torch_scatter import scatter


def split_stack(features, index, relations, dim_size):
    """
    Official Stack accumulation function

    Parameters
    ----------
    features : tensor (relation * num_nodes) x features
        output of messge method in RGCLayer class
    index : tensor (edges)
        edge_index[0]
    relations : teonsor(edges)
        edge_type
    dim_size : tensor(num_nodes)
        input size (the number of nodes)

    Return
    ------
    stacked_out : tensor(relation * nodes x out_dim)
    """
    out_dim = features.shape[0]
    np_index = index.numpy()
    np_relations = relations.numpy()
    splited_features = torch.split(features, int(out_dim / 5), dim=1)

    stacked_out = []
    for r, feature in enumerate(splited_features):
        relation_only_r = torch.from_numpy(np.where(np_relations == r)[0])
        r_index = index[relation_only_r]
        r_feature = feature[relation_only_r]
        stacked_out.append(scatter(r_feature, r_index, dim_size=dim_size, reduce='add'))
        # stacked_out.append(scatter_('add', feature, index, dim_size=dim_size))

    stacked_out = torch.cat(stacked_out, 1)

    return stacked_out


def stack(features, index, relations, dim_size):
    """
    Stack accumulation function in RGCLayer.

    Parameters
    ----------
    features : tensor (relation * num_nodes)
        output of messge method in RGCLayer class
    index : tensor (edges)
        edge_index[0]
    relations : teonsor(edges)
        edge_type
    dim_size : tensor(num_nodes)
        input size (the number of nodes)

    Return
    ------
    out : tensor(relation * nodes x out_dim)
    """
    out = torch.zeros(dim_size * (torch.max(relations) + 1), features.shape[1])
    tar_idx = relations * dim_size + index
    out[tar_idx] = features
    # for feature, idx, relation in zip(features, index, relations):
    #     tar_idx = relation * dim_size + index
    #     out[tar_idx] = feature
    return out


from collections import defaultdict


def get_indices_withlt(dataset, api_freq, threshold=4):
    data_idx = list(range(len(dataset)))
    shuffle(data_idx)
    split_num = int(len(dataset) / 10)
    val_idx = data_idx[:split_num]
    test_idx = data_idx[split_num:split_num * 2]
    train_idx = data_idx[split_num * 2:]

    sample_lt = []

    counter = defaultdict(int)
    for i in train_idx:
        counter[len(torch.where(dataset[i][1] != 1)[0])] += 1
    sorted_dict = dict(sorted(counter.items()))
    print("train: mashup_ds_len:{0}".format(sorted_dict))
    draw_counter_pic(sorted_dict, 'Sorted dict visualization of train')

    counter = defaultdict(int)
    for i in val_idx:
        counter[len(torch.where(dataset[i][1] != 1)[0])] += 1
    sorted_dict = dict(sorted(counter.items()))
    print("val: mashup_ds_len:{0}".format(sorted_dict))
    draw_counter_pic(sorted_dict, 'Sorted dict visualization of val')

    counter = defaultdict(int)
    for i in test_idx:
        # if api_freq[i] == 1:
        #     sample_lt.append(i)
        # print("len:{0}".format(len(torch.where(dataset[i][1] != 1)[0])))
        counter[len(torch.where(dataset[i][1] != 1)[0])] += 1
        for api in np.where(dataset[i][3] == 1)[0]:
            flag = False
            if api_freq[api] <= threshold:
                sample_lt.append(i)
                flag = True
                # print(dataset[i])
            if flag:
                break

    print("length: {0}==> threshold: {1}".format(len(sample_lt), threshold))

    sorted_dict = dict(sorted(counter.items()))
    print("test: mashup_ds_len:{0}".format(sorted_dict))

    draw_counter_pic(sorted_dict, 'Sorted dict visualization of test')

    return train_idx, val_idx, test_idx, sample_lt


import matplotlib.pyplot as plt


def draw_counter_pic(sorted_dict, name):
    # 提取键和值，用于绘图
    keys = list(sorted_dict.keys())
    values = list(sorted_dict.values())

    # 绘制柱状图
    plt.bar(keys, values)

    # 添加坐标轴标签和标题
    plt.xlabel('key')
    plt.ylabel('value')
    plt.title(name)

    # 显示图形
    plt.show()


def get_time(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def tokenize(text):
    """
    tokenize
    :param input_str:
    :return:
    """
    if text == '' or text is None:
        return []
    text = text.lower().replace('\n', ' ')
    # 缩写替换
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    #
    r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    text = re.sub(r4, ' ', text)

    tokens = tokenizer.tokenize(text)
    # 词性还原
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # print(tokens)
    # 词干提取
    # tokens = [stemmer.stem(word) for word in tokens]
    # print(tokens)
    # 去除停用词
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(
                                device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


'''
def nodecount_for_DPP(steiner_trees):
    # construct similarity matrix
    weight_vector, similarity_matrix = create_similarity_matrix_nodecount(steiner_trees)
    # construct DPP kernel matrix
    kernel_matrix = weight_vector.reshape(
        (similarity_matrix.shape[0], 1)) * similarity_matrix * weight_vector.reshape(
        (1, similarity_matrix.shape[0]))

    return kernel_matrix, similarity_matrix


def weight_for_DPP(steiner_trees):
    # construct similarity matrix
    weight_vector, similarity_matrix = create_similarity_matrix_weight(steiner_trees)
    # construct DPP kernel matrix
    kernel_matrix = weight_vector.reshape(
        (similarity_matrix.shape[0], 1)) * similarity_matrix * weight_vector.reshape(
        (1, similarity_matrix.shape[0]))

    return kernel_matrix, similarity_matrix


def ratio_for_DPP(steiner_trees):
    # construct similarity matrix
    weight_vector, similarity_matrix = create_similarity_matrix_ratio(steiner_trees)
    # construct DPP kernel matrix
    kernel_matrix = weight_vector.reshape(
        (similarity_matrix.shape[0], 1)) * similarity_matrix * weight_vector.reshape(
        (1, similarity_matrix.shape[0]))

    return kernel_matrix, similarity_matrix


def adaptive_threshold(result_trees):
    for alist_i, alist_j in combinations(result_trees, 2):
        if set(alist_j.nodes) == set(alist_i.nodes):
            if alist_j in result_trees:
                result_trees.remove(alist_j)

    return result_trees


# def metric(steiner_trees, DPP_results, similarity_matrix, api_names, test_apis):
#     DPP_metric_object = DPP_metric(steiner_trees, DPP_results)
#     ILAD, ILMD = DPP_metric_object.compare_similarity(similarity_matrix)
#
#     return ILAD, ILMD


def run_ratio_steiner_nodecount(graph, keywords, api_categories, tree_num):
    anchor = time.time()
    ratiosteiner = RatioSteinerAlgorithm(graph, api_categories)
    result_trees = ratiosteiner.run(keywords)
    anchor_ratio = time.time()
    cost_run = anchor_ratio - anchor

    # agjust parameter Z
    if result_trees and len(result_trees) > tree_num:
        steiner_trees = adaptive_threshold(result_trees)
        kernel_matrix, similarity_matrix = nodecount_for_DPP(steiner_trees)
        DPP_object = DPP()
        DPP_results = DPP_object.run(kernel_matrix, tree_num)
        cost_DPP = time.time() - anchor_ratio
        cost = time.time() - anchor

        return steiner_trees, DPP_results, cost, cost_run, cost_DPP

    return [], [], 0, 0, 0


def run_ratio_steiner_weight(steiner_trees, tree_num):
    if steiner_trees:
        anchor = time.time()
        kernel_matrix, similarity_matrix = weight_for_DPP(steiner_trees)
        DPP_object = DPP()
        DPP_results = DPP_object.run(kernel_matrix, tree_num)
        cost_DPP = time.time() - anchor

        return steiner_trees, DPP_results, cost_DPP

    return [], [], 0


def run_ratio_steiner_ratio(steiner_trees, tree_num):
    if steiner_trees:
        anchor = time.time()
        kernel_matrix, similarity_matrix = ratio_for_DPP(steiner_trees)
        DPP_object = DPP()
        DPP_results = DPP_object.run(kernel_matrix, tree_num)
        cost_DPP = time.time() - anchor

        return steiner_trees, DPP_results, cost_DPP

    return [], [], 0


def get_api_names(api_names, nodes):
    """
    将api的索引列表转换为api name列表
    :param api_names:
    :param nodes:
    :return:
    """
    names = []
    for v in nodes:
        names.append(api_names[v])
    return names
'''


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            torch.save(model.state_dict(), self.path)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
