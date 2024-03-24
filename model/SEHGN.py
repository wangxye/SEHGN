# _*_ coding:utf-8 _*_
"""
@Time     : 2023/08/27 16:14
@Author   : Wangxuanye
@File     : SEHGN.py
@Project  : SEHGN
@Software : PyCharm
@License  : (C)Copyright 2018-2028, Taogroup-NLPR-CASIA
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/08/27 16:14        1.0             None
"""
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

print(torch.cuda.is_available())
print(torch.__version__)
curPath = os.path.abspath(os.path.dirname('__file__'))
# rootPath = os.path.split(curPath)[0]
rootPath = curPath
sys.path.append(rootPath)
from tools.dataset_class_loader import *
from tools.utils import *
from tools.metric import *
import scipy.sparse as sp
import torch.nn.functional as f

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse

parser = argparse.ArgumentParser(description='SEHGN')
parser.add_argument('--lr', type=float, help='input learning rate', default=1e-3)
parser.add_argument('--multi', type=float, help='input Effective API proportion', default=1)
parser.add_argument('--latent', type=int, help='input Mashup/API latent feature', default=128)
parser.add_argument('--L2', type=float, help='input L2 regularization', default=1e-7)
parser.add_argument('--lt_threshold', type=int, help='input long tail threshold', default=4)
args = parser.parse_args()


class SEHGNConfig(object):
    def __init__(self, ds_config):
        self.model_name = 'SEHGN-lr{0}-x{1}-withcat-dim_{2}-l2_{3}'.format(
            args.lr,
            args.multi,
            args.latent,
            args.L2)
        self.embed_dim = ds_config.embed_dim
        self.max_doc_len = ds_config.max_doc_len
        self.num_category = ds_config.num_category

        self.feature_dim = args.latent
        self.num_kernel = 256
        self.dropout = 0.2

        self.kernel_size = [2, 4, 8, 16]
        self.num_mashup = ds_config.num_mashup
        self.num_api = ds_config.num_api
        self.vocab_size = ds_config.vocab_size
        self.embed = ds_config.embed

        print("{0}==>{1}".format(self.num_mashup, self.num_api))
        self.ds = ds_config

        self.api_tag_embed = torch.zeros(len(ds_config.api_ds), ds_config.api_ds.num_category)
        for i, api in enumerate(ds_config.api_ds):
            self.api_tag_embed[i] = api[2]

        self.category_freq, self.api_freq = self.get_freq_matrix()
        self.api_freq = self.api_freq.numpy()[0]
        print("api_freq:{0}".format(self.api_freq))

        threshold = args.lt_threshold
        self.popular_items = [item for item in range(len(self.api_freq)) if
                              self.api_freq[item] > threshold]

        # 128
        self.lantent_dim = args.latent
        self.K = 3

        self.lr = args.lr
        self.L2 = args.L2

        self.batch_size = 128
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.model_name)
        print(self.device)

    def get_freq_matrix(self):
        category_freq = torch.zeros(1, len(self.ds.mashup_ds.category_mlb.classes_))
        api_freq = torch.zeros(1, len(self.ds.mashup_ds.used_api_mlb.classes_))
        for category in self.ds.mashup_ds.category:
            category_freq += torch.tensor(self.ds.mashup_ds.category_mlb.transform([category])).squeeze()
        for api in self.ds.mashup_ds.used_api:
            api_freq += torch.tensor(self.ds.mashup_ds.used_api_mlb.transform([api])).squeeze()
        return category_freq, api_freq


# MessagePassing
class SEHGN(nn.Module):
    def __init__(self, config, idxs):
        super(SEHGN, self).__init__()

        if config.embed is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embed, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)

        self.num_mashup = config.num_mashup
        self.num_api = config.num_api

        self.sc_convs = nn.ModuleList([
            nn.Sequential(SelfAttention(dim_embedding=config.embed_dim, dim_qk=config.num_kernel, dim_v=h),
                          nn.Sigmoid(),
                          nn.MaxPool1d(kernel_size=h))
            for h in config.kernel_size
        ])

        self.sc_fcl = nn.Linear(in_features=config.embed_dim,
                                out_features=config.num_api)
        
        self.fc = nn.Linear(config.embed_dim, config.num_api)

        self.api_api_compatibility_graph = config.ds.mashup_ds.api_api_compatibility_matrix
        self.api_api_affinity_graph = config.ds.mashup_ds.api_api_affinity_matrix

        self.mashup_api_graph = np.zeros((self.num_mashup, self.num_api), dtype='float32')
        self.mashup_api_graph[idxs] = config.ds.mashup_ds.mashup_api_matrix[idxs]

        self.graph_mat = self.load_edge(self.mashup_api_graph)
        print("-" * 20)

        self.api_api_compatibility_mat = self.load_edge(self.api_api_compatibility_graph, is_graph=False)
        self.api_api_affinity_mat = self.load_edge(self.api_api_affinity_graph, is_graph=False)

        self.api_api_compatibility_weight = nn.Parameter(torch.rand(1, 1))
        self.api_api_affinity_weight = nn.Parameter(torch.rand(1, 1))

        # self.att_fc = nn.Linear(config.feature_dim * 2, 1)
        self.softmax = nn.Softmax(dim=-1)

        self.K = config.K

        self.api_tag_embed = nn.Embedding.from_pretrained(config.api_tag_embed, freeze=True)
        self.api_tag_layer = nn.Linear(in_features=config.num_category, out_features=config.feature_dim)

        ''' '''
        self.api_sc_convs = nn.ModuleList([
            nn.Sequential(SelfAttention(dim_embedding=config.embed_dim, dim_qk=config.num_kernel, dim_v=h),
                          nn.Sigmoid(),
                          nn.MaxPool1d(kernel_size=h))
            for h in config.kernel_size
        ])

        self.api_sc_output = nn.Linear(in_features=config.embed_dim,
                                       out_features=config.feature_dim)

        self.users_emb = nn.Embedding(
            num_embeddings=config.num_mashup, embedding_dim=config.lantent_dim)  # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=config.num_api, embedding_dim=config.lantent_dim)  # e_i^0

        self.init_weight()

        self.fic_fcl = nn.Linear(config.feature_dim * 2, config.feature_dim)
        self.api_api_fcl = nn.Linear(config.feature_dim * 2, config.feature_dim)
        self.fusion_layer = nn.Linear(config.num_api * 2, config.num_api)
        self.api_task_layer = nn.Linear(config.num_api, config.num_api)

        self.mashup_api_level_dropout = nn.Dropout(config.dropout, True)
        self.api_api_compatibility_level_dropout = nn.Dropout(config.dropout, True)
        self.api_api_affinity_level_dropout = nn.Dropout(config.dropout, True)

        self.logistic = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def load_edge(self, graph, is_graph=True):
        if is_graph:
            n_nodes = config.num_mashup + config.num_api
        else:
            n_nodes = config.num_api + config.num_api

        edge_index = [[], [], []]
        for i in range(graph.shape[0]):
            api_list = np.where(graph[i] > 0)
            for j in api_list[0]:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[2].append(graph[i][j])

        if is_graph:
            tmp_adj = sp.csr_matrix(
                (np.array(edge_index[2]), (np.array(edge_index[0]), np.array(edge_index[1]) + config.num_mashup)),
                shape=(n_nodes, n_nodes))
        else:
            tmp_adj = sp.csr_matrix(
                (np.array(edge_index[2]), (np.array(edge_index[0]), np.array(edge_index[1]) + config.num_api)),
                shape=(n_nodes, n_nodes))

        adj_mat = tmp_adj + tmp_adj.T

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        # normalize by user counts
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        # normalize by item counts
        normalized_adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # convert to torch sparse matrix
        adj_mat_coo = normalized_adj_matrix.tocoo()

        values = adj_mat_coo.data
        indices = np.vstack((adj_mat_coo.row, adj_mat_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        # v = torch.HalfTensor(values)
        shape = adj_mat_coo.shape
        #  .cuda()  .cuda()
        return torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(config.device)

    def init_weight(self):
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def semantic_represent(self, des, sc_convs):
        api_embed = self.embedding(des)
        # api_embed = api_embed.permute(0, 2, 1)
        e = [conv(api_embed) for conv in sc_convs]
        e = torch.cat(e, dim=2)
        e = e.view(e.size(0), -1)
        return e

    def transform_semantic_represent(self, des, encoder):
        api_embed = self.embedding(des)
        e = encoder(api_embed)
        return e

    def forward(self, mashup_des, user_indices, api_des, test=False):
        # api semantic component
        api_embed = self.semantic_represent(api_des, self.sc_convs)
        api_sc = self.api_sc_output(api_embed)
        api_sc = self.tanh(api_sc)
        api_sc = api_sc.permute(1, 0)

        # api tag layer
        api_tag_value = self.api_tag_layer(self.api_tag_embed.weight)
        api_tag_value = api_tag_value.permute(1, 0)
        api_tag_value = self.tanh(api_tag_value)

        # semantic component
        u_embed = self.semantic_represent(mashup_des, self.sc_convs)
        u_sc = self.sc_fcl(u_embed)

        # feature interaction component
        all_users, all_items = self.graph_comupte(graph=self.graph_mat, mess_dropout=self.mashup_api_level_dropout,test=test)
        all_users_api_comp, all_items_api_comp = self.graph_comupte(graph=self.api_api_compatibility_mat, mess_dropout=self.api_api_compatibility_level_dropout, test=test,
                                                                    is_graph=False)
        all_users_api_affi, all_items_api_affi = self.graph_comupte(graph=self.api_api_affinity_mat, mess_dropout=self.api_api_affinity_level_dropout, test=test
                                                                    , is_graph=False)

        api_linear_feature = torch.mul(api_sc, api_tag_value)

        mashup_api_matrix = torch.matmul(all_users, all_items.T)

        all_items_api = (
                                self.api_api_compatibility_weight * all_items_api_comp + self.api_api_affinity_weight * all_items_api_affi) / (
                                self.api_api_compatibility_weight + self.api_api_affinity_weight)

        all_items_api = self.fic_fcl(torch.cat((all_items_api, api_linear_feature.T), dim=1))
        apis_compatibility_matrix = torch.matmul(all_items_api, all_items_api.T)

        pre_matrix = torch.matmul(mashup_api_matrix, apis_compatibility_matrix)
        pre_matrix = f.normalize(pre_matrix, p=2, dim=1)

        u_fic = torch.index_select(pre_matrix, 0, user_indices)
        # fusion layer
        u_mmf = self.fusion_layer(torch.cat((u_sc, u_fic), dim=1))
        y_m = self.api_task_layer(u_mmf)

        return self.logistic(y_m)

    def graph_comupte(self, graph, mess_dropout, test, is_graph=True):
        if is_graph:
            all_emb = torch.cat([self.users_emb.weight, self.items_emb.weight])  # E^0
            # graph = self.graph_mat
        else:
            all_emb = torch.cat([self.items_emb.weight, self.items_emb.weight])
            # graph = self.api_mat

        layer_embeddings = [all_emb]

        for i in range(self.K):
            all_emb = torch.sparse.mm(graph, all_emb)
            if not test: # !!! important
                all_emb = mess_dropout(all_emb)
            layer_embeddings.append(all_emb)
        layer_embeddings = torch.stack(layer_embeddings, dim=1)

        final_embeddings = layer_embeddings.mean(dim=1)  # output is mean of all layers
        if is_graph:
            all_users, all_items = torch.split(final_embeddings, [self.num_mashup, self.num_api])
        else:
            all_users, all_items = torch.split(final_embeddings, [self.num_api, self.num_api])
        
        return all_users, all_items


def adjust_learning_rate(optimizer, epoch):
    # 37 40
    modellrnew = config.lr * (0.1 ** (epoch // 30))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


class Train(object):
    def __init__(self, input_model, input_config, train_iter, test_iter, val_iter, sample_lt_iter, log, input_ds,
                 model_path=None):
        self.model = input_model
        self.config = input_config
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.sample_lt_iter = sample_lt_iter
        self.api_cri = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=config.L2)
        self.epoch = 100
        self.top_k_list = [1, 5, 10, 15, 20, 25, 30]
        self.log = log
        self.ds = input_ds

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = curPath + '/model/checkpoint/%s.pth' % self.config.model_name
        self.early_stopping = EarlyStopping(patience=40, path=self.model_path)
        print(self.early_stopping.patience)
        self.api_des = torch.LongTensor(self.ds.api_ds.description).to(self.config.device)

    def train(self):

        data_iter = self.train_iter
        self.model.train()
        print('Start training ...')

        for epoch in range(self.epoch):
            # print(config.model_name)
            # adjust_learning_rate(self.optim, epoch)
            self.model.train()
            api_loss = []

            for batch_idx, batch_data in enumerate(data_iter):
                # batch_data: index, des, category_performance, used_api
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                api_target = batch_data[3].float().to(self.config.device)
                # keywords = batch_data[7]

                self.optim.zero_grad()
                api_pred = self.model(des, index, self.api_des)
                api_loss_ = self.api_cri(api_pred, api_target)

                api_loss_.backward()
                self.optim.step()
                api_loss.append(api_loss_.item())

            api_loss = np.average(api_loss)

            info = '[Epoch:%s] ApiLoss:%s ' % (epoch + 1, api_loss.round(6))
            print(info)
            self.log.write(info + '\n')
            self.log.flush()
            val_loss = self.evaluate()
            self.early_stopping(float(val_loss), self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def evaluate(self, test=False, sample=False):
        if test:
            if sample:
                data_iter = self.sample_lt_iter
                label = 'Sample Test Long-tailed'
                print("Start Sample Test lt")
            else:
                data_iter = self.test_iter
                label = 'Test'
                print('Start testing ...')

        else:
            data_iter = self.val_iter
            label = 'Evaluate'
        self.model.eval()

        # API
        ndcg_a = np.zeros(len(self.top_k_list))
        recall_a = np.zeros(len(self.top_k_list))
        ap_a = np.zeros(len(self.top_k_list))
        pre_a = np.zeros(len(self.top_k_list))
        pb_a = np.zeros(len(self.top_k_list))

        api_loss = []
        num_batch = len(data_iter)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                category_target = batch_data[2].float().to(self.config.device)
                api_target = batch_data[3].float().to(self.config.device)

                api_pred = self.model(des, index, self.api_des, test=True)
                api_loss_ = self.api_cri(api_pred, api_target)
                api_loss.append(api_loss_.item())

                api_pred = api_pred.cpu().detach()

                ndcg_, recall_, ap_, pre_, pb_ = metric(batch_data[3], api_pred.cpu(),
                                                               self.config.api_tag_embed, self.config.popular_items,
                                                               top_k_list=self.top_k_list)

                ndcg_a += ndcg_
                recall_a += recall_
                ap_a += ap_
                pre_a += pre_
                pb_a += pb_

        api_loss = np.average(api_loss)

        ndcg_a /= num_batch
        recall_a /= num_batch
        ap_a /= num_batch
        pre_a /= num_batch
        pb_a /= num_batch

        info = '[%s] ApiLoss:%s \n' \
               'NDCG_A:%s\n' \
               'AP_A:%s\n' \
               'Pre_A:%s\n' \
               'Recall_A:%s\n' \
               'Popular_basis_A:%s' % (
                   label, api_loss.round(6), ndcg_a.round(6), ap_a.round(6), pre_a.round(6), recall_a.round(6),
                   pb_a.round(6))

        print(info)
        self.log.write(info + '\n')
        self.log.flush()
        return api_loss

    def case_analysis(self):
        case_path = curPath + '/model/case/{0}.json'.format(config.model_name)
        a_case = open(case_path, mode='w')
        api_case = []
        self.model.eval()
        with torch.no_grad():

            for batch_idx, batch_data in enumerate(self.test_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                category_target = batch_data[2].argsort(descending=True)[:, :3].tolist()
                api_target = batch_data[3].argsort(descending=True)[:, :3].tolist()
                api_pred_ = self.model(des, index, self.api_des)

                api_pred_ = api_pred_.cpu().argsort(descending=True)[:, :3].tolist()

                for i, api_tuple in enumerate(zip(api_target, api_pred_)):
                    target = []
                    pred = []
                    name = self.ds.mashup_ds.name[index[i].cpu().tolist()]
                    for t in api_tuple[0]:
                        target.append(self.ds.mashup_ds.used_api_mlb.classes_[t])
                    for t in api_tuple[1]:
                        pred.append(self.ds.mashup_ds.used_api_mlb.classes_[t])
                    api_case.append((name, target, pred))

        json.dump(api_case, a_case)
        a_case.close()


if __name__ == '__main__':
    # load ds
    print('Start ...')
    start_time = time.time()
    now = time.time()
    ds = TextDataset(args.multi, is_random=False)
    print('Time for loading dataset: ', get_time(now))
    setup_seed(2020)

    # initial
    config = SEHGNConfig(ds)
    train_idx, val_idx, test_idx, sample_lt = get_indices_withlt(ds.mashup_ds, config.api_freq,
                                                                 threshold=args.lt_threshold)

    idx = train_idx
    model = SEHGN(config, idx)
    model.to(config.device)
    train_iter = DataLoader(ds.mashup_ds, batch_size=config.batch_size,
                            sampler=SubsetRandomSampler(train_idx), drop_last=True)
    val_iter = DataLoader(ds.mashup_ds, batch_size=config.batch_size,
                          sampler=SubsetRandomSampler(val_idx), drop_last=True)
    test_iter = DataLoader(ds.mashup_ds, batch_size=1,
                           sampler=SubsetRandomSampler(test_idx), drop_last=True)

    sample_lt_iter = DataLoader(ds.mashup_ds, batch_size=1,
                                sampler=SubsetRandomSampler(sample_lt), drop_last=True)

    # training
    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d", timeStruct)
    log_path = curPath + '/model/log/{0}.log'.format(config.model_name)
    log = open(log_path, mode='a')
    log.write(strTime + '\n')
    log.flush()

    # model_path = curPath + '/model/checkpoint/%s.pth' % config.model_name
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    train_func = Train(input_model=model,
                       input_config=config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       sample_lt_iter=sample_lt_iter,
                       log=log,
                       input_ds=ds,
                       )
    # training
    train_func.train()

    # testing
    train_func.evaluate(test=True)
    train_func.evaluate(test=True, sample=True)
    train_func.case_analysis()
    log.close()

    print(config.model_name)
