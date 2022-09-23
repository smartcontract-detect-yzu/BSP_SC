"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import collections
import random
import dgl
from dgl.data import DGLDataset
import torch
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.nn import AvgPooling

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f_t = self.U_f(nodes.mailbox['h'])
        f = th.sigmoid(f_t)
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}

class TreeLSTM(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 num_classes,
                 dropout):
        
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        cell = ChildSumTreeLSTMCell
        self.cell = cell(x_size, h_size)
        self.avgpooling = AvgPooling()

    def forward(self, g, h, c):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        g : dgl.DGLGraph
            Tree for computation.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # feed embedding
        embeds = g.ndata["feature"]
        g.ndata['iou'] = self.cell.W_iou(embeds)  # hsize:150  [1*100] * [100*3*150] => [1*300]
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = g.ndata.pop('h')
        h = self.avgpooling(g, h)
        logits = self.linear(h)
        logits = F.log_softmax(logits, 1)
        return logits

class StmtDataset(DGLDataset):
    
    def __init__(self):
        super().__init__(name='statement_dataset')

    def process(self):
        self.graphs = []
        self.labels = []
        
        graphs, label_dict = dgl.load_graphs("reentrancy.bin")
        self.graphs += graphs
        self.labels += label_dict["glabel"]

        random.shuffle(self.graphs)
        
        print("TOTAL: {}".format(len(self.graphs)))

        train_size = int(0.8*len(self.graphs))
        train_dataset = self.graphs[0:train_size]

        valid_size = int(0.1*len(self.graphs))
        valid_dataset = self.graphs[train_size + 1: train_size + valid_size]

        test_dataset = self.graphs[train_size + valid_size + 1: ]
        
        return train_dataset, valid_dataset, test_dataset
        
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


StmtAstBatch = collections.namedtuple('StmtAstBatch', ['graph', 'feature', 'label', "size"])
def batcher(device):
    def batcher_dev(batch):
        labels = torch.zeros(len(batch), 2, dtype=torch.float32)
        for idx, graph in enumerate(batch):
            labels[idx] = graph.ndata["label"][0]
        batch_trees = dgl.batch(batch)
        return StmtAstBatch(graph=batch_trees, feature=batch_trees.ndata['feature'].to(device), label=labels.to(device), 
                            size=batch_trees.batch_size)   
    return batcher_dev


def calculate_metrics(preds, labels, vul_label_is=1):
    """
    
    """

    TP = FP = TN = FN = 0
    vul_cnt = no_vul_cnt = 0

    for pred, label in zip(preds, labels):
        if label == vul_label_is:
            vul_cnt += 1
            if pred == label:
                TP += 1
            else:
                FP += 1
        else:
            no_vul_cnt += 1
            if pred == label:
                TN += 1
            else:
                FN += 1

    total_data_num = TP + TN + FP + FN

    # 计算acc
    acc = (TP + TN) / (TP + TN + FP + FN)

    # 计算recall
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    # 计算precision
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    # 计算f1
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    print("ACC:{}, RE:{}, P:{}, F1:{},TOTAL:{}".format(acc, recall, precision, f1, total_data_num))
    return acc, recall, precision, f1, total_data_num


def do_test_model(model, dataset_loader, device):

    # 模型测试
    with torch.no_grad():
        model.eval()
        predicts = []
        labels = []
        for _, batch in enumerate(dataset_loader):  
            g = batch.graph.to(device)
            n = g.number_of_nodes()
            h = th.zeros((n, 150)).to(device)  # [number_of_nodes * h_size(is 150)]
            c = th.zeros((n, 150)).to(device)

            logits = model(g, h, c)
            # logits = F.log_softmax(logits, 1)
            
            predicts += logits.argmax(dim=1)  
            labels += batch.label.argmax(dim=1)

        acc, recall, precision, f1, total_data_num = calculate_metrics(predicts, labels)
        return acc, recall, precision, f1, total_data_num

if __name__ == '__main__':

    if torch.cuda.is_available():
         device = torch.device("cuda:0")
         th.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    print(device)

    dataset = StmtDataset()
    train_dataset, valid_dataset, test_dataset = dataset.process()
    print("train:{} valid:{} test:{}".format(len(train_dataset), len(valid_dataset), len(test_dataset)))

    # train_dataset, valid_dataset, test_dataset = split_dataset(dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=None)
    
    batch_size = 128
    epoch = 256
    h_size = 150
    c_size = 150

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              collate_fn=batcher(device),
                              shuffle=True,
                              num_workers=0)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              collate_fn=batcher(device),
                              shuffle=True,
                              num_workers=0)

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              collate_fn=batcher(device),
                              shuffle=True,
                              num_workers=0)
    

    model = TreeLSTM(x_size=100,
                     h_size=150,
                     num_classes=2,
                     dropout = 0).to(device)

    
    print(model) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08)
    
    # 模型训练
    for epoch in range(epoch):
        t_epoch = time.time()
        model.train()
        training_loss = 0
        for step, batch in enumerate(train_loader):
            g = batch.graph.to(device)
            n = g.number_of_nodes()
            h = th.zeros((n, h_size)).to(device)  # [number_of_nodes * h_size(is 150)]
            c = th.zeros((n, c_size)).to(device)
            
            logits = model(g, h, c)
            # logits = F.log_softmax(logits, 1)  # log_softmax(logits, 1)
            loss = F.cross_entropy(logits, batch.label)
             # loss = F.nll_loss(logp, batch.label, reduction='sum')
            training_loss += loss.item() * batch.size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        training_loss /= len(train_loader.dataset)
        print("EPOCH:{} training_loss:{}".format(epoch, training_loss))


        if (epoch + 1) % 8 == 0:
            print("Test @EPOCH:{}".format(epoch))
            do_test_model(model, valid_loader, device)
            acc, recall, precision, f1, total_data_num = do_test_model(model, test_loader, device)

            model_name = "stmt_encoder_{}_{}_{}_{}_{}.pt".format(epoch, acc, recall, precision, f1)
            torch.save(model.state_dict(), model_name)

                    