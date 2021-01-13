import torch
import torch.nn as nn
import torch.nn.functional as F

class gcn_layer(nn.Module):
    def __init__(self, in_c, out_c):
        super(gcn_layer, self).__init__()
        # affine function layer
        self.fc = nn.Linear(in_c, out_c)
    def forward(self, x, adj):
        '''
        x: input features [N, F_1]
        adj: adjacent matrix of graph $G=(V,E)$ [N, N]
        w: the parameter of self.fc [F_1, F_2]
        N: the number of nodes
        F_1: is the length of input feature vector $x_i, i=[1,2, \cdots, N]$
        F_2: is the length of output feature vector $y_i, i=[1,2, \cdots, N]$
        $y = Adj x w$
        '''
        x = self.fc(x) # space affine
        if x.size().__len__() == 3:
            Batch = x.size(0)
            adj = adj.repeat(Batch, 1, 1)
            x = adj.bmm(x) # batch information propagation on the graph
        elif x.size().__len__() == 2:
            x = adj.mm(x) # information propagation on the graph
        else:
            raise ValueError('Please input the valid feature shape [N, F]')
        return  x

class gcn_model(nn.Module):
    def __init__(self, in_c, out_c, hid_c=16):
        super(gcn_model, self).__init__()
        '''
        two graph convolutional layers model
        hidden layer is default as 16 
        '''
        self.gconv1 = gcn_layer(in_c, hid_c)
        self.gconv2 = gcn_layer(hid_c, out_c)

    def forward(self, x, adj):
        '''
        x^0: input features [N, F_1]
        adj: adjacent matrix of graph $G=(V,E)$ [N, N]
        w_1: the parameter of self.gconv1 [F_1, F_2]
        w_2: the parameter of self.gconv2 [F_2, F_3]
        $x^1 = relu(Adj x^0 w_1)$
        $x^2 = softmax(Adj x^1 w_2)$
        '''
        if Adj.layout == torch.sparse_coo: # torcg,sparse_coo - torch.strided
            adj = adj.to_dense() # convert adjacent matrix from sparse to dense
        x = self.gconv1(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.gconv2(x, adj)
        # x for cross_entropy_loss with softmax
        # F.softmax(x, dim=1) for nll_loss
        return x
        # return F.softmax(x, dim=1)
        

if __name__ == '__main__':
    N, F_1, F_3, B = 10, 128, 7, 32
    Adj = torch.rand(N, N)
    Adj = torch.where(Adj>0.5, torch.ones_like(Adj), torch.zeros_like(Adj) )
    # Adj = Adj.to_sparse()

    Fea = torch.randn(B, N, F_1)

    model = gcn_model(F_1, F_3)
    output = model(Fea, Adj)
    print(output.shape)



    