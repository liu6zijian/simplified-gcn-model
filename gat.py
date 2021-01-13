import torch
import torch.nn as nn
import torch.nn.functional as F

class attention_layer(nn.Module):
    def __init__(self, in_c, out_c):
        super(attention_layer, self).__init__()
        '''
        w: the parameter of self.fc
        a1: the parameter of front affine
        a2: the parameter of back affine
        h = x w
        att = [a1||a2] * [h||h^T]
        att = att \odot adj
        att = softmax(att)
        '''
        self.fc = nn.Linear(in_c, out_c)
        self.a1 = nn.Linear(out_c, 1)
        self.a2 = nn.Linear(out_c, 1)

    def forward(self, x, adj):
        x = self.fc(x) # [B, N, F] [N, 1]
        h1 = self.a1(x) # [B, N, 1] [N, 1]
        h2 = self.a2(x) # [B, N, 1] [N, 1]
        if x.size().__len__() == 3:
            B, N = h1.size()[:2]
            h1 = h1.repeat(1,1,N)
            h2 = h2.repeat(1,1,N).permute(0,2,1)
            adj = adj.unsqueeze(0).repeat(B, 1, 1)
        elif x.size().__len__() == 2:
            N = h1.size()[0]
            h1 = h1.repeat(1,N)
            h2 = h2.repeat(1,N).t()
            # adj = adj.unsqueeze(0).repeat(B, 1, 1)
        else:
            raise ValueError('Please input the valid feature shape [N, F]')
        e = F.leaky_relu(h1 + h2)
        
        att = torch.where(adj>0, e, -1e9*torch.ones_like(e))
        att = F.softmax(att, dim=1)
        att = F.dropout(att, p=0.5, training=self.training)
        if x.size().__len__() == 3:
            return torch.bmm(att, x)
        elif x.size().__len__() == 2:
            return torch.mm(att, x)

class gat_model(nn.Module):
    def __init__(self, in_c, out_c, hid_c=16, head=8):
        super(gat_model, self).__init__()

        self.atts = [attention_layer(in_c,hid_c) for _ in range(head)]
        self.out_att = attention_layer(hid_c*head, out_c)
        for i, att in enumerate(self.atts):
            self.add_module('att_{}'.format(i), att)

    def forward(self, x, adj):
        x = torch.cat([F.relu(att(x, adj) ) for att in self.atts], dim=-1)
        x = F.dropout(x, p=.5, training=self.training)
        x = self.out_att(x, adj)
        return x

if __name__ == '__main__':
    N, F_1, F_3, B = 10, 128, 7, 32
    Adj = torch.rand(N, N)

    Fea = torch.randn(B, N, F_1)
    model = gat_model(F_1, F_3)
    output = model(Fea, Adj)
    print(output.shape)

