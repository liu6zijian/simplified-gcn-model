# simplified-gcn-model
我们提供了简化的GCN和GAT模型，原模型取自于ICLR的会议文章
## source
GCN paper >https://arxiv.org/pdf/1609.02907.pdf

GCN pytorch version >https://github.com/tkipf/pygcn

GAT paper >https://arxiv.org/pdf/1710.10903.pdf

GAT pytorch version >https://github.com/Diego999/pyGAT

## GCN
对于GCN，我们将模型参数的初始化交给了kai_ming_uniform_，定义线性层(`torch.nn.Linear(in_c, out_c)`)来替代定义变量与变化初始化，并将模型并行化`[B, N, F]`，引入batch的维度

###code
```
from gcn import gcn_model # 导入模型

model = gcn_model(inchannels, out_channels, hid_c=16) # 初始化模型
model = model.cuda()
for idx, (image, adj, label) in enumerate(dataloader):
    '''
    image [N, F_1]
    adj [N, N]
    label [N], one hot of [N, F_3]
    '''
    image, adj, label = image.cuda(), adj.cuda(), label.cuda()
    loss = F.cross_entropy_loss(output, label)
    output = model(image, adj) # 将数据喂入模型
```
## GAT
对于GAT，我们将attention concat的操作简化为简单的矩阵相加，并将模型并行化，引入batch的维度

简化过程如下:
```
h = Wx
e_ij = a^T (h_i||h_j)
a^T = (a_1^T || a_2^T)
e_ij = (a_1^T || a_2^T) * (h_i||h_j) = a_1^T h_i + a_2^T h_j
h1 = a_1 h
h2 = a_2 h
e = h1.repeat(1,N) + h2.repeat(N,1).t()
```
![简化计算过程]https://github.com/liu6zijian/simplified-gcn-model/blob/main/simplified_calculation.png


```
from gat import gat_model

model = gat_model(inchannels, out_channels, hid_c=16, head=8)
model = model.cuda()
for idx, (image, adj, label) in enumerate(dataload):
    image, adj, label = image.cuda(), adj.cuda(), label.cuda()
    output = model(image, adj)
    loss = F.cross_entropy_loss(output, label)
```


