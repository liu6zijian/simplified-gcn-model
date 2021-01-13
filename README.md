# simplified-gcn-model
我们提供了简化的GCN和GAT模型，原模型取自于ICLR的会议文章
>https://github.com/Diego999/pyGAT

## GCN
对于GCN，我们将模型参数的初始化交给了kai_ming_uniform_，定义线性层来替代定义变量与变化初始化，并将模型并行化，引入batch的维度
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
```
from gat import gat_model

model = gat_model(inchannels, out_channels, hid_c=16, head=8)
model = model.cuda()
for idx, (image, adj, label) in enumerate(dataload):
    image, adj, label = image.cuda(), adj.cuda(), label.cuda()
    output = model(image, adj)
    loss = F.cross_entropy_loss(output, label)
```


