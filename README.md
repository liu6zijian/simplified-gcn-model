# simplify-gcn-model
我们提供了简化的GCN和GAT模型，原模型取自于ICLR的会议文章

## GCN
对于GCN，我们将模型参数的初始化交给了kai_ming_uniform_，定义线性层来替代定义变量与变化初始化，并将模型并行化，引入batch的维度

## GAT
对于GAT，我们将attention concat的操作简化为简单的矩阵相加，并将模型并行化，引入batch的维度


