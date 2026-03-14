import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.collect_env import TORCH_AVAILABLE

#embedding 以序列建模为例子
#考虑source sentence 和 target sentence
#构建序列，序列的字符以其载词表中的索引的形式表示
#单词表
max_num_src_words=8
max_num_tgt_words=8
max_position_len=8
batch_size=8
model_dim=8
src_len = torch.Tensor([2, 3]).to(torch.int32)
tgt_len = torch.Tensor([3, 4]).to(torch.int32)




pos_mat=torch.unsqueeze(torch.arange(max_position_len),1)
pos_mat=torch.arange(max_position_len).reshape((-1,1))#构造列张量
i_mat=pow(10000,torch.unsqueeze(torch.arange(0,model_dim,2),0)/model_dim)#构造行张量
position_embedding_table=torch.zeros(max_position_len,model_dim)#初始零化一个二维矩阵
position_embedding_table[:,0::2]=torch.sin(pos_mat/i_mat)
position_embedding_table[:,1::2]=torch.cos(pos_mat/i_mat)

position_embedding=nn.Embedding(max_position_len,model_dim)
position_embedding.weight=nn.Parameter(position_embedding_table,requires_grad=False)
src_position=torch.cat([torch.unsqueeze(torch.arange(max(src_len)),0) for _ in src_len])
tgt_position=torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)),0) for _ in tgt_len])#需要为填充数字0提供位置索引，必须等长度。而且不等长度也不可能拼接

src_position_embedding=position_embedding(src_position)
tgt_position_embedding=position_embedding(tgt_position)



print(src_position)
print(position_embedding)
print(position_embedding_table)
print(position_embedding.weight)
print(src_position_embedding)
print(pos_mat)
print(i_mat)
print(pos_mat/i_mat)
