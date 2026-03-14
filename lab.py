import torch
import  numpy
import torch.nn.functional as F
import torch.nn as nn
#基本设定
batch_size=2
max_num_src_words=5
max_num_tgt_words=7
src_len=torch.randint(2,5,(batch_size,))
tgt_len=torch.randint(2,5,(batch_size,))
model_dim=128
#构建索引句子
src_seq=[(F.pad(torch.randint(2,max_num_src_words,(L,)),(0,max_num_src_words-L)))for L in src_len]

#验证
print(src_seq)
print(src_len)