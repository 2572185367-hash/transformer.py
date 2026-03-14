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
model_dim=8
#构建索引句子
src_seq=torch.cat([torch.unsqueeze(F.pad(torch.randint(2,max_num_src_words,(L,)),(0,max_num_src_words-L)),0)for L in src_len])
tgt_seq=torch.cat([torch.unsqueeze(F.pad(torch.randint(2,max_num_tgt_words,(L,)),(0,max_num_tgt_words-L)),0)for L in tgt_len])
#构建词嵌入表
src_Embedding_table=nn.Embedding(max_num_src_words+1,model_dim)
tgt_Embedding_table=nn.Embedding(max_num_tgt_words+1,model_dim)
#构建句映射
src_Embedding=src_Embedding_table(src_seq)
tgt_Embedding=tgt_Embedding_table(tgt_seq)
#验证
print(src_Embedding_table)
print(src_Embedding_table.weight)

print(src_Embedding_table)
print(src_Embedding_table.weight)
#