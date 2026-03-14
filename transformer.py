import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from fontTools.voltLib.ast import SubstitutionReverseChainingSingleDefinition
from torch.utils.collect_env import TORCH_AVAILABLE

#embedding 以序列建模为例子
#考虑source sentence 和 target sentence
#构建序列，序列的字符以其载词表中的索引的形式表示
#单词表
max_num_src_words=8
max_num_tgt_words=8
max_position_len=8
batch_size=2

model_dim=8


#src_len=torch.randint(2,10,(batch_size,))#创建一个与batch_size等长度的一维张量，范围2到9
#tgt_len=torch.randint(2,10,(batch_size,))

src_len = torch.Tensor([2, 3]).to(torch.int32)
tgt_len = torch.Tensor([3, 4]).to(torch.int32)

src_seq =torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)),(0,max_num_src_words-L)) ,0)for L in src_len] )
#tgt_seq=torch.cat([F.pad(torch.unsqueeze(torch.randint(1,max_num_tgt_words,(L,)),0),(0,max_num_src_words-L))for L in tgt_len])
#tgt_seq=F.pad(torch.unsqueeze(torch.cat([torch.randint(1,max_num_tgt_words,(L,)) for L in tgt_len]),0),(0,7))
tgt_seq=torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)),(0,max_num_tgt_words-L)) ,0)for L in tgt_len] )
# 循环创建一个长度为l的一维张量，l的范围由src_len中的长度参数决定。(如tensor[2，3]则其中的长度参数为2和3)
# 循环填充创建一个长度为l的一维张量，l的范围由src_len中的长度参数决定。
#src_seq = torch.cat([F.pad(torch.randint(1, max_num_src_words, (L,)),(0,max_num_src_words-L)) for L in src_len])
#src_seq = [torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)),(0,max_num_src_words-L)),0) for L in src_len]
#src_seq = [F.pad(torch.randint(1, max_num_src_words, (L,)),(0,max_num_src_words-L)) for L in src_len]
#tgt_seq=[torch.randint(1,max_num_tgt_words,(L,)) for L in tgt_len]

#第三步，构造embedding
src_Embedding_table=nn.Embedding(max_num_src_words+1,model_dim)
tgt_Embedding_table=nn.Embedding(max_num_tgt_words+1,model_dim)
src_Embedding=src_Embedding_table(src_seq)
tgt_Embedding=tgt_Embedding_table(tgt_seq)

#position embedding
pos_mat=torch.unsqueeze(torch.arange(max_position_len),1)
pos_mat=torch.arange(max_position_len).reshape((-1,1))#构造列张量
i_mat=pow(10000,torch.unsqueeze(torch.arange(0,model_dim,2),0)/model_dim)#构造行张量
position_embedding_table=torch.zeros(max_position_len,model_dim)#初始零化一个二维矩阵
position_embedding_table[:,0::2]=torch.sin(pos_mat/i_mat)
position_embedding_table[:,1::2]=torch.cos(pos_mat/i_mat)

position_embedding=nn.Embedding(max_position_len,model_dim)
position_embedding.weight=nn.Parameter(position_embedding_table,requires_grad=False)
src_position=torch.cat([torch.unsqueeze(torch.arange(src_seq.shape[1]),0) for _ in src_len])
tgt_position=torch.cat([torch.unsqueeze(torch.arange(tgt_seq.shape[1]),0) for _ in tgt_len])
#先前我们使用max(src-len）生成位置编码，这是不对的，这样的结果必然是[2,3,8]维度的张量，无法与填充过的src-E相加 维度不匹配
# (填充长度需要为填充数字0提供位置索引，必须等长度。而且不等长度也不可能拼接)这是先前的化
#说实话 不如直接填入max_position_len省事
src_position_embedding=position_embedding(src_position)
tgt_position_embedding=position_embedding(tgt_position)

#print(position_embedding_table)
#print(position_embedding.weight)
#print(src_position_embedding)
total_src_embedding=src_Embedding+src_position_embedding
total_tgt_embedding=tgt_Embedding+tgt_position_embedding


################
################{
# softmax
#a1=0.1
#a2=10
#score=torch.randn(5)#生成一个包含 5 个随机数的一维张量，每个数服从标准正态分布（均值为 0，方差为 1）。
#prob=F.softmax(score,-1)
#prob2=F.softmax(score*0.1,-1)
#prob3=F.softmax(score*10,-1)
#softmax(score, dim=-1) 或 dim=1 是对每一行的所有列元素做 softmax，使得每行的和为 1（因为归一化发生在列之间）。
#softmax(score, dim=0) 是对每一列的所有行元素做 softmax，使得每列的和为 1。
#def softmax_func(score):
 #   return F.softmax(score)
#jaco_mat1=torch.autograd.functional.jacobian(softmax_func,score*a1)
#jaco_mat2=torch.autograd.functional.jacobian(softmax_func,score*a2)
#print(score)
#print(prob)
#print(prob2)
#print(prob3)
#print(jaco_mat1)
#print(jaco_mat2)
#print(total_src_embedding)
#####################


#mask 的shape(bacth_size,max_src_len,max_src_len)值为1和-inf
valid_encoder_pos=torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max_num_src_words-L)) ,0)for L in src_len]),2)
valid_encoder_pos1=torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max_num_src_words-L)) ,0)for L in src_len]),0)
#c=torch.unsqueeze(torch.randn(5),0)
valid_encoder_pos_matrix=torch.bmm(valid_encoder_pos,valid_encoder_pos.transpose(1,2))
invalid_encoder_pos_matrix=1-valid_encoder_pos_matrix
mask_encoder_self_attention=invalid_encoder_pos_matrix.to(torch.bool)


score=torch.randn(batch_size,max_num_src_words,max_num_src_words)

masked_score=score.masked_fill(mask_encoder_self_attention,-1e9)
prob=F.softmax(masked_score,-1)

print(score)
print(masked_score)
print(prob)
