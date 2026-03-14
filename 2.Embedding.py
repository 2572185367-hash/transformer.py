import torch
from torch.nn.utils.rnn import pad_sequence
# 原始句子
src_sentences = ["I am a student ", "I am  teacher WHO ARE YOU BABAY "]
tgt_sentences = ["我是一个学生", "我是一个老师"]

# 1. 分词（英文按空格，中文按字符）
src_tokens = [sent.split() for sent in src_sentences]          # [['I', 'am', 'a', 'student'], ['I', 'am', 'a', 'teacher']]
tgt_tokens = [list(sent) for sent in tgt_sentences]            # [['我', '是', '一', '个', '学', '生'], ['我', '是', '一', '个', '老', '师']]

# 2. 分别构建源语言和目标语言的词汇表
# 源语言词汇表
src_all_tokens = []
for tokens in src_tokens:
    src_all_tokens.extend(tokens)
src_unique_tokens = sorted(set(src_all_tokens))
src_vocab = {token: idx for idx, token in enumerate(src_unique_tokens, start=1)}  # 索引从1开始
src_vocab['<PAD>'] = 0
src_vocab['<UNK>'] = len(src_vocab)  # 当前长度即为<UNK>的索引

# 目标语言词汇表
tgt_all_tokens = []
for tokens in tgt_tokens:
    tgt_all_tokens.extend(tokens)
tgt_unique_tokens = sorted(set(tgt_all_tokens))
tgt_vocab = {token: idx for idx, token in enumerate(tgt_unique_tokens, start=1)}
tgt_vocab['<PAD>'] = 0
tgt_vocab['<UNK>'] = len(tgt_vocab)

# 3. 将每个句子转换为索引列表（使用对应的词汇表处理未知词）
src_indices = [[src_vocab.get(token, src_vocab['<UNK>']) for token in sent] for sent in src_tokens]
tgt_indices = [[tgt_vocab.get(token, tgt_vocab['<UNK>']) for token in sent] for sent in tgt_tokens]

# 4. 转换为 PyTorch 的 LongTensor
src_tensors = [torch.tensor(indices, dtype=torch.long) for indices in src_indices]
tgt_tensors = [torch.tensor(indices, dtype=torch.long) for indices in tgt_indices]
src_batch = pad_sequence([torch.tensor(seq) for seq in src_indices],batch_first=True,padding_value=0)
print(src_batch)
print(src_tensors)
print(src_vocab)