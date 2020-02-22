import torch
from collections import Counter
import numpy as np
import random
import collections
import torch.nn.functional as F
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
print(torch.__version__)
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
USE_CUDA=torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(53113)

with open("text8.train.txt",'r',encoding="utf-8") as f:
    lines=f.readlines()
assert type(lines)==list and type(lines[0])==str
text=lines[0].strip().split()
vocab=Counter(text)#统计每一单词出现的次数
assert len(vocab)<len(text) and type(vocab)==collections.Counter and type(text)==list
MAX_VOCAB_SIZE=10000#根据电脑配置来，高一点的可以大一点
vocab_common=vocab.most_common(MAX_VOCAB_SIZE-1)#留一个给unk
vocab_common_dict=dict(vocab_common)#keys are word and values is the frequency of word
vocab_common_dict['unk']=len(text)-np.sum(list(vocab_common_dict.values()))
#vocab_common_dict.values()的总和是所有的常见的单词在text中出现的次数之和，len(text)-这个和就是text中所有不常见的单词的数量总和
all_common_words=[word for word in vocab_common_dict.keys()]
len(all_common_words)==MAX_VOCAB_SIZE
word2id={}
for word in all_common_words:
    word2id[word]=len(word2id)
id2word={id_:word for word,id_ in word2id.items()}

word_freq_counts=np.array([freqs for freqs in vocab_common_dict.values()],dtype=np.float32)
word_freqs=word_freq_counts/np.sum(word_freq_counts)
word_freqs=word_freqs**(3./4.)
C=3#每个单词周围取三个单词
K=5#负采样的个数，对于每一个中心词，取6个周围的单词,每个周围的单词取5个不是周围的单词
class WordEmbeddingDataSet(torch.utils.data.Dataset):
    def __init__(self,text,word2id,id2word,word_freqs,word_freq_counts):
        super(WordEmbeddingDataSet,self).__init__()
        self.text_encoded=[word2id.get(word,word2id['unk']) for word in text]
        self.text_encoded=torch.Tensor(self.text_encoded).long()
        self.word2id=word2id
        self.id2word=id2word
        self.word_freqs=torch.Tensor(word_freqs)#word_freqs计算的是每一个常见的单词在text中出现的频数
        self.word_freq_counts=torch.Tensor(word_freq_counts)#将这些频数表示成概率的形式
 #通过定义一个类继承torch.utils.data.Dataset，实现__getitem__方法，那么就可以定义一个自己的数据集   
    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self,index):
        center_word=self.text_encoded[index]#返回中心词的id
        near_indices=list(range(index-C,index))+list(range(index+1,index+C+1))
        near_indices=[i % len(self.text_encoded) for i in near_indices]#最后几个单词防止溢出
        near_indices_tensor=self.text_encoded[near_indices]#near_indices_tensor就是center_word的周围的六个单词
        neg_indices=[]
        for i in range(0,K):
            neg_id=torch.multinomial(self.word_freqs,1,replacement=True)
            while neg_id in near_indices_tensor:#如果负采样的单词恰好在中心词的周围，那就重新采样
                neg_id=torch.multinomial(self.word_freqs,1)
            assert neg_id not in near_indices_tensor
            neg_indices.append(neg_id)
        neg_indices_tensor=torch.Tensor(neg_indices).long()
        return center_word,near_indices_tensor,neg_indices_tensor

        #multinomial(input,num_samples,replacement=False,name=None)
        #这个函数就是把input的张量按照其值的大小进行采样K*C个，返回的是张量值所对应的下标
        #值越大的越先被踩，这就是负采样：对于那些不是中心词的单词，出现的次数越多，越容易被拿出来训练，replacement=True就是类似于放回采样
        #这样概率高的单词总是会大概率的被踩出来

batch_size=100
dataset=WordEmbeddingDataSet(text,word2id,id2word,word_freqs,word_freq_counts)
data_loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2)
#迭代data_loader时就会调用dataset的getitem方法返回自己想要的数据集，否则就需要分别计算center_word,near_indices_tensor以及neg_indices_tensor
#然后写一个函数迭代
class EmbeddingModel(torch.nn.Module):
    def __init__(self,common_vocab_size,embedding_dim):
        super(EmbeddingModel,self).__init__()
        self.common_vocab_size=common_vocab_size
        self.embedding_dim=embedding_dim

        init_range=6/self.embedding_dim
        #模型的目的就是将这些常见的单词嵌入成embedding_dim维度的向量
        self.input_layer=torch.nn.Embedding(self.common_vocab_size,self.embedding_dim,sparse=False)
        self.input_layer.weight.data.uniform_(-init_range,init_range)  
        self.output_layer=torch.nn.Embedding(self.common_vocab_size,self.embedding_dim,sparse=False)
        self.output_layer.weight.data.uniform_(-init_range,init_range)

    def forward(self,input_wordids,near_wordids,neg_wordids):
        #assert input_wordids.shape==(batch_size,) and near_wordids.shape==(batch_size,2*C) and neg_wordids.shape==(batch_size,K)
        #注意最后的批次的长度不会那么巧是batch_size,通常比batch_size小
        input_embedding=self.input_layer(input_wordids)#(batch_size,embedding_dim)
        input_embedding=input_embedding.unsqueeze(2)#(batch_size,embedding_dim,1)
        near_embedding=self.input_layer(near_wordids)#(batch_size,2*C,embedding_dim)
        neg_embedding=self.input_layer(neg_wordids)#(batch_size,K,embedding_dim)
        #bmm运算就是三维张量的运算，前面的运算的意义是将取出来的batch_size个单词，每一个嵌入成一个100维的向量
        #然后这batch_size个单词，每一个单词取出来周围2*C个单词，再将这些单词嵌入成一个100维的向量
        #根据负采样技术，对这batch_size个单词,每一个单词取出来K个不相邻的单词，再将这些单词嵌入成一个100维的向量
        #也就是说现在得到了三个张量，一个是中心词的embedding,一个是周围单词的embedding，一个是不相邻的单词的embedding
        #根据论文中提出来的loss计算方法计算loss
        log_near=torch.bmm(near_embedding,input_embedding)#(batch_size,2*C,1)
        log_neg=torch.bmm(neg_embedding,-input_embedding)#(batch_size,K,1)

        log_near=log_near.squeeze(2)#(batch_size,2*C)
        log_neg=log_neg.squeeze(2)#(batch_size,K)

        log_near=F.logsigmoid(log_near).sum(1)
        log_neg=F.logsigmoid(log_neg).sum(1)
        loss=log_near+log_neg
        #论文中损失函数的定义是\log\sigma()
        return -loss
    def input_embedding(self):
        return self.input_layer.weight.data.cpu().numpy()
        #input_layer的权重就是这些常见的单词的嵌入向量,word2id中的是这些单词的对应的id，在word2id中查找是否有某个单词，有的话给出id，然后在
        #input_layer的权重中就可以得到这个单词的向量表示

embedding_dim=100
model=EmbeddingModel(MAX_VOCAB_SIZE,embedding_dim)
if USE_CUDA:
    model=model.cuda()

optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
for epoch in range(100):
    for i,(input_wordids,near_wordids,neg_wordids) in enumerate(data_loader):
        input_wordids=input_wordids.long()
        near_wordids=near_wordids.long()
        neg_wordids=neg_wordids.long()

        if USE_CUDA:
            input_wordids=input_wordids.cuda()
            near_wordids=near_wordids.cuda()
            neg_wordids=neg_wordids.cuda()

        optimizer.zero_grad()
        loss=model(input_wordids,near_wordids,neg_wordids).mean()
        loss.backward()
        optimizer.step()
        if i%1000==0:
            print("loss value is ",loss.item())

    embedding_weights=model.input_embedding()
    np.save("embedding-{}".format(embedding_dim),embedding_weights)
    torch.save(model.state_dict(),"embedding-{}".format(embedding_dim))

print("Training process is over")
model.load_state_dict(torch.load("embedding-{}".format(embedding_dim)))
embedding_matrix=model.input_embedding()
assert embedding_matrix.shape==(MAX_VOCAB_SIZE,embedding_dim)

def evaluate(evaluate_file_name,embedding_matrix):
    if evaluate_file_name.endswith(".csv"):
        evaluate_data=pd.read_csv(evaluate_file_name,sep=",")
    else:
        assert evaluate_file_name.endswith(".txt")
        evaluate_data=pd.read_csv(evaluate_file_name,sep="\t")
    golden_similarity=[]
    model_similarity=[]
    for i in evaluate_data.iloc[:,0:2].index:
        word1,word2=evaluate_data.iloc[i,0],evaluate_data.iloc[i,1]
        if word1 not in word2id or word2 not in word2id:
            continue
        #因为要计算的是两个单词之间的相似度，所以有一个单词没有出现在word2id中，也就是没有被模型所训练，那就无法计算相似度
        else:
            golden_similarity.append(float(evaluate_data.iloc[i,2]))
            word1_id,word2_id=word2id[word1],word2id[word2]
            word1_embedding,word2_embedding=embedding_matrix[[word1_id]],embedding_matrix[[word2_id]]#多一个[]的意思是返回的向量是一个二维的，因为
            #cosine_similarity()要求输入的形状必须是二维的
            assert word1_embedding.shape==word2_embedding.shape==(1,embedding_dim)
            model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embedding,word2_embedding)))

    return scipy.stats.spearmanr(golden_similarity,model_similarity)

def find_nearest(word,embedding_matrix,amount=10):
    if word not in word2id:
        print("Find nothing")
        return
    assert word in word2id
    word_id=word2id[word]
    word_embedding=embedding_matrix[word_id]
    assert word_embedding.shape==(embedding_dim,)
    cos_distance=np.array([scipy.spatial.distance.cosine(another_embedding,word_embedding) for another_embedding in embedding_matrix])
    assert type(cos_distance)==np.ndarray and cos_distance.shape==(MAX_VOCAB_SIZE,)#cos_distance记录的就是所有常见的单词与当前单词之间的cos距离
    return [id2word[id_] for id_ in cos_distance.argsort()[:amount]]#argsort 就是arg 和sort，返回的cos距离最大的前amount个下标，通过下标找到单词


simlex_path="./embedding/simlex-999.txt"
men_path="./embedding/men.txt"
wordsim_path="./embedding/wordsim353.csv"

print("simlex result ",evaluate(simlex_path,embedding_matrix))
print("men result ",evaluate(men_path,embedding_matrix))
print("wordsim_path ",evaluate(wordsim_path,embedding_matrix))

for word in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]:
    print(word, find_nearest(word,embedding_matrix,20))




        



