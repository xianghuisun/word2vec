import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)
batch_size=100
embedding_dim=100
max_vocab_size=10000
hidden_dim=64
seq_length=50
train_path='/home/sun/Documents/NLP/text8/text8.dev.txt'
import operator
from collections import Counter

def get_data(train_path):
	with open(train_path,'r',encoding='utf-8') as f:
	    lines=f.readlines()
	assert len(lines)==1 and type(lines)==list
	text_data=lines[0].strip().split()
	print("The length of total text is ",len(text_data))
	counter_dict=Counter(text_data)
	sorted_list=sorted(counter_dict.items(),key=operator.itemgetter(1),reverse=True)
	if len(sorted_list)>=max_vocab_size:
		common_words=sorted_list
	else:
		common_words=sorted_list[:max_vocab_size]
	all_words=[]
	for word,freq in common_words:
		all_words.append(word)
	word2id={}
	for word in all_words:
	    word2id[word]=len(word2id)
	word2id['unk']=len(word2id)
	
	print("The length of word2id is ",len(word2id))
	train_data=[word2id.get(word,word2id['unk']) for word in text_data]
	label_data=[word2id.get(word,word2id['unk']) for word in text_data[1:]]
	#上面两条语句就是把文本中每一个句子的每一个单词转成对应的id
	label_data.append(word2id['unk'])
	assert len(train_data)==len(label_data)==len(text_data)

	return text_data,word2id,train_data,label_data

text_data,word2id,train_data,label_data=get_data(train_path)
# class DataSet_class(torch.utils.data.Dataset):
# 	def __init__(self,text_data,word2id,train_data,label_data):
# 		self.train_data_tensor=torch.Tensor(train_data).long()
# 		self.label_data_tensor=torch.Tensor(label_data).long()
# 	def __len__(self):
# 		return self.train_data_tensor.shape[0]
# 	def __getitem__(self):
# 		return self.train_data_tensor,self.label_data_tensor

# pool_dataset=DataSet_class(text_data,word2id,train_data,label_data)
# data_loader=torch.utils.data.DataLoader(dataset=pool_dataset,batch_size=batch_size,shuffle=True,num_workers=1)

def generate_data(text_data,word2id,seq_length,train_data,label_data):
    train_seq,label_seq=[],[]
    train_seqs,label_seqs=[],[]
    for word_id,label_id in zip(train_data,label_data):
        train_seq.append(word_id)
        label_seq.append(label_id)
        if len(train_seq)==seq_length:
            train_seqs.append(train_seq)
            label_seqs.append(label_seq)
            train_seq,label_seq=[],[]
    train_tensor=torch.Tensor(train_seqs).long()
    label_tensor=torch.Tensor(label_seqs).long()
    assert train_tensor.shape==label_tensor.shape==(len(text_data)//seq_length,seq_length)
    #train_tensor,label_tensor就是把所有的单词，每５０个组成一个句子和相应的标签，也就是说整个文本中有len(text_data)//seq_length个句子
    #每次拿batch_size个句子出来，每个句子的长度是固定的seq_length的
    return train_tensor,label_tensor

train_tensor,label_tensor=generate_data(text_data,word2id,seq_length,train_data,label_data)
#train_tensor,label_tensor是句子的索引表示
def batch_yield(train_tensor,label_tensor,batch_size):
    num_batches=len(train_tensor)//batch_size#所有的句子会取多少个batch_size次
    start=0
    shuffle_index=np.random.permutation(train_tensor.shape[0])#每次将所有的句子顺序重新打乱，其实训练语言模型不应该打乱顺序
    train_tensor=train_tensor[shuffle_index]
    label_tensor=label_tensor[shuffle_index]
    for i in range(num_batches):
        yield train_tensor[start:start+batch_size],label_tensor[start:start+batch_size]#yield出batch_size个句子和相应的标签
        start+=batch_size

num_layers=1
num_directions=1
class RNNmodel(torch.nn.Module):
    def __init__(self,embedding_size,embedding_dim,hidden_dim,batch_size,seq_length,num_classes):
    	#注意embedding_size就是句子的总数len(text_data)//seq_length
        super(RNNmodel,self).__init__()
        self.embedding_layer=torch.nn.Embedding(embedding_size,embedding_dim)
        #通过Embedding层就是将batch_size个句子中的每一个句子的每一个单词嵌入成一个embedding_dim的向量
        self.lstm_layer=torch.nn.LSTM(embedding_dim,hidden_dim,num_layers=1,batch_first=False)
        self.output_layer=torch.nn.Linear(hidden_dim,num_classes)
        self.hidden_dim=hidden_dim
        self.embedding_size=embedding_size
        self.seq_length=seq_length
        self.batch_size=batch_size
        self.num_classes=num_classes#num_classes=len(word2id)
        
    def forward(self,batch_text,hidden_state):
        assert batch_text.shape==(batch_size,seq_length)
        embedding_out=self.embedding_layer(batch_text)
        assert embedding_out.shape==(batch_size,seq_length,embedding_dim)
        embedding_out.transpose_(1,0)
        assert embedding_out.shape==(seq_length,batch_size,embedding_dim)
        output,hidden_state=self.lstm_layer(embedding_out,hidden_state)
        assert output.shape==(seq_length,batch_size,hidden_dim*num_directions)
        assert len(hidden_state)==2 and hidden_state[0].shape==(num_layers*num_directions,batch_size,self.hidden_dim)
        output=output.view(-1,self.hidden_dim)
        output=self.output_layer(output)
        assert output.shape==(batch_size*seq_length,self.num_classes)
        output=output.view(self.batch_size,self.seq_length,self.num_classes)
        #目的就是把batch_size个seq_length长的句子的每一个单词都预测出其对应的在word2id中的id
        return output,hidden_state
    
    def init_hidden(self,batch_size,requires_grad=True):
        weights=next(self.parameters())
        return (weights.new_zeros((num_layers*num_directions,batch_size,self.hidden_dim),requires_grad=requires_grad),
               weights.new_zeros((num_layers*num_directions,batch_size,self.hidden_dim),requires_grad=requires_grad))

embedding_size=train_tensor.shape[0]
print("How many sentence in this text ",embedding_size)
num_classes=len(word2id)
model=RNNmodel(embedding_size,embedding_dim,hidden_dim,batch_size=batch_size,seq_length=seq_length,num_classes=num_classes)
#detach的方法，将variable参数从网络中隔离开，不参与参数更新
def repackage_hidden(hidden_state):
    if isinstance(hidden_state,torch.Tensor):
        return hidden_state.detach()
    else:
        return tuple(repackage_hidden(value) for value in hidden_state)

loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
vocab_size=num_classes
for epoch in range(10):
    batch_data=batch_yield(train_tensor,label_tensor,batch_size)
    hidden_state=model.init_hidden(batch_size)
    for i,(batch_data,batch_label) in enumerate(batch_data):
        assert batch_data.shape==batch_label.shape==(batch_size,seq_length)
        hidden_state=repackage_hidden(hidden_state)
        output,hidden_state=model(batch_data,hidden_state)
        assert output.shape==(batch_size,seq_length,vocab_size)
        logits_=output.view(-1,vocab_size)
        labels_=batch_label.view(-1)
        assert logits_.shape==(batch_size*seq_length,vocab_size) and labels_.shape==(batch_size*seq_length,)
        
        loss=loss_fn(logits_,labels_)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        optimizer.step()
        
        if i%100==0:
            print("loss is ",loss.item())



