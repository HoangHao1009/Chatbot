# Seq2Seq With Attention CHATBOT
This project is built to help you traning a simple chatbot by your own data. You can also use pre-train chatbot that is trained with [Cornell Movie Dialogs](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).

The architectures of chatbot are: modern RNN (GRU), seq2seq, encoder - decoder, Doc-type-attention mechanism (LuongAttention).

# Library installation
Run this code to use chatbot:
```
git clone https://github.com/HoangHao1009/Chatbot
cd Chatbot
pip install -e .
```

# Usage
## I. Train your model by yourself
### 1. Prepare data
The input data format is csv/ txt file with '\n' between lines and '\t' between a question and answer.
Demo file: `csvdata.txt`
Or you can try `Datapreparer` to create a file that can be used to train:
```python
from chatbot import Datapreparer
import torch

movie_data = Datapreparer.LoadmovieData('movie-corpus/utterances.jsonl') #utterences.jsonl that contain lines of question-answer
#load data and take pairs
movie_data.loadLinesAndConversations()
movie_data.extractSentencePairs()

#write csv file
delimiter = '\t'
csvdata = Datapreparer.Csvwriter('csvdata.txt', movie_data.qa_pairs, delimiter)
csvdata.write()
```

### 2. Transform raw csv data to data that model can use
```python
from chatbot import Vocabulary
#Take vocabulary and pairs from csv file
max_length = 15 #max_length that you accept 
vocandpair = Vocabulary.VocAndPairLoader()
vocandpair.PrepareDataFrom('csvdata.txt', max_length) #filter pair lengths from csv file

#Transform it to right format with pytorch model
modeldata = Vocabulary.ModelData(vocandpair.voc, vocandpair.pairs)
#You can cut rare words out of data
modeldata.trimRareWords(2)
```

### 3. Design Model
For an understanding of model building, you should refer to [Put the link]()
```python
from chatbot import Model
#Set parameters for model
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Design model
embedding = nn.Embedding(modeldata.voc.num_words, hidden_size)
encoder = Model.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = Model.AttnDecoderRNN(embedding, hidden_size, modeldata.voc.num_words, decoder_n_layers, dropout)
encoder.to(device)
decoder.to(device)
```

### 4. Training

```python
#Set training parameters
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 10
save_every = 500
encoder.train()
decoder.train()

#Chose optimizer for training
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = learning_rate * decoder_learning_ratio)
training = Trainer.TrainIter(modeldata, encoder, decoder,
                             encoder_optimizer, decoder_optimizer, 
                             embedding, n_iteration, batch_size, print_every, clip)
training.run(max_length, teacher_forcing_ratio)
```

### 5. Save model 
```python
torch.save({
    'en': encoder.state_dict(),
    'de': decoder.state_dict(),
    'en_opt': encoder_optimizer.state_dict(),
    'de_opt': decoder_optimizer.state_dict(),
    'voc_dict': voc.__dict__
}, 'save_model.pt')
```

## II. Use Pre-train Model

### 1. Load parameters and data
```python
from chatbot import Chatbot, Model, Vocabulary
import torch
import torch.nn as nn

checkpoint = torch.load('save_model.pt')
for keys in checkpoint.keys():
    print(keys)

voc = Vocabulary.Voc()
voc.__dict__.update(checkpoint['voc_dict'])
```

### 2. Model setting
```python
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding = nn.Embedding(voc.num_words, hidden_size)
encoder = Model.EncoderRNN(hidden_size, embedding, encoder_n_layers)
decoder = Model.AttnDecoderRNN(embedding, hidden_size, voc.num_words, decoder_n_layers)

encoder.load_state_dict(checkpoint['en'])
decoder.load_state_dict(checkpoint['de'])
```

## III. Chatting with model
```python

encoder.eval()
decoder.eval()

searcher = Chatbot.GreedySearchDecoder(encoder, decoder)
chatbot = Chatbot.Chatbot(searcher, voc)
chatbot.chatting(20)
```

# Reference
1. [Seq2seq apply attention mechanism paper](https://arxiv.org/ftp/arxiv/papers/2006/2006.02767.pdf)
2. LuongAttention Paper: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
3. Foundation knowledge and coding from scratch with: [Dive Into Deep Learning](https://d2l.aivivn.com/) (Especially Chapter 9, Chapter 14)


