import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt
from utils import init_weights

torch.manual_seed(100)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#convert data from word to integer representations
def LoadTrainData(filename):
    file = open(filename, 'r', encoding='utf8')
    content_list = file.read().split(" ")
    word2int = {}
    ix = 0
    for word in content_list:
        if word not in word2int:
            word2int[word]=ix
            ix+=1
            
    encoded_list = []
    for word in content_list:
        encoded_list.append(word2int[word])
    
    return(encoded_list, word2int)

def LoadTestData(filename, word2int):
    file = open(filename, 'r', encoding='utf8')
    content_list = file.read().split(" ")
    encoded_list = []
    for word in content_list:
        if word in word2int:
            encoded_list.append(word2int[word])
        else:
            encoded_list.append(word2int['<unk>'])
    
    return encoded_list


class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window
        
    def __getitem__(self, index):
        x = torch.tensor(self.data[index:index+self.window]) #x = the five context words
        y = torch.tensor(self.data[index+self.window+1]) #y = the target word to be predicted
        return x,y
    
    def __len__(self):
        return len(self.data)-self.window


class FFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim, device = device):
        super(FFNN, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size 

        # embedding (input tied with output)
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.linear1 = nn.Linear((self.context_size * self.embedding_dim), self.hidden_dim, bias = False)
        self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.vocab_size, bias = False)
        init_weights(self)
        
    def forward(self, inputs):
        # originally (words_input, embedding_dimension), but reshape to (1 x (words_input * embedding_dimension))
        embedding = self.embeddings(inputs).view((-1, self.context_size * self.embedding_dim))
        output_linear1 = torch.tanh(self.linear1(embedding))
        output_hidden = torch.tanh(self.hidden(output_linear1))
        output_linear2 = torch.tanh(self.linear2(output_hidden))

        return F.softmax(output_linear2, dim = 1)

def prepare_loader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

def train(dataloader, model, loss_func, optimizer):
    model.train()
    train_loss = []
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_func(pred, y.unsqueeze(1))

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        
        if batch % 10==0:
            current = batch*len(X)
            print(f"loss: {loss:>6f} [{current:>5d}/{17000}]")
            train_loss.append(loss.item())
            
    return(train_loss)

def test(dataloader, model, loss_func):
    size = len(dataloader)
    num_batches = size/20 
    model.eval()
    test_loss=0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred, y.unsqueeze(1)).item()
            
    test_loss /= num_batches
    print(f"Avg Loss: {test_loss:>8f}\n")
    return(test_loss)


if __name__ == "__main__":
    #convert text to integer representations
    loadtrain, word_to_int_dict = LoadTrainData('wiki.train.txt')
    loadvalid = LoadTestData('wiki.valid.txt', word_to_int_dict)
    loadtest = LoadTestData('wiki.test.txt', word_to_int_dict)
    
    #specify some parameters
    window_size=5
    vocab_size = len(word_to_int_dict)
    embedding_dim_size = 100
    hidden_dim_size = 100
    
    #load data
    train_data = WikiDataset(loadtrain, window_size)
    valid_data = WikiDataset(loadvalid, window_size)
    test_data = WikiDataset(loadtest, window_size)
    
    train_loader = prepare_loader(train_data, batch_size=20, num_workers=mp.cpu_count())
    test_loader = prepare_loader(test_data, batch_size=20, num_workers=mp.cpu_count())
    
    #define model
    ffmodel = FFNN(vocab_size, embedding_dim_size, window_size, hidden_dim_size)
    print(ffmodel)
    
    #specify some more parameters
    loss_func = nn.CrossEntropyLoss()
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(ffmodel.parameters(), lr=learning_rate)
    epochs = 20
    
    #run training and store loss values
    train_loss = []
    test_loss = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        losses = train(train_loader, ffmodel, loss_func, optimizer)
        train_loss.append(losses)
        test_loss.append(test(test_loader, ffmodel, loss_func))
        
    #calculate perplexity values from loss values
    train_perplexity = []
    test_perplexity = []
    for loss in train_loss:
        train_perplexity.append(torch.exp(loss))
    for loss in test_loss:
        test_perplexity.append(torch.exp(loss))
    
    #plot perplexity
    plt.plot([i for i in range(len(train_perplexity))], torch.tensor(train_perplexity).mean(axis=1))
    plt.plot([i for i in range(len(test_perplexity))], test_perplexity)