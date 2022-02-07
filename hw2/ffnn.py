import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime

torch.manual_seed(100)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FFNNParams():

    self.context_size = 4 # since the sliding window is 5
    self.embedding_dim = 100 # n x 100 embedding space
    self.hiddn_dim = 100 # an arbitrarily chosen dimension
    self.cpu_count = mp.cpu_count()

    self.batch_size = 20
    self.epochs = 20
    self.loss_func = nn.MSELoss()
    self.optimizer = optim.Adam # pass this function later in train() with the model params
    self.learning_rate = 1e-2


class FFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim, device = device):
        super(FFNN, self).__init__()
        self.context_size = context_size 
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size 

        # embedding (input tied with output)
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        # add one more for the bias
        self.linear1 = nn.Linear((self.context_size * self.embedding_dim) + 1, self.hidden_dim, bias = False)
        self.hidden = nn.Linear((self.hidden_dim, self.hidden_dim))
        self.linear2 = nn.Linear(self.hidden_dim, self.vocab_size, bias = False)
        
    def forward(self, input):
        # originally (words_input, embedding_dimension), but reshape to (1 x (words_input * embedding_dimension))
        embedding = self.embeddings(inputs).view((-1, self.context_size * self.embedding_dim))

        output_linear1 = torch.tanh(self.linear1(embedding))


        output_hidden = torch.tanh(self.hidden(output_linear1))


        output_linear2 = torch.tanh(self.linear2(output_hidden))

        return F.soft_max(output_linear2, dim = 1)
def prepare_dataset():
    raise NotImplementedError

def prepare_loader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)



def train(dataloader,  model, loss_func, optimizer):
    model.train()

    # perplexity, loss
    values_to_track = defaultdict(list)

    original_time = datetime.now()

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        pred = model(X)

        loss = loss_func(pred, y.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
