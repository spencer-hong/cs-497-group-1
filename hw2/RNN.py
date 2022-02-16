import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
from utils import init_weights


class RNN_cell(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(RNN_cell, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(self, token, prev_hidden):
        """
        Args:
            token: Tensor [batch_size, embedding_size]
            prev_hidden: Tensor [2, batch_size, hidden_size] previous hidden state of each layer
        Return:
            output: Tensor [2, batch_size, hidden_size] hidden state of each layer
        """
        input_1 = torch.cat((token, prev_hidden[0]), 1)
        hidden_1 = self.linear_1(input_1)
        input_2 = torch.cat((hidden_1, prev_hidden[1]), 1)
        hidden_2 = self.linear_2(input_2)
        output = torch.stack((hidden_1.unsqueeze(0), hidden_2.unsqueeze(0)), dim=0)
        return output


class RNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_dim=100, device='cpu'):
        super(RNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn_cell = RNN_cell(embedding_dim, hidden_size)
        self.output = nn.Sequential(
                nn.Linear(hidden_size, vocab_size),
                nn.Softmax()
        )
        init_weights(self)
        self.to(device)

    def forward(self, sequence):
        """
        Args:
            sequence: LongTensor [batch_size, sequence_length]
        Output:
            probabilities: Tensor [vocab_size] assign probability to each word
        """
        embeddings = self.embeddings(sequence)
        batch_size, sequence_length = sequence.shape
        # change below to gen hid
        init_hidden = generate_initial_hidden_state(batch_size)
        for i in range(sequence_length):
            init_hidden = self.rnn_cell(embeddings[:, i, :], init_hidden)
        probabilities = self.output(init_hidden[1]) #take hidden state of last layer

        return probabilities

    def generate_initial_hidden_state(self, batch_size):
        state = torch.stack(
            (nn.init.uniform_(torch.ones(batch_size, self.hidden_size), a=-0.1, b=0.1), nn.init.uniform_(torch.ones(batch_size, self.hidden_size), a=-0.1, b=0.1)),
        dim=0)
        state.to(self.device)
        return state


def build_vocab(corpus):
    """
    Args:
        corpus: List[] list of tokens
    Return:
        vocab: torchtext.vocab.Vocab
    """
    vocab = build_vocab_from_iterator(corpus)
    #add output vocabulary token, its correxponding value is len(vocab)+1
    
    vocab.set_default_index(len(vocab)+1)
    return vocab

