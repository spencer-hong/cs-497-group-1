import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator

class RNN_cell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(RNN_cell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.layer1 = nn.Linear(input_size+hidden_size, hidden_size)
    self.layer2 = nn.Linear(hidden_size+hidden_size, hidden_size)
  def forward(self, token, prev_hidden):
    """
    Args:
      token: Tensor [batch_size, input_size]
      prev_hidden: Tensor [2, batch_size, hidden_size] previous hidden state of each layer
    Return:
      output: Tensor [2, batch_size, hidden_size] hidden state of each layer
    """
    input = torch.cat((token, prev_hidden[0]), 1)
    hidden1 = self.layer1(input)
    input = torch.cat((hidden1, prev_hidden[1]), 1)
    hidden2 = self.layer2(input)
    output = torch.stack((hidden1, hidden2), dim=0)
    return output

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, vocab_size):
    super(RNN, self).__init__()
    self.rnn_cell = RNN_cell(input_size, hidden_size)
    self.output = nn.Sequential(
        nn.Linear(hidden_size, vocab_size),
        nn.Softmax()
    )
  def forward(self, sequence, init_hidden):
    """
    Args:
      sequence: Tensor [sequence_length, batch_size, input_size]
      init_hidden: Tensor [n_layer=2, batch_size, hidden_size]
    Output:
      Distribution: Tensor [vocab_size] assign probability to each word
    """
    for i in range(len(sequence)):
      init_hidden = self.rnn_cell(sequence[i], init_hidden)
    Distribution = self.output(init_hidden[1])#take hidden state of last layer

    return Distribution

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
