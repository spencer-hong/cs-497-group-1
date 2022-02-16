import torch
from torch import nn, LongTensor
from torch.utils.data import Dataset


# cross-model utils
def init_weights(m):
	for p in m.parameters():
		p = nn.init.uniform_(p, a=-0.1, b=0.1)

def model_to_device(m, device):
	for p in m.parameters():
		p = p.to(device)

# data reading utils
class TextDataset(Dataset):
	def __init__(self, text_file, model='RNN', batch_size=20, train_dataset=None, sliding_window=30, debugging=False):
		self.sliding_window = sliding_window
		self.batch_size = batch_size

		if model not in ['RNN', 'FFNN']:
			raise ValueError('model parameter must be set to "RNN" or "FFNN" -- also FFNN example generation not implemented')
		self.model = model

		# read file and preprocess
		with open(text_file, encoding='utf-8') as f:
			raw_text = self.preprocess_readlines(f.readlines())
		self.split_text = self.tokenize_raw(raw_text)

		if debugging:
			self.split_text = self.split_text[:100]
		
		# get vocabulary
		if train_dataset is None:
			self.vocabulary, self.vocabulary_size = self.generate_vocabulary()
			self.token_to_int, self.int_to_token = self.generate_vocabulary_mappings()
		else:
			self.vocabulary, self.vocabulary_size = train_dataset.return_vocabulary()
			self.token_to_int, self.int_to_token = train_dataset.return_token_to_int(), train_dataset.return_int_to_token()
		
		self.encoded_text = self.encode()

		if self.model == 'RNN':
			self.input_windows, self.labels = self.generate_examples_rnn()
		elif self.model == 'FFNN':
			self.input_windows, self.labels = self.generate_examples_ffnn()

		assert len(self.input_windows) == len(self.labels)

	# Dataset functions
	def __len__(self):
	    return len(self.labels)

	def __getitem__(self, index):
		return self.input_windows[index], self.labels[index]

	# utils
	def preprocess_readlines(self, readlines_list):
		text = [line.replace('\n', '').strip() for line in readlines_list]
		text = [line for line in text if len(line)]
		return ' '.join(text)

	def tokenize_raw(self, raw):
		return raw.split()

	# set/generate methods
	def generate_vocabulary(self):
		vocabulary_unique = list(set(self.split_text))
		if '<oov>' not in vocabulary_unique:
			vocabulary_unique.append('<oov>')
		return vocabulary_unique, len(vocabulary_unique)

	def generate_vocabulary_mappings(self):
		return {k: v for v, k in enumerate(self.vocabulary)}, {k: v for k, v in enumerate(self.vocabulary)}

	def encode(self):
		corpus_encoding = []
		for token in self.split_text:
			if token in self.vocabulary:
				corpus_encoding.append(self.token_to_int[token])
			else:
				corpus_encoding.append(self.token_to_int['<oov>'])
		return corpus_encoding

	def decode(self, encoded_text):
		decoded_text = []
		for encoded_token in self.encoded_text:
			if encoded_token in self.int_to_token:
				decoded_text.append(self.int_to_token[encoded_token])
			else:
				raise ValueError('Decode given numericalized token outside of vocabulary size.')
		return decoded_text

	def generate_examples_rnn(self):
		x, y = [], []
		corpus_length = len(self.encoded_text)
		for ix in range(0, corpus_length, self.sliding_window):
			if ix + self.sliding_window + self.batch_size > corpus_length:
				break
			for jx in range(self.batch_size):
				start_index = ix + jx
				end_index = start_index + self.sliding_window
				x.append(self.encoded_text[start_index:end_index])
				y.append(self.encoded_text[end_index])

		return x, y

	def generate_examples_ffnn(self):
		raise NotImplementedError('no implementation for generating examples for ffnn')

	# get methods
	def return_vocabulary(self):
		return self.vocabulary, self.vocabulary_size

	def return_vocabulary_size(self):
		return self.vocabulary_size

	def return_token_to_int(self):
		return self.token_to_int

	def return_int_to_token(self):
		return self.int_to_token


def text_collate_fn(batch):
	sources = LongTensor([example[0] for example in batch])
	targets = LongTensor([example[1] for example in batch])
	return sources, targets


def compute_perplexity(loss):
	return round(torch.exp(loss).item(), 4)


def save_perplexity(ppl_list, output_dir, file_name):
	if not output_dir.endswith('/'):
		output_dir += '/'
	if '.' not in file_name:
		file_name += '.csv'
	with open(output_dir + file_name, 'w') as f:
		f.write('\n'.join(['perplexity'] + [str(val) for val in ppl_list]))
