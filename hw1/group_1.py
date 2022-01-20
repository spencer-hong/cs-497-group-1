import collections
from nltk.tokenize import word_tokenize
unseen_token = 0
class my_corpus():
    def __init__(self, corpus):
        super().__init__() 
        #load tokenized corpus to create vocabulary
        self.corpus = corpus

        #get unique tokens in corpus and their counts
        self.word_count = collections.Counter(corpus)

        #save word tokens to its corresponding int value and vice versa
        self.word2int = {'<unseen>':unseen_token}
        self.int2word = {unseen_token:'<unseen>'}
        self.n_word = 1 #total number of unique word tokens in corpus
        for i, word in enumerate(self.word_count):
          self.word2int[word] = i + 1
          self.int2word[i+1] = word
          self.n_word = i + 2

        print('setting vocabulary according to corpus')
    
    def encode_as_ints(self, sequence):
        
        int_represent = []
        # if input sentence is a string, we will tokenize it
        if type(sequence) == str:
          sequence = word_tokenize(sequence.lower())
        
        for word in sequence:
            #if encounter tokens not in vocabulary, we map it to unseen token
          if word not in self.word2int:
            int_represent.append(unseen_token)
          else:
            int_represent.append(self.word2int[word])
        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')
        
        return(int_represent)
    
    def encode_as_text(self,int_represent):

        text = []
        for i in int_represent:
          if i not in self.int2word:
            text.append(self.int2word[unseen_token])
          else:
            text.append(self.int2word[i])
        text = ' '.join(text)
        
        print('encode this list', int_represent)
        print('as a text sequence.')
        
        return(text)
    
def main():
    corpus_path = "training_set.txt"
    tokens_corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
      for line in f:
        tokens_corpus.extend(line)
    corpus = my_corpus(tokens_corpus)
    
    text = input('Please enter a test sequence to encode and recover: ')
    print(' ')
    ints = corpus.encode_as_ints(text)
    print(' ')
    print('integer encodeing: ',ints)
    
    print(' ')
    text = corpus.encode_as_text(ints)
    print(' ')
    print('this is the encoded text: %s' % text)
    
if __name__ == "__main__":
    main()
