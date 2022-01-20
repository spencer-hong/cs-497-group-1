import collections
class my_corpus():
    def __init__(self, corpus):
        super().__init__() 
        #load tokenized corpus to create vocabulary
        self.corpus = corpus

        #get unique tokens in corpus and their counts
        self.word_count = collections.Counter(corpus)

        #save word tokens to its corresponding int value and vice versa
        self.word2int = {}
        self.int2word = {}
        self.n_word = 0 #total number of unique word tokens in corpus
        for i, word in enumerate(self.word_count):
          self.word2int[word] = i
          self.int2word[i] = word
          self.n_word = i + 1

        print('setting vocabulary according to corpus')
    
    def encode_as_ints(self, sequence):
        
        int_represent = []
        for word in sequence:
          if word not in self.word2int:
            self.word2int[word] = self.n_word
            self.int2word[self.n_word] = word
          int_represent.append(self.word2int[word])
        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')
        
        return(int_represent)
    
    def encode_as_text(self,int_represent):

        text = []
        for i in int_represent:
          text.append(self.int2word[i])
        text = ' '.join(text)
        
        print('encode this list', int_represent)
        print('as a text sequence.')
        
        return(text)

def main():
    corpus = my_corpus(None)
    
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
