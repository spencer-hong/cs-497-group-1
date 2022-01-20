from nltk.tokenize import word_tokenize, MWETokenizer

################Q1
# read data
data_path = 'source_text.txt'

corpus = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        corpus.append(line.lower())

sentences = []
for line in corpus[:1000]:
    sentences.append(word_tokenize(line))

################# Q2

################# Q3

################# Q4

from collections import Counter
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')

# assuming that splits will be lists of list: [example[tokens]]

def apply_frequency_threshold(split, threshold=3):
    # flatten split for easier processing
    flattened_split = [token for sentence in split for token in sentence]

    # create Counter from flattened split
    token_counter = Counter(flattened_split)

    # TODO: does frequency threshold of 3 mean that we keep all tokens appearing 3 or more times? or is it 4 or more?
    tokens_that_fail_threshold = [k for k, v in token_counter.items() if v < 3 ]

    # convert oov tokens into <unk>s
    threshold_output = []
    for example in split:
        threshold_output.append(['<unk>' if token in tokens_that_fail_threshold else token for token in example])

    # return thresholded output and # of tokens that fail threshold (for metric 5)
    return threshold_output, {'num_types_mapped_to_unk': len(tokens_that_fail_threshold)}

def calculate_metrics(split):
    flattened_split = [token for sentence in split for token in sentence]
    print(flattened_split)
    split_counter = Counter(flattened_split)

    def tokens_per_split(fs):
        # length of flattened split
        return len(fs)

    def vocab_size(fs):
        # length of flattened split as a set
        return len(set(fs))

    def count_unk_tokens(sc):
        # can use counter here
        return sc['<unk>']

    def count_oov_words(fs):
        # TODO: how does this differ from the metric above? is it to count the tokens like <year> etc.?
        raise NotImplementedError('needs resolving')

    def count_stopwords(sc):
        # use counter and nltk list of stopwords
        return sum([v for k, v in sc.items() if k in STOPWORDS])

    # custom metric 1: average number of tokens per passage (including <start_of_passage> and <end_of_passage> tokens)
    def avg_tokens_per_passage(fs, sc):
        return round(len(fs)/sc['<start_of_passage>'], 2)

    # custom metric 2: average vocabulary overlap between passages (including all special tokens like <unk>, <year>)
    def avg_vocabulary_overlap_between_passages(fs):
        def get_vocabulary_overlap(p1s, p2s):
            return len(p1s.intersection(p2s))

        overlap_sum, overlap_counter = 0, 0
        passage_groups = [(passage.strip() + ' <end_of_passage>').split() for passage in ' '.join(fs).split('<end_of_passage>')]
        for group_ix, passage_1 in enumerate(passage_groups):
            passage_1_set = set(passage_1)
            for passage_2 in passage_groups[group_ix + 1:]:
                passage_2_set = set(passage_2)
                overlap_sum += get_vocabulary_overlap(passage_1_set, passage_2_set)
                overlap_counter += 1

        return round(overlap_sum / overlap_counter, 2)

    return {
        'number_of_tokens_per_split': tokens_per_split(flattened_split),
        'vocab_size': vocab_size(flattened_split),
        'number_of_unk_tokens': count_unk_tokens(split_counter),
        # 'number_of_oov_words': count_oov_words(flattened_split),
        'number_of_stopwords': count_stopwords(split_counter),
        'avg_tokens_per_passage': avg_tokens_per_passage(flattened_split, split_counter),
        'avg_vocab_overlap_between_passages': avg_vocabulary_overlap_between_passages(flattened_split)
    }

# TODO: apply frequency threshold on full corpus, or individual splits? (I think former)
# sentences, num_types_mapped_to_unk = apply_frequency_threshold(sentences)
# split_metrics = dict(calculate_metrics(sentences), **num_types_mapped_to_unk)
# print(split_metrics)

################# Q6
