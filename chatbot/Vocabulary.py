import re
import itertools
import random
import torch
import unicodedata


PAD_token = 0
SOS_token = 1
EOS_token = 2


class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

class TextProcessor:
    def __init__(self):
        pass
    @staticmethod
    def normalizString(s, unicodeToAscii = True):
        def unicodeToAscii(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        if unicodeToAscii:
            s = unicodeToAscii(s.lower().strip())
        S = s.lower()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s
    def filterPairs(max_length, pairs):
        def filterPair(p):
            return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length
        pairs = [pair for pair in pairs if filterPair(pair)]
        return pairs

class VocAndPairLoader:
    def __init__(self):
        self.voc = Voc()
        self.pairs = []

    def PrepareDataFrom(self, datafile, filterPairslength, encoding = 'utf-8'):
        print('Reading lines...')
        lines = open(datafile, encoding = encoding).read()
        lines = lines.strip().split('\n')
        pairs = [[TextProcessor.normalizString(s) for s in l.split('\t')] for l in lines]
        pairs = TextProcessor.filterPairs(filterPairslength, pairs)
        for pair in pairs:
            self.voc.addSentence(pair[0])
            self.voc.addSentence(pair[1])
            self.pairs.append(pair)

class ModelDatafunction:
    def __init__(self):
        pass
    @staticmethod
    def indexesFromSentence(voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

    def zeroPadding(l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binaryMatrix(l, value=PAD_token):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == value:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m
    
class ModelData:
    def __init__(self, voc, pairs):
        self.voc = voc
        self.pairs = pairs
        self.keep_pairs = None

    def trimRareWords(self, min_count):
        self.voc.trim(min_count)
        keep_pairs = []
        for pair in self.pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            for word in input_sentence.split(' '):
                if word not in self.voc.word2index:
                    keep_input = False
                    break
            for word in output_sentence.split(' '):
                if word not in self.voc.word2index:
                    keep_output = False
                    break

            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(self.pairs), len(keep_pairs), len(keep_pairs) / len(self.pairs)))
        self.keep_pairs = keep_pairs
    
    def batch2TrainData(self, small_batch_size):
        def inputVar(l):
            indexes_batch = [ModelDatafunction.indexesFromSentence(self.voc, sentence) for sentence in l]
            # indexes_batch = (batch_size, ... len of each sent)
            lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
            # lengths = (batch_size)
            padList = ModelDatafunction.zeroPadding(indexes_batch)
            padVar = torch.LongTensor(padList)
            # padVar = (max_length, batch_size)
            return padVar, lengths
        def outputVar(l):
            indexes_batch = [ModelDatafunction.indexesFromSentence(self.voc, sentence) for sentence in l]
            max_target_len = max([len(indexes) for indexes in indexes_batch])
            # max_target_len = scalar
            padList = ModelDatafunction.zeroPadding(indexes_batch)
            mask = ModelDatafunction.binaryMatrix(padList)
            mask = torch.BoolTensor(mask)
            padVar = torch.LongTensor(padList)
            #padVar, mask = (max_length, batch_size)
            return padVar, mask, max_target_len

        pair_batch = [random.choice(self.keep_pairs) for _ in range(small_batch_size)]

        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = inputVar(input_batch)
        output, mask, max_target_len = outputVar(output_batch)

        return inp, lengths, output, mask, max_target_len
    
