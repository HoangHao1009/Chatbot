import torch
import torch.nn as nn
from chatbot.Vocabulary import TextProcessor, ModelDatafunction

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PAD_token = 0  
SOS_token = 1
EOS_token = 2

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        #encoder_outputs: (max_lengths, batch_size, hidden_size)
        #encoder_hidden: (n_layers * num_directions, batch_size, hidden_size)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1, 1, device = device, dtype = torch.long) * SOS_token
        all_tokens = torch.zeros([0], device = device, dtype = torch.long)
        all_scores = torch.zeros([0], device = device)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )
            #decoder_output: (batch_size, voc.num_words)
            #decoder_hidden: (n_layers * num_directions, batch_size, hidden_size)
            decoder_scores, decoder_input = torch.max(
                decoder_output,
                dim = 1
            )
            #decoder_scores: (batch_size, ) ->max of probs of voc.numwords
            #decoder_input: (batch_size, ) -> index of decoder_scores

            all_tokens = torch.cat((all_tokens, decoder_input), dim = 0) #(batch_size, )
            all_scores = torch.cat((all_scores, decoder_scores), dim = 0) #(batch_size, )

            decoder_input = torch.unsqueeze(decoder_input, 0) #(add extra dim for next loop)

        return all_tokens, all_scores #(max_length, batch_size)


class Chatbot:
    def __init__(self, searcher, vocab):
        self.searcher = searcher 
        self.voc = vocab

    def chatting(self, max_length):
        def evaluate(sentence, max_length):
            indexes_batch = [ModelDatafunction.indexesFromSentence(self.voc, sentence)]
            lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
            input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
            input_batch = input_batch.to(device)
            lengths = lengths.to('cpu')
            tokens, scores = self.searcher(input_batch, lengths, max_length)
            decoded_words = [self.voc.index2word[token.item()] for token in tokens]
            return decoded_words
        
        input_sentence = ''
        while(1):
            try:
                input_sentence = input('Me > ')
                if input_sentence == 'q' or input_sentence == 'quit': break
                input_sentence = TextProcessor.normalizString(input_sentence)
                output_words = evaluate(input_sentence, max_length)
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))
            except KeyError:
                print('Error: unknown word')


