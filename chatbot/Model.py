import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers = 1, dropout = 0):
        super().__init__()
        self.embedding = embedding #emb layers (will be set emb-dim = hidden_size)
        self.n_layers = n_layers #num of encoder layers
        self.hidden_size = hidden_size #size of hidden_state
        
        #GRU: input_size = emb_size = hidden_size 
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout = (0 if n_layers == 1 else dropout),
                          bidirectional = True)
        
    def forward(self, input_seq, input_lengths, hidden = None):
        #input_seq: (max_lengths, batch_size)
        #input_lengths: (batch_size)
        #hidden (hidden_state): (n_layers * num_direction, batch_size, hidden_size)

        #because emb_size = hidden_size:
        embedded = self.embedding(input_seq) #(max_lengths, batch_size, hidden_size)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths) # remain shape
        outputs, hidden = self.gru(packed, hidden)
        # -> because bidirectional:
        #-> output: (max_lengths, batch_size, hidden_size * 2)
        #-> hidden: remain shape
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs) #remain_shape
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        #-> output: (max_lengths, batch_size, hidden_size)
        #-> hidden: (n_layers * num_directions, batch_size, hidden_size)

        return outputs, hidden
    
class DotAttn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch_size, hidden_size) -> one_word with batch_size after embed
        # encoder_output: (max_lengths, batch_size, hidden_size)
        dot = hidden * encoder_outputs # (max_lengths, batch_size, hidden_size)
        attn_energies = torch.sum(dot, dim = 2) # (max_lengths, batch_size)
        attn_energies = attn_energies.t() # (batch_size, max_lenghts)

        return F.softmax(attn_energies, dim = 1).unsqueeze(1)
        #calculate softmax based on max_lengths
        #-> (batch_size, 1, max_lengths)
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size,
                 n_layers = 1, dropout = 0.1):
        super().__init__()
        self.hidden_size = hidden_size # size hidden_state
        self.output_size = output_size # size vocav
        self.n_layers = n_layers # num decoder layers
        self.dropout = dropout

        self.embedding = embedding # use both for encoder and decoder
        self.embedding_dropout = nn.Dropout(dropout)

        #decoder gru
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout = (0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = DotAttn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        #input_seq: (1, batch_size) -> one time step
        #last_hidden: (n_layers * num_directions, batch_size, hidden_size)
        #encoder_outputs: (max_lengths, batch_size, hidden_size)

        embedded = self.embedding(input_seq) # (1, batch_size, hidden_size)
        embedded = self.embedding_dropout(embedded) #remain shape
        rnn_output, hidden = self.gru(embedded, last_hidden)
        #-> rnn_output: (1, batch_size, hidden_size)
        #-> hidden: (n_layers * num_directions (=1), batch_size, hidden_size)

        attn_weights = self.attn(rnn_output, encoder_outputs) # (batch_size, 1, max_length)
        context = attn_weights @ encoder_outputs.transpose(0, 1)
        #encoder_outputs.transpose(0, 1) -> (batch_size, max_lenghts, hidden_size)
        #-> (batch_size, 1, max_length) @ (batch_size, 1, max_lengths)
        #-> context = (batch_size, 1, hidden_size)
        rnn_output = rnn_output.squeeze(0) # (batch_size, hidden_size)
        context = context.squeeze(1) # (batch_size, hidden_size)
        concat_input = torch.cat((rnn_output, context), 1) # (batch_size, hidden_size * 2)
        concat_output = torch.tanh(self.concat(concat_input)) # (batch_size, hidden_size)
        output = self.out(concat_output) # (batch_size, output_size (vocab_size))
        output = F.softmax(output, dim = 1) # (batch_size, vocab_size (probs))
        return output, hidden
        # output: (batch_size, voc.num_words)
        # hidden: (n_layers * num_directions, batch_size, hidden_size)