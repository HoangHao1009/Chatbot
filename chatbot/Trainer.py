import torch
import torch.nn as nn
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Trainer:
    def __init__(self, input_variable, lengths, target_variable, mask, max_target_len,
                 encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
                 batch_size, clip, max_length, teacher_forcing_ratio):
        self.max_length = max_length
        self.input_variable = input_variable
        self.lengths = lengths
        self.target_variable = target_variable
        self.mask = mask
        self.max_target_len = max_target_len
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.batch_size = batch_size
        self.clip = clip
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio

    
    def train(self):
        def maskNLLLoss(inp, target, mask):
            nTotal = mask.sum()
            crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
            loss = crossEntropy.masked_select(mask).mean()
            loss = loss.to(device)
            return loss, nTotal.item()

        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Set device options
        input_variable = self.input_variable.to(device)
        target_variable = self.target_variable.to(device)
        mask = self.mask.to(device)
        # Lengths for RNN packing should always be on the CPU
        lengths = self.lengths.to("cpu")

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(self.batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(self.max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(self.max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = decoder_input.to(device)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        # Perform backpropagation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_losses) / n_totals
    

class TrainIter:
    def __init__(self, modeldatapreparer, encoder, decoder, 
                 encoder_optimizer, decoder_optimizer, embedding,
                 n_iteration, batch_size, print_every, 
                 clip):
        self.modeldatapreparer = modeldatapreparer
        self.print_every = print_every
        self.batch_size = batch_size
        self.n_iteration = n_iteration
        self.trainer = None
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.embedding = embedding
        self.clip = clip
    
    def run(self, max_length, teacher_forcing_ratio):
        training_batches = [self.modeldatapreparer.batch2TrainData(self.batch_size) for _ in range(self.n_iteration)]
        print('Initializing...')
        start_iteration = 1
        print_loss = 0
        print('Training...')
        for iteration in range(start_iteration, self.n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            input_variable, lengths, target_variable, mask, max_target_len = training_batch
            self.trainer = Trainer(input_variable, lengths, target_variable, mask, max_target_len,
                                   self.encoder, self.decoder, self.embedding, 
                                   self.encoder_optimizer, self.decoder_optimizer,
                                   self.batch_size, self.clip, max_length, teacher_forcing_ratio)
            loss = self.trainer.train()
            print_loss += loss
            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                print(f'Iteration: {iteration}, Percent complete: {iteration / self.n_iteration * 100}, Avg loss: {print_loss_avg}')
                print_loss = 0