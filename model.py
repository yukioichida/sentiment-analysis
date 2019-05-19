import torch.nn as nn
import torch


class RecurrentNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_index,
                 bidirectional=True):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

        # The input of lstm is a tensor with embedding_dim as input dimensionality
        self.recurrent_layer = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                                       bidirectional=bidirectional, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        # Getting the recurrent output tensor and transform into a tensor with size of label
        output_rnn_size = hidden_dim * 2  # bidirectional = forward representation concatenated with backward
        self.fc = nn.Linear(output_rnn_size, output_dim)

    def forward(self, text, text_lengths):
        embeddings = self.embedding_layer(text)
        #print(text)
        #print(text_lengths)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeddings, text_lengths)

        packed_output, hidden = self.recurrent_layer(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden_concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        return self.fc(hidden_concat.squeeze(0))


