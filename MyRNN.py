import torch
import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MyRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        # self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=2, nonlinearity='relu', batch_first=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, ct) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # 文本分类问题，将隐藏状态输入全连接层
        hidden.to(self.device)
        hidden = hidden[-1, :, :]
        hidden = self.relu(hidden)
        # hidden = self.dropout(hidden)
        net_output = self.fc(hidden)
        return self.softmax(net_output)
