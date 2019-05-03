import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		self.hidden_layer = nn.Linear(178, 16)
		self.output_layer = nn.Linear(16, 5)

	def forward(self, x):
		sigmoid = torch.nn.Sigmoid()
		x = sigmoid(self.hidden_layer(x))
		x = self.output_layer(x)
		return x

class MyMLP_Improved(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		self.hidden_layer1 = nn.Linear(178, 32)
		self.hidden_layer2 = nn.Linear(32, 32)
		self.hidden_layer3 = nn.Linear(32, 16)
		self.hidden_layer4 = nn.Linear(16, 16)
		self.hidden_layer5 = nn.Linear(16, 8)
		self.output_layer = nn.Linear(8, 5)

	def forward(self, x):
		x = F.relu(self.hidden_layer1(x))
		x = F.relu(self.hidden_layer2(x))
		x = F.relu(self.hidden_layer3(x))
		x = F.relu(self.hidden_layer4(x))
		x = F.relu(self.hidden_layer5(x))
		x = self.output_layer(x)
		return x

class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(6, 16, 5)
		self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		self.fc2 = nn.Linear(128, 5)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 41)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

class MyCNN_Improved(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=5)
		self.pool = nn.AvgPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(10, 16, 5)
		self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		self.fc2 = nn.Linear(128, 5)

	def forward(self, x):
		x = (self.pool(F.relu(self.conv1(x))))
		x = (self.pool(F.relu(self.conv2(x))))
		x = x.view(-1, 16 * 41)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=1, batch_first=True, dropout=0.5)
		self.fc = nn.Linear(in_features=32, out_features=5)


	def forward(self, x):
		x, _ = self.rnn(x)
		x = F.tanh(x[:, -1, :])
		x = self.fc(x)
		return x

class MyRNN_Improved(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=128, num_layers=1, batch_first=True, dropout=0.5)
		self.fc = nn.Linear(in_features=128, out_features=5)


	def forward(self, x):
		x, _ = self.rnn(x)
		x = F.tanh(x[:, -1, :])
		x = self.fc(x)
		return x

class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()

		self.fc1 = nn.Linear(dim_input, 32)
		self.batch_first = True
		self.rnn = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True, dropout=0.5)
		self.fc2 = nn.Linear(in_features=16, out_features=2)
		# You may use the input argument 'dim_input', which is basically the number of features

	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		x, lengths = input_tuple
		emb = self.fc1(x)
		g, _ = self.rnn(emb)
		x = F.tanh(g[:, -1, :])
		x = (self.fc2(x))
		return x

class MyVariableRNN_Initial(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()

		self.fc1 = nn.Linear(dim_input, 32)
		self.batch_first = True
		self.rnn = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True, dropout=0.5)
		self.fc2 = nn.Linear(in_features=16, out_features=2)
		# You may use the input argument 'dim_input', which is basically the number of features

	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		x, lengths = input_tuple
		batch_size, max_len = x.size()[:2]

		emb = self.fc1(x)

		packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)

		g, _ = self.rnn(packed_input)

		alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)
		mask = Variable(torch.FloatTensor(
			[[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2),
						requires_grad=False)

		seqs = F.tanh((alpha_unpacked * mask)[:, -1, :])

		seqs = self.fc2(seqs)

		return seqs