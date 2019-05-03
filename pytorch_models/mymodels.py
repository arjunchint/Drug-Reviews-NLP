import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

HIDDEN_LAYER_SIZE=16

class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		self.fc1 = nn.Linear(178, 64)
		self.relu = nn.SELU()
		self.fc2 = nn.Linear(64, 5)
	def forward(self, x):
		out_one = self.fc1(x)
		out_two = self.relu(out_one)
		return self.fc2(out_two)


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
		self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
		self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
		self.fc1 = nn.Linear(in_features=16*41, out_features=256)
		self.fc2 = nn.Linear(in_features=256, out_features=5)
	def forward(self, x):
		out_one = self.pool1(F.relu(self.conv1(x.unsqueeze(1))))
		out_two = self.pool2(F.relu(self.conv2(out_one)))
		x = out_two.view(-1, 16 * 41)
		return self.fc2(torch.tanh(self.fc1(x)))
	# first conv 4450, 

class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.gru_layer = nn.GRU(178, 48, num_layers=1, batch_first=True)
		self.fc1 = nn.Linear(48, 5)
		#self.fc2 = nn.Linear(16, 5)
	def forward(self, x):
		x, _ = self.gru_layer(x.unsqueeze(1))
		x = self.fc1(x[:, -1, :])
		return x

#DEFAULT WORKING RNN MODEL
"""
class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.fc1 = nn.Linear(dim_input, 32)
		self.relu = nn.Tanh()
		self.gru_layer = nn.GRU(32, 16, num_layers=1)
		self.fc2 = nn.Linear(16, 2)
	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		seqs, lengths = input_tuple
		out_one = self.fc1(seqs)
		pack = pack_padded_sequence(out_one, lengths, batch_first=True)
		x, _ = self.gru_layer(pack)
		tmp = pad_packed_sequence(x)
		outputs, output_lengths = tmp
		x = torch.tanh(outputs[-1])
		return self.fc2(x)
"""


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.fc1 = nn.Linear(dim_input, 128)
		self.gru_layer = nn.GRU(128, 64, num_layers=1)
		self.fc2 = nn.Linear(64, 32)
		self.gru_layer_two = nn.GRU(32, 8, num_layers=1)
		self.fc3 = nn.Linear(8, 2)
	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		seqs, lengths = input_tuple
		out_one = torch.tanh(self.fc1(seqs))
		pack = pack_padded_sequence(out_one, lengths, batch_first=True)
		x, _ = self.gru_layer(pack)
		tmp = pad_packed_sequence(x)
		outputs, output_lengths = tmp
		y = self.fc2(outputs)
		x, _ = self.gru_layer_two(y)
		y = self.fc3(x[-1])
		return y