import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset


def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.

	df = pd.read_csv(path)

	target = df['y'].as_matrix()

	target = target - 1

	data = df.loc[:, 'X1':'X178'].as_matrix()

	if model_type == 'MLP':
		# data = torch.zeros((2, 2))
		# target = torch.zeros(2)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')),torch.from_numpy(target.astype('long')))
	elif model_type == 'CNN':
		# data = torch.zeros((2, 2))
		# target = torch.zeros(2)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(1),torch.from_numpy(target.astype('long')))
	elif model_type == 'RNN':
		# data = torch.zeros((2, 2))
		# target = torch.zeros(2)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(2),torch.from_numpy(target.astype('long')))
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""
		self.seqs = []
		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		for sequence, dataLabel in zip(seqs, labels):
			sequence = sequence
			value = []
			dataRow = []
			dataColumn = []
			for i, visit in enumerate(sequence):
				for code in visit:
					if code < num_features:
						value.append(1.0)
						dataRow.append(i)
						dataColumn.append(code)

			self.seqs.append(sparse.coo_matrix((np.array(value, dtype=np.float32), (np.array(dataRow), np.array(dataColumn))),
										shape=(len(sequence), num_features)))
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence

	batch_seq, batch_label = zip(*batch)

	num_features = batch_seq[0].shape[1]
	seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
	max_length = max(seq_lengths)

	sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
	sorted_seqs = []
	sorted_labels = []

	for i in sorted_indices:
		length = batch_seq[i].shape[0]

		if length < max_length:
			padded = np.concatenate(
				(batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
		else:
			padded = batch_seq[i].toarray()

		sorted_seqs.append(padded)
		sorted_labels.append(batch_label[i])

	seq_tensor = np.stack(sorted_seqs, axis=0)
	label_tensor = torch.LongTensor(sorted_labels)
	return (torch.from_numpy(seq_tensor), list(sorted_lengths)), label_tensor
