import os
import pickle
import pandas as pd
from datetime import datetime

PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"

def convert_to_dateTime(dateTimeObject):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	admissionDateStr = str(dateTimeObject)
	return datetime.strptime(admissionDateStr, '%Y-%m-%d %H:%M:%S')

def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	# # TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# # TODO: Read the homework description carefully.
	icd9_str = str(icd9_object)
	if icd9_str.startswith('E'):
		if len(icd9_str) > 4:
			return icd9_str[:4]
		else:
			return icd9_str
	else:
		if len(icd9_str) > 3:
			return icd9_str[:3]
		else:
			return icd9_str

def build_codemap():
	"""
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
	# TODO: We build a code map using ONLY train data. Think about how to construct validation/test sets using this.
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
	df_digits = df_icd9['ICD9_CODE'].apply(convert_icd9)
	df_digits = sorted(set(df_digits))
	codemap = dict(zip(df_digits,range(len(df_digits))))
	return codemap


def create_dataset(path, codemap):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# TODO: 1. Load data from the three csv files
	# TODO: Loading the mortality file is shown as an example below. Load two other files also.
	# TODO: 2. Convert diagnosis code in to unique feature ID.
	# TODO: HINT - use 'convert_icd9' you implemented and 'codemap'.
	# TODO: 3. Group the diagnosis codes for the same visit.
	# TODO: 4. Group the visits for the same patient.
	# TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
	# TODO: Visits for each patient must be sorted in chronological order.

	# TODO: 6. Make patient-id List and label List also.
	# TODO: The order of patients in the three List output must be consistent.
	patientIDAdmissionMap = {}
	admDateMap = {}
	infd = open(os.path.join(path, "ADMISSIONS.csv"), 'r')
	infd.readline()
	for line in infd:
		tokens = line.strip().split(',')
		pid = int(tokens[1])
		admId = int(tokens[2])
		admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
		admDateMap[admId] = admTime
		if pid in patientIDAdmissionMap:
			patientIDAdmissionMap[pid].append(admId)
		else:
			patientIDAdmissionMap[pid] = [admId]
	infd.close()

	admission_diag_map = {}
	infd = open(os.path.join(path, "DIAGNOSES_ICD.csv"), 'r')
	infd.readline()
	for line in infd:
		tokens = line.strip().split(',')
		admId = int(tokens[2])
		dxStr_3digit = convert_icd9(tokens[4])
		if admId in admission_diag_map:
			admission_diag_map[admId].append(dxStr_3digit)
		else:
			admission_diag_map[admId] = [dxStr_3digit]
	infd.close()
	patientid_seq_map = {}
	for pid, admIdList in patientIDAdmissionMap.items():
		if len(admIdList) < 2:
			continue

		sortedList_3digit = sorted([(admDateMap[admId], admission_diag_map[admId]) for admId in admIdList])
		patientid_seq_map[pid] = sortedList_3digit

	pids = []
	dates = []
	df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
	patLabels = []
	seqList = []
	for pid, visits in patientid_seq_map.items():
		pids.append(pid)
		seq = []
		date = []
		for visit in visits:
			date.append(visit[0])
			seq.append(visit[1])
		dates.append(date)
		seqList.append(seq)
		deadoralive = df_mortality.loc[df_mortality['SUBJECT_ID'] == pid, 'MORTALITY'].iloc[0]
		patLabels.append(deadoralive)

	finalSeqList = []
	for patient in seqList:
		newPatient = []
		for visit in patient:
			newVisit = []
			for code in set(visit):
				if code in codemap:
					newVisit.append(codemap[code])
			if len(newVisit) != 0:
				newPatient.append(newVisit)
		finalSeqList.append(newPatient)

	return pids, patLabels, finalSeqList


def main():
	# Build a code map from the train set
	print("Build feature id map")
	codemap = build_codemap()
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
