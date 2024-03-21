import pandas as pd
import numpy as np
import os

np.random.seed(999)


class MozillaVoiceDataset:
	def __init__(self, base_path, *, val_dataset_size):
		self.base_path = base_path
		self.val_dataset_size = val_dataset_size

	def get_mozilla_voice_files(self, dataframe_name='train.tsv'):
		mozilla_metadata = pd.read_csv(os.path.join(
			self.base_path, dataframe_name), sep='\t')
		clean_files = mozilla_metadata['path'].values
		np.random.shuffle(clean_files)
		print("Total number of files to be trained:", len(clean_files))
		return clean_files

	def get_train_val_files(self):
		clean_files = self.get_mozilla_voice_files(
			dataframe_name='train.tsv')

		clean_files = [os.path.join(
			self.base_path, 'clips', filename) for filename in clean_files]

		clean_files = clean_files[:-self.val_dataset_size]
		clean_val_files = clean_files[-self.val_dataset_size:]
		print("# Number of clean files for training:", len(clean_files))
		print("# Number of clean files for validation:", len(clean_val_files))
		return clean_files, clean_val_files

	def get_test_files(self):
		clean_files = self.get_mozilla_voice_files(
			dataframe_name='test.tsv')

		clean_files = [os.path.join(
			self.base_path, 'clips', filename) for filename in clean_files]

		print("# Number of clean files for testing:", len(clean_files))
		return clean_files