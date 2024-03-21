import numpy as np
import pickle
import librosa
import sounddevice as sd
import tensorflow as tf
from pathlib import Path

'''Performs Inverse Short-Time Fourier Transform'''


def inverse_stft_transform(stft_features, window_length, overlap):
	return librosa.istft(stft_features, win_length=window_length, hop_length=overlap)


def features_to_audio(features, phase, window_length, overlap, clean_mean=None, clean_std=None):
	# Rescale the output back to original range
	if clean_mean and clean_std:
		features = clean_std * features + clean_mean

	phase = np.transpose(phase, (1, 0))
	features = np.squeeze(features)
	# Undo the previously done abs() operation
	features = features * np.exp(1j * phase)

	features = np.transpose(features, (1, 0))
	return inverse_stft_transform(features, window_length=window_length, overlap=overlap)


def play_audio(audio, sampling_rate):
	# ipd.display(ipd.Audio(data=audio, rate=sampling_rate))  # For wav file
	sd.play(audio, sampling_rate, blocking=True)


def add_noise_to_clean_audio(clean_audio, noise_signal):
	if len(clean_audio) >= len(noise_signal):
		# print("Noise signal is smaller than clean audio input. Noise is being replicated.")
		while len(clean_audio) >= len(noise_signal):
			noise_signal = np.append(noise_signal, noise_signal)

	# Take a random portion of noise from the noise file
	ind = np.random.randint(0, noise_signal.size - clean_audio.size)

	noise_part = noise_signal[ind: ind + clean_audio.size]

	speech_power = np.sum(clean_audio ** 2)
	noise_power = np.sum(noise_part ** 2)
	noisy_audio = clean_audio + \
		np.sqrt(speech_power / noise_power) * noise_part
	return noisy_audio


def read_audio(file_path, sampling_rate, normalize=True):
	audio, sr = librosa.load(str(Path(file_path)), sr=sampling_rate)
	if normalize is True:
		div_fac = 1 / np.max(np.abs(audio)) / 3.0
		audio = audio * div_fac
		# audio = librosa.util.normalize(audio)
	return audio, sr


def prepare_input_features(stft_features, num_segments, num_features):
	noisy_stft = np.concatenate(
		[stft_features[:, 0:num_segments - 1], stft_features], axis=1)
	stft_segments = np.zeros(
		(num_features, num_segments, noisy_stft.shape[1] - num_segments + 1))

	for index in range(noisy_stft.shape[1] - num_segments + 1):
		stft_segments[:, :, index] = noisy_stft[:, index:index + num_segments]
	return stft_segments


def get_input_features(identifier_list):
	identifiers = []
	for noisy_stft_magnitude_features in identifier_list:
		# For CNN, input feature is consecutive 8 noisy
		# STFT magnitude vectors: 129 x 8
		input_features = prepare_input_features(
			noisy_stft_magnitude_features)
		identifiers.append(input_features)

	return identifiers


def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		# BytesList won't unpack a string from an EagerTensor.
		value = value.numpy()
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_tf_feature(noisy_stft_magnitude_features, clean_stft_magnitude, noise_stft_phase):
	noisy_stft_magnitude_features = noisy_stft_magnitude_features.astype(
		np.float32).tostring()
	clean_stft_magnitude = clean_stft_magnitude.astype(np.float32).tostring()
	noise_stft_phase = noise_stft_phase.astype(np.float32).tostring()

	example = tf.train.Example(features=tf.train.Features(feature={
		'noise_stft_phase': _bytes_feature(noise_stft_phase),
		'noisy_stft_magnitude_features': _bytes_feature(noisy_stft_magnitude_features),
		'clean_stft_magnitude': _bytes_feature(clean_stft_magnitude)}))
	return example