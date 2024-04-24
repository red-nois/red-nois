import librosa
import numpy as np
import math
import multiprocessing
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from feature_extractor import FeatureExtractor
from utils import prepare_input_features, read_audio, get_tf_feature

np.random.seed(999)
tf.random.set_seed(999)


class Dataset:
    def __init__(self, clean_file_paths, noisy_file_paths, **config):
        self.clean_file_paths = clean_file_paths
        self.noisy_file_paths = noisy_file_paths
        self.sample_rate = config["sample_rate"]
        self.overlap = config["overlap"]
        self.window_size = config["window_size"]
        self.max_audio_length = config["max_audio_length"]

    def _random_noisy_file_path(self):
        return np.random.choice(self.noisy_file_paths)

    def _remove_silent_frames(self, audio):
        trimmed_audio = []
        intervals = librosa.effects.split(audio, hop_length=self.overlap, top_db=20)

        for interval in intervals:
            trimmed_audio.extend(audio[interval[0] : interval[1]])
        return np.array(trimmed_audio)

    def _scale_phase_difference(
        self, clean_spectral_magnitude, clean_phase, noisy_phase
    ):
        assert clean_phase.shape == noisy_phase.shape, "Shapes must match."
        return clean_spectral_magnitude * np.cos(clean_phase - noisy_phase)

    def get_noisy_audio(self, *, file_path):
        return read_audio(file_path, self.sample_rate)

    def _randomly_crop_audio(self, audio, duration):
        audio_duration_seconds = librosa.core.get_duration(y=audio, sr=self.sample_rate)

        if duration >= audio_duration_seconds:
            return audio

        audio_duration_ms = math.floor(audio_duration_seconds * self.sample_rate)
        duration_ms = math.floor(duration * self.sample_rate)
        idx = np.random.randint(0, audio_duration_ms - duration_ms)
        return audio[idx : idx + duration_ms]

    def _add_noise_to_clean_audio(self, clean_audio, noise_signal):
        if len(clean_audio) >= len(noise_signal):
            while len(clean_audio) >= len(noise_signal):
                noise_signal = np.append(noise_signal, noise_signal)

        ind = np.random.randint(0, noise_signal.size - clean_audio.size)

        noise_part = noise_signal[ind : ind + clean_audio.size]

        speech_power = np.sum(clean_audio**2)
        noise_power = np.sum(noise_part**2)
        noisy_audio = clean_audio + np.sqrt(speech_power / noise_power) * noise_part
        return noisy_audio

    def process_audio_in_parallel(self, clean_file_path):
        try:
            clean_audio, _ = read_audio(clean_file_path, self.sample_rate)
        except:
            return

        clean_audio = self._remove_silent_frames(clean_audio)

        noisy_file_path = self._random_noisy_file_path()

        noisy_audio, sr = read_audio(noisy_file_path, self.sample_rate)

        noisy_audio = self._remove_silent_frames(noisy_audio)

        clean_audio = self._randomly_crop_audio(
            clean_audio, duration=self.max_audio_length
        )

        noisy_input = self._add_noise_to_clean_audio(clean_audio, noisy_audio)

        noisy_input_features = FeatureExtractor(
            noisy_input,
            window_size=self.window_size,
            overlap=self.overlap,
            sample_rate=self.sample_rate,
        )
        noisy_spectrogram = noisy_input_features.get_stft_spectrogram()

        noisy_phase = np.angle(noisy_spectrogram)

        noisy_magnitude = np.abs(noisy_spectrogram)

        clean_input_features = FeatureExtractor(
            clean_audio,
            window_size=self.window_size,
            overlap=self.overlap,
            sample_rate=self.sample_rate,
        )
        clean_spectrogram = clean_input_features.get_stft_spectrogram()

        clean_phase = np.angle(clean_spectrogram)

        clean_magnitude = np.abs(clean_spectrogram)

        clean_magnitude = self._scale_phase_difference(
            clean_magnitude, clean_phase, noisy_phase
        )

        scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
        noisy_magnitude = scaler.fit_transform(noisy_magnitude)
        clean_magnitude = scaler.transform(clean_magnitude)

        return noisy_magnitude, clean_magnitude, noisy_phase

    def create_tf_records(self, *, prefix, subset_size, parallel=True):
        count = 0
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        for i in range(0, len(self.clean_file_paths), subset_size):
            tfrecord_filename = prefix + "_" + str(count) + ".tfrecords"

            if os.path.isfile(tfrecord_filename):
                print(f"Skipping {tfrecord_filename}")
                count += 1
                continue

            writer = tf.io.TFRecordWriter(tfrecord_filename)
            clean_file_paths_subset = self.clean_file_paths[i : i + subset_size]

            print(f"Processing files: {i} to {i + subset_size}")

            if parallel:
                outputs = pool.map(
                    self.process_audio_in_parallel, clean_file_paths_subset
                )
            else:
                outputs = [
                    self.process_audio_in_parallel(file_path)
                    for file_path in clean_file_paths_subset
                ]

            for output in outputs:
                if output is None:
                    continue

                noisy_stft_magnitude = output[0]
                clean_stft_magnitude = output[1]
                noisy_stft_phase = output[2]

                noisy_stft_magnitude_features = prepare_input_features(
                    noisy_stft_magnitude, num_segments=8, num_features=129
                )

                noisy_stft_magnitude_features = np.transpose(
                    noisy_stft_magnitude_features, (2, 0, 1)
                )
                clean_stft_magnitude = np.transpose(clean_stft_magnitude, (1, 0))
                noisy_stft_phase = np.transpose(noisy_stft_phase, (1, 0))

                noisy_stft_magnitude_features = np.expand_dims(
                    noisy_stft_magnitude_features, axis=3
                )
                clean_stft_magnitude = np.expand_dims(clean_stft_magnitude, axis=2)

                for x_, y_, p_ in zip(
                    noisy_stft_magnitude_features,
                    clean_stft_magnitude,
                    noisy_stft_phase,
                ):
                    y_ = np.expand_dims(y_, 2)
                    example = get_tf_feature(x_, y_, p_)
                    writer.write(example.SerializeToString())

            count += 1
            writer.close()
