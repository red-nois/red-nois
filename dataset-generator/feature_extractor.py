import librosa
import numpy as np


class FeatureExtractor:
    def __init__(self, audio, *, window_size, overlap, sampling_rate):
        self.audio = audio
        self.fft_length = window_size
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.window = np.hamming(self.window_size)

    def get_stft_spectrogram(self):
        return librosa.stft(
            self.audio,
            n_fft=self.fft_length,
            win_length=self.window_size,
            hop_length=self.overlap,
            window=self.window,
            center=True,
        )

    def get_audio_from_stft_spectrogram(self, stft_features):
        return librosa.istft(
            stft_features,
            win_length=self.window_size,
            hop_length=self.overlap,
            window=self.window,
            center=True,
        )

    def get_mel_spectrogram(self):
        return librosa.feature.melspectrogram(
            self.audio,
            sr=self.sampling_rate,
            power=2.0,
            pad_mode="reflect",
            n_fft=self.fft_length,
            hop_length=self.overlap,
            center=True,
        )

    def get_audio_from_mel_spectrogram(self, M):
        return librosa.feature.inverse.mel_to_audio(
            M,
            sr=self.sampling_rate,
            n_fft=self.fft_length,
            hop_length=self.overlap,
            win_length=self.window_size,
            window=self.window,
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_iter=32,
            length=None,
        )
