from tqdm import tqdm
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def extract_signal_features(signal, sr, n_mels=64, frames=5, n_fft=1024):
    # Compute a mel-scaled spectrogram:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels
    )

    # Convert to decibel (log scale for amplitude):
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram


# Load sound file
def load_sound_file(wav_name, channel=0):
    multi_channel_data, sampling_rate = librosa.load(wav_name, sr=None, mono=False)

    if len(multi_channel_data.shape) > 1:
        signal = multi_channel_data[channel, :]
    else:
        signal = multi_channel_data

    return signal, sampling_rate


# 加载音频文件
audio_path_fan = r'E:\VAE\Data\fan\train\normal_id_00_00000000.wav'  # 替换为你的音频文件路径
audio_path_pump =r'E:\VAE\Data\pump\train\normal_id_00_00000000.wav'
audio_path_slider =r'E:\VAE\Data\slider\train\normal_id_00_00000000.wav'
audio_path_ToyCar =r'E:\VAE\Data\ToyCar\train\normal_id_01_00000000.wav'
audio_path_ToyConveyor =r'E:\VAE\Data\ToyConveyor\train\normal_id_01_00000000.wav'
audio_path_value =r'E:\VAE\Data\valve\train\normal_id_00_00000000.wav'

signal, sr = librosa.load(audio_path_ToyCar, sr=None)

#振幅图
plt.figure(figsize=(10, 4))
plt.plot(signal)
plt.title('Amplitude')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.show()

#频谱图
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

plt.figure(figsize=(10, 4))
plt.plot(frequency[:len(frequency)//2], magnitude[:len(magnitude)//2])
plt.title('Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()

#频谱图
D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

#梅尔频谱图
log_mel_spectrogram = extract_signal_features(signal, sr)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    log_mel_spectrogram,
    sr=sr,
    x_axis='time',
    y_axis='mel',
    fmax=8000
)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (dB)')
plt.tight_layout()
plt.show()