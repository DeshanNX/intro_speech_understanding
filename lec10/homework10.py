import numpy as np
import torch
import torch.nn as nn
import librosa


def get_features(waveform, Fs):
    # ===== Pre-emphasis =====
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # ===== VAD (25 ms window, 10 ms step) =====
    vad_len = int(0.025 * Fs)
    vad_step = int(0.010 * Fs)

    vad_frames = librosa.util.frame(
        waveform, frame_length=vad_len, hop_length=vad_step
    )

    energy = np.sum(vad_frames ** 2, axis=0)
    threshold = 0.1 * np.max(energy)
    speech = energy > threshold

    labels = []
    current_label = 0

    for s in speech:
        if s:
            labels.extend([current_label] * 5)
            current_label += 1
        else:
            labels.extend([0] * 5)

    labels = np.array(labels)
    labels = np.clip(labels, 0, 5)

    # ===== Spectrogram (4 ms frame, 2 ms step) =====
    frame_len = int(0.004 * Fs)
    frame_step = int(0.002 * Fs)

    frames = librosa.util.frame(
        waveform, frame_length=frame_len, hop_length=frame_step
    )

    spectrum = np.abs(np.fft.rfft(frames, axis=0)).T

    # Keep only low-frequency half
    features = spectrum[:, :spectrum.shape[1] // 2]

    # ===== Align feature and label lengths =====
    N = min(features.shape[0], len(labels))
    features = features[:N]
    labels = labels[:N]

    return features, labels


def train_neuralnet(features, labels, iterations):
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    NFEATS = X.shape[1]
    NLABELS = int(max(labels)) + 1

    model = nn.Sequential(
        nn.LayerNorm(NFEATS),
        nn.Linear(NFEATS, NLABELS)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    lossvalues = np.zeros(iterations)

    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        lossvalues[i] = loss.item()

    return model, lossvalues


def test_neuralnet(model, features):
    X = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.detach().numpy()
