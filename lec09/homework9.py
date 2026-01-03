import numpy as np

# ---------- helpers ----------

def waveform_to_frames(waveform, frame_length, step):
    N = len(waveform)
    num_frames = int((N - frame_length) / step)
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start:start + frame_length]
    return frames

def frames_to_mstft(frames):
    return np.abs(np.fft.fft(frames, axis=1))

def mstft_to_spectrogram(mstft):
    floor = 0.001 * np.amax(mstft)
    return 20 * np.log10(np.maximum(mstft, floor))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- assignment functions ----------

def VAD(waveform, Fs):
    '''
    Extract waveform segments with energy > 10% of max.
    '''
    frame_length = int(0.025 * Fs)
    step = int(0.01 * Fs)

    frames = waveform_to_frames(waveform, frame_length, step)

    # Frame energy
    energy = np.sum(frames**2, axis=1)
    threshold = 0.1 * np.max(energy)

    voiced = energy > threshold

    segments = []
    start = None

    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            s = start * step
            e = i * step + frame_length
            segments.append(waveform[s:e])
            start = None

    if start is not None:
        s = start * step
        segments.append(waveform[s:])

    return segments


def segments_to_models(segments, Fs):
    '''
    Create average log-spectrum models for each segment.
    '''
    models = []

    for seg in segments:
        # Pre-emphasis
        seg = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])

        frame_length = int(0.004 * Fs)
        step = int(0.002 * Fs)

        frames = waveform_to_frames(seg, frame_length, step)
        mstft = frames_to_mstft(frames)
        spec = mstft_to_spectrogram(mstft)

        # Keep low-frequency half
        spec = spec[:, :spec.shape[1] // 2]

        # Average spectrum
        model = np.mean(spec, axis=0)
        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Recognize speech using cosine similarity.
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K))
    test_outputs = []

    for k, tmodel in enumerate(test_models):
        for y, model in enumerate(models):
            sims[y, k] = cosine_similarity(model, tmodel)

        best = np.argmax(sims[:, k])
        test_outputs.append(labels[best])

    return sims, test_outputs
