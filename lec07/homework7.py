import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord.
    '''
    duration = 0.5  # seconds
    N = int(Fs * duration)
    n = np.arange(N)

    # Frequencies of the major chord
    f_root = f
    f_major_third = f * (2 ** (4/12))
    f_major_fifth = f * (2 ** (7/12))

    # Convert to radial frequencies
    omega_root = 2 * np.pi * f_root / Fs
    omega_third = 2 * np.pi * f_major_third / Fs
    omega_fifth = 2 * np.pi * f_major_fifth / Fs

    # Generate tones and sum
    x = (
        np.cos(omega_root * n) +
        np.cos(omega_third * n) +
        np.cos(omega_fifth * n)
    )

    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix.
    '''
    k = np.arange(N).reshape((N, 1))
    n = np.arange(N).reshape((1, N))

    W = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    N = len(x)

    # Compute DFT
    X = np.fft.fft(x)
    magnitudes = np.abs(X)

    # Only look at positive frequencies
    half = N // 2
    magnitudes = magnitudes[:half]

    # Find indices of three largest peaks
    indices = np.argsort(magnitudes)[-3:]

    # Convert indices to frequencies
    freqs = indices * Fs / N

    # Sort frequencies
    freqs = np.sort(freqs)

    return freqs[0], freqs[1], freqs[2]
