"""
Dual-channel audio preprocessing module

DCASE 2026 Task 2 audio is recorded in dual-channel:
    - Ch1 (channel 0): Close-range microphone, high SNR
    - Ch2 (channel 1): Far-range microphone, low SNR (more environmental noise)

This module implements Wiener denoising: estimates noise power spectrum from far-mic
signal and applies frequency-domain gain filtering on near-mic signal.
Experiments show α=0.5 is optimal (searched on dev set, Ω=0.6567).

Usage:
    from inference.preprocess import wiener_denoise, load_audio
    
    # Denoise a single audio
    enhanced = wiener_denoise(ch1, ch2, sr=16000, alpha=0.5)
    
    # Batch load (auto-detect dual-channel and denoise)
    wavs = load_audio(file_paths)
"""
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
from typing import List, Optional

# Default STFT parameters
DEFAULT_N_FFT = 1024
DEFAULT_HOP = 256
DEFAULT_ALPHA = 0.5


def wiener_denoise(ch1: np.ndarray, ch2: np.ndarray, sr: int = 16000,
                   alpha: float = DEFAULT_ALPHA,
                   n_fft: int = DEFAULT_N_FFT,
                   hop: int = DEFAULT_HOP) -> np.ndarray:
    """
    Wiener denoising: estimate noise from far-mic signal, enhance near-mic signal.
    
    Principle:
        gain(f,t) = sqrt(max(|Z1|² - α·|Z2|², 0) / (|Z1|² + ε))
        enhanced = iSTFT(Z1 · gain)
    
    where Z1 is near-mic STFT, Z2 is far-mic STFT, α controls denoising strength.
    
    Args:
        ch1: near-mic signal (T,)
        ch2: far-mic signal (T,), used as noise reference
        sr: sample rate (default 16000)
        alpha: noise reduction strength (0=no reduction, 0.5=optimal, 1.0=full reduction)
        n_fft: STFT window length
        hop: STFT hop length
        
    Returns:
        Enhanced mono signal (T,), same length as ch1
    """
    _, _, Z1 = stft(ch1, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Z2 = stft(ch2, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    
    signal_power = np.abs(Z1) ** 2
    noise_power = np.abs(Z2) ** 2
    
    # Wiener gain: retain signal power above noise estimate
    gain = np.sqrt(
        np.maximum(signal_power - alpha * noise_power, 0) / (signal_power + 1e-10)
    )
    
    enhanced = Z1 * gain
    _, wav_out = istft(enhanced, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    
    return wav_out[:len(ch1)].astype(np.float32)


def load_audio(paths: List[str], alpha: float = DEFAULT_ALPHA,
               n_fft: int = DEFAULT_N_FFT, hop: int = DEFAULT_HOP,
               verbose: bool = True) -> List[np.ndarray]:
    """
    Batch load audio files, auto-detect dual-channel and apply Wiener denoising.
    
    Processing strategy:
        - Dual-channel files → Wiener denoising (ch2 as noise ref, enhance ch1)
        - Mono files → return directly
    
    Args:
        paths: list of WAV file paths
        alpha: Wiener denoising parameter
        n_fft: STFT window length
        hop: STFT hop length
        verbose: whether to print statistics
        
    Returns:
        List of processed mono audio arrays
    """
    wavs = []
    n_stereo = 0
    
    for path in paths:
        wav, sr = sf.read(path, dtype='float32')
        
        if wav.ndim > 1 and wav.shape[1] >= 2:
            # Dual-channel: Wiener denoising
            n_stereo += 1
            ch1, ch2 = wav[:, 0], wav[:, 1]
            wav_out = wiener_denoise(ch1, ch2, sr=sr, alpha=alpha,
                                    n_fft=n_fft, hop=hop)
            wavs.append(wav_out)
        else:
            # Mono: use directly
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wavs.append(wav.astype(np.float32))
    
    if verbose and n_stereo > 0:
        print(f"  → Stereo files: {n_stereo}/{len(paths)} files, "
              f"Wiener denoising (α={alpha})")
    
    return wavs
