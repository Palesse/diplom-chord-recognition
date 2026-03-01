import numpy as np
import os
import matplotlib.pyplot as plt
import json
from chromagram import compute_chroma  # оставляем для возможности использования
import librosa
import argparse


def get_templates(chords):
    """read from JSON file to get chord templates"""
    with open("data/chord_templates.json", "r") as fp:
        templates_json = json.load(fp)
    templates = []

    for chord in chords:
        if chord in templates_json:
            templates.append(templates_json[chord])
        else:
            continue

    return templates


def get_chords_list():
    """Return list of chords including 'N' for no chord"""
    chords = [
        "N",
        "C:maj",
        "C#:maj",
        "D:maj",
        "D#:maj",
        "E:maj",
        "F:maj",
        "F#:maj",
        "G:maj",
        "G#:maj",
        "A:maj",
        "A#:maj",
        "B:maj",
        "C:min",
        "C#:min",
        "D:min",
        "D#:min",
        "E:min",
        "F:min",
        "F#:min",
        "G:min",
        "G#:min",
        "A:min",
        "A#:min",
        "B:min",
    ]
    return chords


def find_chords(
        x: np.ndarray,
        fs: int,
        templates: list,
        chords: list,
        plot: bool = False,
        use_custom_chroma: bool = False  # параметр для выбора метода вычисления хромаграммы
):
    """
    Given a mono audio signal x, and its sampling frequency, fs,
    find chords using template matching

    Args:
        x : mono audio signal
        fs : sampling frequency (Hz)
        templates: list of chord templates
        chords: list of chords to search over
        plot: if results should be plotted
        use_custom_chroma: if True, use compute_chroma from chromagram.py, otherwise use librosa
    """

    # framing audio, window length = 8192, hop size = 1024
    nfft = int(8192 * 0.5)
    hop_size = int(1024 * 0.5)
    nFrames = int(np.round(len(x) / (nfft - hop_size)))

    # zero padding to make signal length long enough to have nFrames
    x = np.append(x, np.zeros(nfft))
    xFrame = np.empty((nfft, nFrames))
    start = 0
    num_chords = len(templates)
    id_chord = np.zeros(nFrames, dtype="int32")
    timestamp = np.zeros(nFrames)
    max_cor = np.zeros(nFrames)

    # compute timestamps and optionally framewise chroma using custom method
    chroma = None

    if use_custom_chroma:
        # chromagram.py
        chroma = np.zeros((12, nFrames))
        for n in range(nFrames):
            xFrame[:, n] = x[start: start + nfft]
            start = start + nfft - hop_size
            timestamp[n] = n * (nfft - hop_size) / fs
            chroma[:, n] = compute_chroma(xFrame[:, n], fs)
    else:
        # Librosa
        for n in range(nFrames):
            xFrame[:, n] = x[start: start + nfft]
            start = start + nfft - hop_size
            timestamp[n] = n * (nfft - hop_size) / fs

        chroma = librosa.feature.chroma_stft(y=x, sr=fs, n_fft=nfft, hop_length=hop_size)

    # visualize the chroma if requested
    if plot:
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(chroma, sr=fs, x_axis="frames", y_axis="chroma")
        plt.title("Chroma Features")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # correlate 12D chroma vector with each chord template
    for n in range(nFrames):
        cor_vec = np.zeros(num_chords)
        for ni in range(num_chords):
            cor_vec[ni] = np.correlate(chroma[:, n], np.array(templates[ni]))[0]
        max_cor[n] = np.max(cor_vec)
        id_chord[n] = np.argmax(cor_vec) + 1

    # apply threshold to identify no-chord zones
    threshold = 0.3
    id_chord[np.where(max_cor < threshold * np.max(max_cor))] = 0
    final_chords = [chords[cid] for cid in id_chord]

    if plot:
        plt.figure()
        plt.yticks(np.arange(num_chords + 1), chords)
        plt.plot(timestamp, id_chord, marker="o")
        plt.xlabel("Time in seconds")
        plt.ylabel("Chords")
        plt.title("Identified chords")
        plt.grid(True)
        plt.show()

    return timestamp, final_chords


def get_args():
    parser = argparse.ArgumentParser(description="Chord recognition using template matching")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="input audio file")
    parser.add_argument("-p", "--plot", action="store_true", help="show plots")
    parser.add_argument("-c", "--custom_chroma", action="store_true",
                        help="use custom chroma from chromagram.py (default: use librosa)")
    args = parser.parse_args()
    return args


def main(args):
    print("Input file is:", args.input_file)
    print("Using custom chroma:" if args.custom_chroma else "Using librosa chroma")

    directory = os.getcwd() + "/data/test_chords/"

    # load audio file
    x, fs = librosa.load(directory + args.input_file)

    # Suppress percussive elements
    x = librosa.effects.harmonic(x, margin=4)

    # get chords list
    chords = get_chords_list()

    # get chord templates
    templates = get_templates(chords)

    # find the chords using template matching
    timestamp, final_chords = find_chords(
        x,
        fs,
        templates=templates,
        chords=chords,
        plot=args.plot,
        use_custom_chroma=args.custom_chroma
    )

    # print chords with timestamps
    print("\nTime (s) - Time (s)  Chord")
    print("-" * 40)
    start_time = timestamp[0]
    for n in range(len(timestamp) - 1):
        if final_chords[n] == final_chords[n + 1]:
            continue
        else:
            print(f"{start_time:8.3f} - {timestamp[n + 1]:8.3f}  {final_chords[n]}")
            start_time = timestamp[n + 1]

    # print last chord if needed
    if start_time != timestamp[-1]:
        print(f"{start_time:8.3f} - {timestamp[-1]:8.3f}  {final_chords[-1]}")


if __name__ == "__main__":
    args = get_args()
    main(args)