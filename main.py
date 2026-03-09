import numpy as np
import os
import matplotlib.pyplot as plt
import json
from chromagram import compute_chroma
import librosa
import argparse
import time  # Добавляем импорт для замера времени


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
        use_custom_chroma: bool = False,
        threshold: float = 0.1
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
        threshold: threshold for no-chord detection (fraction of max correlation)
    """

    # framing audio, window length = 8192, overlap = 1024
    nfft = int(8192 * 0.5)
    overlap = int(1024 * 0.5)
    nFrames = int(np.round(len(x) / (nfft - overlap)))

    # zero padding to make signal length long enough to have nFrames
    x = np.append(x, np.zeros(nfft))
    xFrame = np.empty((nfft, nFrames))
    start = 0
    num_chords = len(templates)
    id_chord = np.zeros(nFrames, dtype="int32")
    timestamp = np.zeros(nFrames)
    max_cor = np.zeros(nFrames)

    # Для сбора статистики времени
    frame_times = []
    total_start_time = time.time()

    # compute timestamps and optionally framewise chroma using custom method
    chroma = None

    if use_custom_chroma:
        # chromagram.py
        chroma = np.zeros((12, nFrames))
        for n in range(nFrames):
            frame_start = time.time()

            xFrame[:, n] = x[start: start + nfft]
            start = start + nfft - overlap
            timestamp[n] = n * (nfft - overlap) / fs
            chroma[:, n] = compute_chroma(xFrame[:, n], fs)

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            # Вывод времени для каждого 10-го фрейма или для первого/последнего
            if n == 0 or n == nFrames - 1 or n % 10 == 0:
                print(f"  Фрейм {n:3d}: {frame_time * 1000:6.2f} мс")

    else:
        # Librosa - здесь сложно замерить пофреймово, т.к. librosa делает всё сразу
        # Но мы можем замерить общее время и показать среднее
        print("Librosa обрабатывает все фреймы сразу...")
        librosa_start = time.time()

        for n in range(nFrames):
            xFrame[:, n] = x[start: start + nfft]
            start = start + nfft - overlap
            timestamp[n] = n * (nfft - overlap) / fs

        chroma = librosa.feature.chroma_stft(y=x, sr=fs, n_fft=nfft, hop_length=overlap)

        librosa_time = time.time() - librosa_start
        print(f"  Librosa общее время: {librosa_time * 1000:.2f} мс")
        print(f"  Среднее на фрейм: {librosa_time * 1000 / nFrames:.2f} мс")

    # correlate 12D chroma vector with each chord template
    print("\n--- Сопоставление с шаблонами ---")
    match_start = time.time()

    for n in range(nFrames):
        cor_vec = np.zeros(num_chords)
        for ni in range(num_chords):
            cor_vec[ni] = np.correlate(chroma[:, n], np.array(templates[ni]))[0]
        max_cor[n] = np.max(cor_vec)
        id_chord[n] = np.argmax(cor_vec) + 1

    match_time = time.time() - match_start
    print(f"Сопоставление с шаблонами: {match_time * 1000:.2f} мс")
    print(f"Среднее на фрейм: {match_time * 1000 / nFrames:.2f} мс")

    # apply threshold to identify no-chord zones
    id_chord[np.where(max_cor < threshold * np.max(max_cor))] = 0
    final_chords = [chords[cid] for cid in id_chord]

    # Общая статистика
    total_time = time.time() - total_start_time
    print(f"\n--- ОБЩАЯ СТАТИСТИКА ---")
    print(f"Всего фреймов: {nFrames}")
    print(f"Общее время обработки: {total_time * 1000:.2f} мс")

    if use_custom_chroma and frame_times:
        print(f"Среднее время на фрейм (chromagram): {np.mean(frame_times) * 1000:.2f} мс")
        print(f"Мин время на фрейм: {np.min(frame_times) * 1000:.2f} мс")
        print(f"Макс время на фрейм: {np.max(frame_times) * 1000:.2f} мс")

    print(f"Время на фрейм (всего с шаблонами): {total_time * 1000 / nFrames:.2f} мс")

    if plot:
        # Создаем одно окно с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # График 1: Chroma Features
        img = librosa.display.specshow(chroma, sr=fs, x_axis="frames", y_axis="chroma", ax=ax1)
        ax1.set_title("Хрома-вектор")
        ax1.set_xlabel("Фреймы")
        ax1.set_ylabel("PCP")
        plt.colorbar(img, ax=ax1, format="%+2.0f dB")

        # График 2: Identified chords
        ax2.set_yticks(np.arange(num_chords + 1))
        ax2.set_yticklabels(chords)
        ax2.plot(timestamp, id_chord, marker="o", linestyle='-', markersize=3)
        ax2.set_xlabel("Время (с)")
        ax2.set_ylabel("Аккорды")
        ax2.set_title(f"Предсказанные аккорды (порог = {threshold})")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return timestamp, final_chords


def get_args():
    parser = argparse.ArgumentParser(description="Chord recognition using template matching")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="input audio file")
    parser.add_argument("-p", "--plot", action="store_true", help="show plots")
    parser.add_argument("-c", "--custom_chroma", action="store_true",
                        help="use custom chroma from chromagram.py (default: use librosa)")
    parser.add_argument("-t", "--threshold", type=float, default=0.1,
                        help="threshold for no-chord detection (default: 0.1, range: 0.0-1.0)")
    args = parser.parse_args()

    if args.threshold < 0 or args.threshold > 1:
        parser.error("Threshold must be between 0.0 and 1.0")

    return args


def main(args):
    print("Input file is:", args.input_file)
    print("Using custom chroma:" if args.custom_chroma else "Using librosa chroma")
    print(f"Threshold: {args.threshold}")

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
        use_custom_chroma=args.custom_chroma,
        threshold=args.threshold  # передаём порог в функцию
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