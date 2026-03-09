import numpy as np
import os
import json
import matplotlib.pyplot as plt
import librosa
from collections import Counter
from main import find_chords, get_chords_list, get_templates
import argparse


def sort_chords_musical(chords_list):
    """
    Сортирует список аккордов в музыкальном порядке:
    сначала все мажорные по кругу (C, C#, D, ..., B),
    затем все минорные в том же порядке
    """
    # Базовый порядок нот
    note_order = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Разделяем мажорные и минорные
    major_chords = []
    minor_chords = []

    for chord in chords_list:
        if chord == 'N':
            continue  # пропускаем 'N', он нам не нужен в матрице
        note, quality = chord.split(':')
        if quality == 'maj':
            major_chords.append(chord)
        else:
            minor_chords.append(chord)

    # Сортируем каждую группу по порядку нот
    def chord_sort_key(chord):
        note, quality = chord.split(':')
        return note_order.index(note)

    major_chords.sort(key=chord_sort_key)
    minor_chords.sort(key=chord_sort_key)

    # Объединяем
    return major_chords + minor_chords


def get_true_chord_from_filename(filename):
    """
    Извлечь истинный аккорд из имени файла.
    Формат: [Нота][# если есть][m если минор]_[способ игры]_[способ игры].wav
    """
    # Убираем расширение .wav
    base = os.path.splitext(filename)[0]

    # Берем первую часть до подчеркивания
    first_part = base.split('_')[0]

    # Определяем, минор ли это
    if first_part.endswith('m'):
        # Убираем 'm' в конце
        note = first_part[:-1]
        quality = 'min'
    else:
        note = first_part
        quality = 'maj'

    # Формируем аккорд в правильном формате
    chord = f"{note}:{quality}"

    return chord


def predict_chord_for_file(filepath, templates, chords, fs=22050, threshold=0.1, use_custom_chroma=False):
    """
    Загрузить файл, прогнать через алгоритм, вернуть:
    - best_chord: аккорд с максимальной суммарной длительностью
    - chord_percentages: словарь {аккорд: процент_времени}
    - total_silence: общее время тишины (аккорд 'N') в файле
    """
    # Загрузка и предобработка
    x, sr = librosa.load(filepath, sr=fs)
    x = librosa.effects.harmonic(x, margin=4)

    # Получение предсказаний по фреймам и временных меток
    timestamp, pred_chords = find_chords(
        x, sr,
        templates=templates,
        chords=chords,
        plot=False,
        use_custom_chroma=use_custom_chroma,
        threshold=threshold
    )

    # Вычисляем длительность каждого фрейма
    frame_durations = []
    for i in range(len(timestamp)):
        if i < len(timestamp) - 1:
            duration = timestamp[i + 1] - timestamp[i]
        else:
            duration = len(x) / sr - timestamp[i]
        frame_durations.append(duration)

    # Суммируем длительности для каждого аккорда (включая 'N')
    chord_durations = {}
    for chord, duration in zip(pred_chords, frame_durations):
        chord_durations[chord] = chord_durations.get(chord, 0) + duration

    # Общее время тишины ('N')
    total_silence = chord_durations.get('N', 0.0)

    # Убираем 'N' из словаря для дальнейшей обработки
    if 'N' in chord_durations:
        del chord_durations['N']

    if not chord_durations:
        return 'N', {'N': 100.0}, total_silence

    # Вычисляем общее время распознанных аккордов (без 'N')
    total_duration = sum(chord_durations.values())

    # Вычисляем проценты (только для распознанных аккордов)
    chord_percentages = {}
    for chord, duration in chord_durations.items():
        chord_percentages[chord] = (duration / total_duration) * 100

    # Находим аккорд с максимальной суммарной длительностью
    sorted_chords = sorted(chord_durations.items(), key=lambda x: x[1], reverse=True)
    best_chord = sorted_chords[0][0]

    return best_chord, chord_percentages, total_silence


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate chord recognition with different algorithms")
    parser.add_argument("-c", "--custom_chroma", action="store_true",
                        help="use custom chroma from chromagram.py (EPCP) (default: use librosa PCP)")
    parser.add_argument("-t", "--thresholds", type=float, nargs="+", default=[0.5],
                        help="thresholds to test (default: 0.5)")
    return parser.parse_args()


def main():
    args = get_args()

    # Загружаем список аккордов и шаблоны
    chords = get_chords_list()
    templates = get_templates(chords)

    # ТЕСТОВАЯ ПАПКА - можно менять здесь
    test_dir = "data/test_chords/base"
    # Извлекаем название конечной папки для использования в именах файлов
    folder_name = os.path.basename(test_dir)

    # Пороги для тестирования
    thresholds_to_try = args.thresholds

    # Выбор алгоритма из аргументов командной строки
    use_custom_chroma = args.custom_chroma

    # Определяем название алгоритма для вывода
    algorithm_name = "chromagram (EPCP)" if use_custom_chroma else "librosa (PCP)"

    # Создаем папку для результатов, если её нет
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for thresh in thresholds_to_try:
        print(f"\n{'=' * 50}")
        print(f"Testing with threshold = {thresh}")
        print(f"Algorithm: {algorithm_name}")
        print(f"Test directory: {test_dir}")
        print('=' * 50)

        y_true = []
        y_pred = []
        filenames = []
        true_chord_list = []
        pred_chord_list = []
        percent_correct_list = []
        all_percentages = []  # для хранения процентов всех аккордов для каждого файла
        silence_times = []    # для хранения времени тишины для каждого файла

        # Перебираем все файлы в папке
        for filename in sorted(os.listdir(test_dir)):
            if not filename.endswith('.wav'):
                continue

            filepath = os.path.join(test_dir, filename)

            # Получаем истинный аккорд из имени файла
            true_chord = get_true_chord_from_filename(filename)
            if true_chord is None:
                print(f"Could not parse chord from {filename}, skipping")
                continue

            # Получаем предсказание, проценты и время тишины
            pred_chord, percentages, silence_time = predict_chord_for_file(
                filepath, templates, chords, threshold=thresh, use_custom_chroma=use_custom_chroma
            )

            # Вычисляем процент времени, когда показывался правильный аккорд (для статистики)
            correct_percent = percentages.get(true_chord, 0.0)

            # Сохраняем данные
            y_true.append(true_chord)
            y_pred.append(pred_chord)
            filenames.append(filename)
            true_chord_list.append(true_chord)
            pred_chord_list.append(pred_chord)
            percent_correct_list.append(correct_percent)
            all_percentages.append(percentages)
            silence_times.append(silence_time)

            # Отмечаем правильность (для лучшего аккорда)
            mark = "✓" if true_chord == pred_chord else "✗"
            print(f"{mark} {filename:30} -> true: {true_chord:6}, "
                  f"pred: {pred_chord:6}, correct: {correct_percent:.1f}%, "
                  f"silence: {silence_time:.3f}s")

        # Вычисляем общую точность (по лучшему аккорду)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0
        print(f"\nAccuracy (by best chord): {correct}/{total} = {accuracy:.2%}")

        # Средний процент времени правильного аккорда
        avg_correct_percent = sum(percent_correct_list) / len(percent_correct_list) if percent_correct_list else 0
        print(f"Average time with correct chord: {avg_correct_percent:.1f}%")

        # Суммарное время тишины по всем файлам
        total_silence_time = sum(silence_times)
        print(f"Total silence time across all files: {total_silence_time:.3f}s")

        # --- СОХРАНЕНИЕ В CSV (с процентом правильного аккорда) ---
        algo_suffix = "chromagram" if use_custom_chroma else "librosa"
        # Добавляем название папки в имя файла
        csv_filename = os.path.join(results_dir, f'predictions_{folder_name}_thresh_{thresh}_{algo_suffix}.csv')
        with open(csv_filename, 'w', encoding='utf-8') as f:
            f.write('filename,true,predicted,correct_by_best,percent_correct,silence_time,all_percentages\n')
            for i, (fname, t, p, corr_flag, perc, silence) in enumerate(zip(
                    filenames, y_true, y_pred,
                    ['yes' if t == p else 'no' for t, p in zip(y_true, y_pred)],
                    percent_correct_list, silence_times)):
                # Формируем строку со всеми процентами
                all_perc_str = '; '.join([f"{chord}:{perc:.1f}%"
                                          for chord, perc in all_percentages[i].items()])

                f.write(f'{fname},{t},{p},{corr_flag},{perc:.2f},{silence:.3f},{all_perc_str}\n')

            # --- ДОБАВЛЕНО: запись статистики в конец CSV ---
            f.write(f'\n# Точность (% времени): ,{avg_correct_percent:.2f}%\n')
            f.write(f'# Точность (кол-во правильно предсказанных): ,{accuracy:.2%}\n')
            f.write(f'# Суммарное время тишины (сек): ,{total_silence_time:.3f}\n')
            # ------------------------------------------------------

        print(f"Результат тестирования сохранен в: {csv_filename}")
        # ---------------------------------------------

        # Строим матрицу ошибок с процентами
        if total > 0:
            # Получаем уникальные аккорды
            unique_chords = sort_chords_musical(set(y_true + y_pred))
            chord_to_idx = {chord: i for i, chord in enumerate(unique_chords)}

            # Считаем количество файлов для каждого истинного аккорда
            true_chord_counts = {}
            for t in y_true:
                true_chord_counts[t] = true_chord_counts.get(t, 0) + 1

            # Считаем количество правильно угаданных файлов для каждого истинного аккорда
            correct_counts = {}
            for t, p in zip(y_true, y_pred):
                if t == p:
                    correct_counts[t] = correct_counts.get(t, 0) + 1

            # Создаем матрицу для хранения сумм процентов
            percent_sum_matrix = np.zeros((len(unique_chords), len(unique_chords)))

            # Для каждого файла добавляем процент в соответствующую клетку
            for i, (t, percentages) in enumerate(zip(y_true, all_percentages)):
                t_idx = chord_to_idx[t]
                # Для каждого предсказанного аккорда в этом файле
                for pred_chord, percent in percentages.items():
                    if pred_chord in chord_to_idx:  # проверяем, что аккорд есть в нашем списке
                        p_idx = chord_to_idx[pred_chord]
                        percent_sum_matrix[t_idx][p_idx] += percent

            # Делим на общее количество файлов для данного истинного аккорда
            percent_matrix = np.zeros_like(percent_sum_matrix)
            for i, true_chord in enumerate(unique_chords):
                n_files = true_chord_counts.get(true_chord, 0)
                if n_files > 0:
                    percent_matrix[i, :] = percent_sum_matrix[i, :] / n_files

            # Визуализация
            plt.figure(figsize=(max(12, len(unique_chords) * 0.6),
                                max(10, len(unique_chords) * 0.5)))

            # Используем percent_matrix для отображения
            im = plt.imshow(percent_matrix, interpolation='nearest',
                            cmap=plt.cm.YlOrRd, vmin=0, vmax=100)

            # Добавляем сетку
            plt.gca().set_xticks(np.arange(-0.5, len(unique_chords), 1), minor=True)
            plt.gca().set_yticks(np.arange(-0.5, len(unique_chords), 1), minor=True)
            plt.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.3)
            plt.tick_params(which='minor', bottom=False, left=False)

            # Заголовок с названием алгоритма и временем тишины
            plt.title(f'Матрица ошибок (порог = {thresh})\n'
                      f'Алгоритм: {algorithm_name}\n'
                      f'Точность (% времени): {avg_correct_percent:.1f}%\n'
                      f'Точность (кол-во): {accuracy:.2%}\n'
                      f'Тишина (суммарно): {total_silence_time:.2f}с')

            plt.colorbar(im, label='Средний % времени')

            # Добавляем количество правильно угаданных / общее количество к истинным аккордам
            y_labels = []
            for chord in unique_chords:
                total_count = true_chord_counts.get(chord, 0)
                correct_count = correct_counts.get(chord, 0)
                y_labels.append(f'{chord} ({correct_count}/{total_count})')

            # Настройка меток
            tick_marks = np.arange(len(unique_chords))
            plt.xticks(tick_marks, unique_chords, rotation=90, fontsize=8)
            plt.yticks(tick_marks, y_labels, fontsize=8)

            # Добавляем проценты в ячейки
            for i in range(len(unique_chords)):
                for j in range(len(unique_chords)):
                    if percent_matrix[i, j] > 0:
                        value = percent_matrix[i, j]
                        text = f'{value:.0f}'
                        text_color = 'white' if value > 50 else 'black'
                        plt.text(j, i, text,
                                 ha="center", va="center",
                                 color=text_color,
                                 fontsize=6, fontweight='bold')

            plt.ylabel('Истинный аккорд', fontsize=8)
            plt.xlabel('Предсказанный аккорд', fontsize=8)
            plt.tight_layout()

            # Сохраняем матрицу ошибок (с названием папки)
            png_filename = os.path.join(results_dir, f'confusion_matrix_{folder_name}_thresh_{thresh}_{algo_suffix}.png')
            plt.savefig(png_filename, dpi=150, bbox_inches='tight')
            print(f"Матрица ошибок сохранена в: {png_filename}")
            plt.show()


if __name__ == "__main__":
    main()