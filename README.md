# diplom-chord-recognition

Chord-Recognition
Automatic chord recognition in Python
Chords are identified automatically from monophonic/polyphonic audio.

Как запустить:
python3 main.py -i "G_open_strum.wav"

С графиками:
python3 main.py -i "G_open_strum.wav" -p

С алгоритмом EPCP вместо стандартного librosa (пока что работает некорректно!):
python3 main.py -i "G_open_strum.wav" -p -c

С настройкой порога (дефолтный порог = 0.1):
python3 main.py -i "G_open_strum.wav" -p -t 0.3
