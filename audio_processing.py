import whisper
import json
import subprocess
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile
import os
import torch
import torchaudio
from pathlib import Path
# from pyannote.audio import Pipeline

############################################################
# 1) Диаризация Pyannote
############################################################
# def diarize_with_pyannote(audio_path, hf_token=""):
#     """
#     Запускает pyannote.audio для диаризации (определения спикеров).
#     Возвращает список сегментов вида: [
#       {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
#       {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
#       ...
#     ]
#     """
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization",
#         use_auth_token=hf_token  # Если вы не логинились глобально
#     )
#     diarization = pipeline(audio_path)
#
#     segments = []
#     # itertracks() -> (segment, track_id, speaker_label)
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         segments.append({
#             "start": turn.start,
#             "end": turn.end,
#             "speaker": speaker
#         })
#     return segments

############################################################
# 2) Делим аудио на чанки по результатам диаризации
############################################################
# def split_audio_by_diarization(audio_path, diar_segments, output_dir="speaker_chunks"):
#     """
#     Принимает аудиофайл и список сегментов {start,end,speaker}.
#     Нарезает ffmpeg'ом каждый сегмент в отдельный .wav файл.
#     Создаёт структуру:
#         speaker_chunks/
#          ├─ SPEAKER_00/
#          │    ├─ chunk_0000.wav
#          │    ├─ chunk_0005.wav
#          ├─ SPEAKER_01/
#               ├─ chunk_0001.wav
#               ...
#     Возвращает список словарей:
#       [
#         {
#           "speaker": str,
#           "chunk_path": str (путь к созданному wav),
#           "start": float,
#           "end": float
#         },
#         ...
#       ]
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     results = []
#
#     for i, seg in enumerate(diar_segments):
#         spk = seg["speaker"]
#         start_time = seg["start"]
#         end_time   = seg["end"]
#
#         spk_dir = Path(output_dir) / spk
#         spk_dir.mkdir(parents=True, exist_ok=True)
#
#         out_file = spk_dir / f"chunk_{i:04d}.wav"
#
#         command = [
#             "ffmpeg", "-y",
#             "-i", str(audio_path),
#             "-ss", str(start_time),
#             "-to", str(end_time),
#             "-c", "copy",
#             str(out_file)
#         ]
#         subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#
#         results.append({
#             "speaker": spk,
#             "chunk_path": str(out_file),
#             "start": start_time,
#             "end": end_time
#         })
#
#     return results

def get_mp3_duration(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    Определяет длительность MP3 файла с использованием FFmpeg.

    :param file_path: Путь к MP3 файлу.
    :return: Длительность в формате "hh:mm:ss".
    :raises RuntimeError: Если длительность не может быть определена.
    """
    process = subprocess.Popen(
        ["ffmpeg", "-i", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8"
    )

    # Ищем строку с "Duration"
    duration_line = None
    for line in process.stdout:
        if "Duration" in line:
            duration_line = line.strip()
            break

    process.stdout.close()
    process.wait()

    if not duration_line:
        raise RuntimeError(f"Не удалось определить длительность файла {file_path}. "
                           f"Убедитесь, что это корректный аудиофайл.")

    # Парсим строку с длительностью
    try:
        # Извлекаем строку с длительностью, формат: "Duration: 00:41:06.26"
        duration_str = duration_line.split("Duration:")[1].split(",")[0].strip()  # Получаем "00:41:06.26"
        time_parts = duration_str.split(":")
        if len(time_parts) != 3:
            raise ValueError(f"Неправильный формат времени: {duration_str}")

        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(float(time_parts[2]))  # Учитываем дробные секунды

        return f"{hours}:{minutes:02}:{seconds:02}"
    except Exception as e:
        raise RuntimeError(f"Ошибка обработки строки длительности: {duration_line}. Ошибка: {e}")



# Конвертация MP3 в WAV с помощью FFmpeg
def convert_to_wav(input_audio, output_dir, output_prefix="part_", num_parts=1):
    """
    Конвертирует MP3 в WAV и делит на равные части.

    :param input_audio: Путь к входному MP3 файлу.
    :param output_dir: Папка для сохранения частей.
    :param output_prefix: Префикс для выходных файлов.
    :param num_parts: Количество частей, на которые нужно разделить.
    :return: Список путей к созданным файлам.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Проверяем существование файла
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Файл не найден: {input_audio}")

    print(f"Обрабатывается файл: {input_audio}")

    # Получаем длительность
    duration_str = get_mp3_duration(input_audio)
    print(f"Длительность аудиофайла: {duration_str}")

    # Переводим длительность в секунды
    time_parts = duration_str.split(":")  # Ожидается формат "hh:mm:ss"
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = int(time_parts[2])
    total_seconds = hours * 3600 + minutes * 60 + seconds

    # Рассчитываем длительность каждой части
    part_duration = total_seconds / num_parts
    print(f"Длительность каждой части: {part_duration:.2f} секунд")

    # Список для хранения путей к частям
    parts = []

    # Разделяем файл на части
    for i in range(num_parts):
        start_time = i * part_duration
        output_file = Path(output_dir) / f"{output_prefix}{i + 1}.wav"
        # Проверяем, существует ли файл
        if output_file.exists():
            print(f"Часть {i + 1} уже существует: {output_file}, пропускаем.")
        else:
            # Выполняем ffmpeg только если файл отсутствует
            command = [
                "ffmpeg", "-i", input_audio,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-ss", str(start_time), "-t", str(part_duration),
                str(output_file)
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Часть {i + 1} сохранена: {output_file}")

        # Добавляем файл (существующий или новый) в список
        parts.append(str(output_file))

    print("Конвертация и разделение завершены.")
    return parts

# Удаление фонового шума и музыки с помощью Demucs
def clean_audio_with_demucs(input_audio, output_dir):

    # Определяем путь для файла vocals
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_file = output_dir / f"{Path(input_audio).stem}_vocals.wav"

    # Если файл уже существует, возвращаем его
    if vocals_file.exists():
        print(f"Файл уже существует: {vocals_file}")
        return str(vocals_file)


    device = torch.device("cpu")
    # Загрузка предобученной модели
    print("Загрузка модели Demucs...")
    model = get_model("htdemucs")   # Используйте 'htdemucs', 'mdx_extra_q', и т.д.
    model.to(device)  # Перемещаем модель на GPU/CPU
    print(f"Модель загружена и перемещена на устройство: {device}")

    # Чтение аудиофайла
    print(f"Чтение аудиофайла: {input_audio}")
    wav = AudioFile(input_audio).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    wav = wav.unsqueeze(0).to(device)  # Перемещаем тензор на устройство
    print(f"Аудиофайл успешно загружен. Размер тензора: {wav.shape}")

    torch.cuda.empty_cache()
    print("Кэш GPU очищен.")

    # Применение модели
    print("Применение модели для разделения аудио...")
    print(f"Занято памяти на GPU: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} ГБ")
    print(f"Свободно памяти на GPU: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} ГБ")
    with torch.amp.autocast(device_type='cuda'):
        sources = apply_model(model, wav, device=device)    # Указываем device
    print("Разделение аудио завершено.")


    for i, source in enumerate(model.sources):
        #output_path = output_dir / f"{Path(input_audio).stem}_{source}.wav"

        # Сохраняем аудио с использованием torchaudio
        torchaudio.save(str(vocals_file), sources[0, i].cpu(), model.samplerate)
        print(f"Сохранено: {vocals_file}")

    if not vocals_file.exists():
        raise FileNotFoundError("Файл с вокалом (vocals) не был создан.")

    # Возвращаем только файл с вокалом
    return vocals_file

############################################################
# 3) Для каждого чанка (уже с указанием, какой это спикер),
#    запускаем Whisper, формируем JSON c результатом.
############################################################
# def transcribe_speaker_chunks(chunks_info, model_name="medium", language="Russian", out_json="transcription.json"):
#     """
#     Принимает список чанков вида:
#       [{"speaker": ..., "chunk_path": ..., "start":..., "end":...}, ...]
#
#     Для каждого чанка вызывает Whisper (модель = model_name),
#     собирает результат в список:
#       [
#         {
#           "speaker": ...,
#           "file_path": ...,
#           "start": ...,
#           "end": ...,
#           "transcription": "текст",
#         },
#         ...
#       ]
#
#     Сохраняет в JSON-файл (out_json).
#
#     Возвращает этот список словарей.
#     """
#     model = whisper.load_model(model_name)
#     final_data = []
#
#     for item in chunks_info:
#         spk = item["speaker"]
#         file_path = item["chunk_path"]
#         start_time = item["start"]
#         end_time   = item["end"]
#
#         print(f"Transcribing speaker={spk}, file={file_path}")
#         # Запуск Whisper
#         result = model.transcribe(file_path, language=language)
#         # Собираем полный текст (или можно по сегментам)
#         # Если хотим просто конкатенировать все сегменты
#         text_all = " ".join(seg["text"] for seg in result["segments"])
#
#         final_data.append({
#             "speaker": spk,
#             "file_path": file_path,
#             "start": start_time,
#             "end": end_time,
#             "transcription": text_all
#         })
#
#     # Сохраняем в JSON
#     with open(out_json, "w", encoding="utf-8") as f:
#         json.dump(final_data, f, ensure_ascii=False, indent=4)
#
#     print(f"Transcription saved to {out_json}")
#     return final_data

# Транскрипция аудио с Whisper
def transcribe_audio(input_audio, output_dir, model, transcription_file="transcription.json"):
    """
    Выполняет транскрипцию аудиофайла и сохраняет результат в JSON.
    Если данные уже существуют, транскрипция не повторяется.
    Новые сегменты добавляются согласованно с существующими чанками.

    :param input_audio: Путь к входному файлу.
    :param output_dir: Путь к папке для сохранения.
    :param model: Предзагруженная модель Whisper.
    :param transcription_file: Имя JSON файла для сохранения.
    :return: Новые сегменты транскрипции.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаем папку для сохранения
    os.makedirs(output_dir, exist_ok=True)
    transcription_path = Path(output_dir) / transcription_file

    # Загрузка существующих данных
    if transcription_path.exists():
        with open(transcription_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = {"chunks": []}

    # Проверяем, есть ли уже транскрипция для текущего файла
    for chunk in existing_data["chunks"]:
        if chunk["audio_path"] == str(input_audio):
            print(f"Транскрипция для {input_audio} уже существует.")
            return chunk["segments"]  # Возвращаем существующие сегменты

    # Выполняем транскрипцию
    print(f"Начало транскрипции файла: {input_audio}")
    result = model.transcribe(str(input_audio), language="Russian")
    new_segments = result["segments"]

    # Генерируем согласованные ID для новых сегментов
    last_id = max(
        (segment["id"] for chunk in existing_data["chunks"] for segment in chunk["segments"]),
        default=0,
    )
    for segment in new_segments:
        last_id += 1
        segment["id"] = last_id

    # Сохраняем информацию о текущем файле и его сегментах
    chunk_data = {
        "audio_path": str(input_audio),
        "segments": new_segments,
    }
    existing_data["chunks"].append(chunk_data)

    # Сохраняем обновленные данные в JSON
    with open(transcription_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"Транскрипция завершена и сохранена в {transcription_path}")
    return new_segments


def slice_audio(input_audio, transcription_result, output_dir="chunks", start_index=0):
    """
    Нарезает аудио на чанки на основе временных меток с проверкой существующих файлов.

    :param input_audio: Путь к аудиофайлу.
    :param transcription_result: Результат транскрипции.
    :param output_dir: Путь для сохранения чанков.
    :param start_index: Начальный индекс для имён файлов.
    :return: Обновлённый transcription_result с путями к чанкам.
    """
    os.makedirs(output_dir, exist_ok=True)
    segments = transcription_result

    for i, segment in enumerate(segments, start=start_index):
        start_time = segment["start"]
        end_time = segment["end"]
        output_file = Path(output_dir) / f"chunk_{i}.wav"

        # Проверяем, существует ли файл
        if output_file.exists():
            print(f"Чанк уже существует: {output_file}, добавляем путь.")
        else:
            # Если файл не существует, создаём его
            command = [
                "ffmpeg", "-i", input_audio,
                "-ss", str(start_time), "-to", str(end_time),
                "-c", "copy", str(output_file)
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Чанк создан: {output_file}")

        # Добавляем путь к чанку в транскрипцию
        segment["chunk_path"] = str(output_file)

    print("Нарезка аудио завершена.")
    return transcription_result


def create_dataset(chunks_data, output_file):
    """
    Создаёт CSV-датасет из списка данных чанков.

    :param chunks_data: Список словарей с полями "chunk" и "text".
    :param output_file: Путь к выходному CSV файлу.
    """
    dataset = [f"{data['chunk']}|{data['text']}" for data in chunks_data]
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(dataset))
    print(f"Датасет сохранён в {output_file}")



# Полный процесс
def process_audio(input_audio, output_dir="output", model_name="medium"):
    """
    Полный процесс обработки аудио:
    - Конвертация в WAV и разделение на части.
    - Очистка каждой части с помощью Demucs.
    - Транскрипция каждой части с Whisper.
    - Нарезка аудио на чанки.
    - Создание датасета.

    :param input_audio: Путь к входному MP3 файлу.
    :param output_dir: Директория для сохранения результатов.
    :param model_name: Whisper модель (по умолчанию 'medium').
    """
    os.makedirs(output_dir, exist_ok=True)

    # Конвертация в WAV и разделение на части
    print("Конвертация и разделение аудио...")
    parts = convert_to_wav(input_audio, output_dir / "parts")
    model = whisper.load_model(model_name)
    all_chunks_data = []  # Для хранения данных о чанках и текстах

    transcription_file = "transcription.json"

    for part_index, part in enumerate(parts):
        # Очистка аудио
        cleaned_audio = clean_audio_with_demucs(part, output_dir / "cleaned")

        # pynote_segments = diarize_with_pyannote(cleaned_audio)
        #
        # dictionary_of_pynote_sliced_audio = split_audio_by_diarization(cleaned_audio, pynote_segments, output_dir /"dataset")

        # Транскрипция
        transcription_result = transcribe_audio(
            cleaned_audio,
            output_dir,
            model,
            transcription_file=transcription_file
        )

        # Считаем текущий индекс для чанков
        start_index = len(all_chunks_data)

        # Нарезка на чанки
        chunks = slice_audio(
            cleaned_audio,
            transcription_result,
            output_dir / "chunks",
            start_index=start_index
        )
        # all_chunks_data.extend(chunks)
        # Добавляем чанки и текст в общий список
        for chunk_path, segment in zip(chunks, transcription_result):
            all_chunks_data.append({"chunk": chunk_path, "text": segment["text"]})

    # Создание датасета
    print("Создание датасета...")
    create_dataset(all_chunks_data, output_file=output_dir / "dataset.csv")

    print("Обработка завершена.")

# Устройство для работы с моделями
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    audio_dir = "./data/audio"  # Директория с файлами MP3
    output_dir = "./processed_data/"  # Директория для обработки

    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)

    for audio in audio_dir.glob("*.mp3"):
        print(f"Обработка файла: {audio}")
        process_audio(str(audio), output_dir=output_dir / audio.stem)

    # transcribe_audio("../processed_data/01/cleaned/part_1_vocals.wav", output_dir / "transcriptions", "medium")