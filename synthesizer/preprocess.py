from multiprocessing.pool import Pool
from synthesizer import audio
from functools import partial
from itertools import chain
from encoder import inference as encoder
from pathlib import Path
from utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa


def preprocess_dataset(datasets_root: Path, out_dir: Path, n_processes: int, skip_existing: bool, hparams,
                       no_alignments: bool, datasets_name: str, subfolders: str):
    # Gather the input directories
    dataset_root = datasets_root.joinpath(datasets_name)
    input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in subfolders.split(",")]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)

    # Create the output directories for each output file type
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)

    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

    # Preprocess the dataset
    speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
    func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing,
                   hparams=hparams, no_alignments=no_alignments)
    job = Pool(n_processes).imap(func, speaker_dirs)
    for speaker_metadata in tqdm(job, datasets_name, len(speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

    # Verify the contents of the metadata file
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))


def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams, no_alignments: bool):
    metadata = []
    for book_dir in speaker_dir.glob("*"):
        if no_alignments:
            # Gather the utterance audios and texts
            # LibriTTS uses .wav but we will include extensions for compatibility with other datasets
            extensions = ["*.wav", "*.flac", "*.mp3"]
            for extension in extensions:
                wav_fpaths = book_dir.glob(extension)

                for wav_fpath in wav_fpaths:
                    # Load the audio waveform
                    wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
                    if hparams.rescale:
                        wav = wav / np.abs(wav).max() * hparams.rescaling_max

                    # Get the corresponding text
                    # Check for .txt (for compatibility with other datasets)
                    text_fpath = wav_fpath.with_suffix(".txt")
                    if not text_fpath.exists():
                        # Check for .normalized.txt (LibriTTS)
                        text_fpath = wav_fpath.with_suffix(".normalized.txt")
                        assert text_fpath.exists()
                    with text_fpath.open("r") as text_file:
                        text = "".join([line for line in text_file])
                        text = text.replace("\"", "")
                        text = text.strip()

                    # Process the utterance
                    metadata.append(process_utterance(wav, text, out_dir, str(wav_fpath.with_suffix("").name),
                                                      skip_existing, hparams))
        else:
            # Process alignment file (LibriSpeech support)
            # Gather the utterance audios and texts
            try:
                alignments_fpath = next(book_dir.glob("*.alignment.txt"))
                with alignments_fpath.open("r") as alignments_file:
                    alignments = [line.rstrip().split(" ") for line in alignments_file]
            except StopIteration:
                # A few alignment files will be missing
                continue

            # Iterate over each entry in the alignments file
            for wav_fname, words, end_times in alignments:
                wav_fpath = book_dir.joinpath(wav_fname + ".flac")
                assert wav_fpath.exists()
                words = words.replace("\"", "").split(",")
                end_times = list(map(float, end_times.replace("\"", "").split(",")))

                # Process each sub-utterance
                wavs, texts = split_on_silences(wav_fpath, words, end_times, hparams)
                for i, (wav, text) in enumerate(zip(wavs, texts)):
                    sub_basename = "%s_%02d" % (wav_fname, i)
                    metadata.append(process_utterance(wav, text, out_dir, sub_basename,
                                                      skip_existing, hparams))

    return [m for m in metadata if m is not None]


def split_on_silences(wav_fpath, words, end_times, hparams):
    # Load the audio waveform
    wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == "" and words[-1] == ""

    # Find pauses that are too long
    mask = (words == "") & (end_times - start_times >= hparams.silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]

    # Profile the noise from the silences and perform noise reduction on the waveform
    silence_times = [[start_times[i], end_times[i]] for i in breaks]
    silence_times = (np.array(silence_times) * hparams.sample_rate).astype(np.int)
    noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
    if len(noisy_wav) > hparams.sample_rate * 0.02:
        profile = logmmse.profile_noise(noisy_wav, hparams.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    # Re-attach segments that are too short
    segments = list(zip(breaks[:-1], breaks[1:]))
    segment_durations = [start_times[end] - end_times[start] for start, end in segments]
    i = 0
    while i < len(segments) and len(segments) > 1:
        if segment_durations[i] < hparams.utterance_min_duration:
            # See if the segment can be re-attached with the right or the left segment
            left_duration = float("inf") if i == 0 else segment_durations[i - 1]
            right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
            joined_duration = segment_durations[i] + min(left_duration, right_duration)

            # Do not re-attach if it causes the joined utterance to be too long
            if joined_duration > hparams.hop_size * hparams.max_mel_frames / hparams.sample_rate:
                i += 1
                continue

            # Re-attach the segment with the neighbour of shortest duration
            j = i - 1 if left_duration <= right_duration else i
            segments[j] = (segments[j][0], segments[j + 1][1])
            segment_durations[j] = joined_duration
            del segments[j + 1], segment_durations[j + 1]
        else:
            i += 1

    # Split the utterance
    segment_times = [[end_times[start], start_times[end]] for start, end in segments]
    segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
    wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
    texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments]

    # # DEBUG: play the audio segments (run with -n=1)
    # import sounddevice as sd
    # if len(wavs) > 1:
    #     print("This sentence was split in %d segments:" % len(wavs))
    # else:
    #     print("There are no silences long enough for this sentence to be split:")
    # for wav, text in zip(wavs, texts):
    #     # Pad the waveform with 1 second of silence because sounddevice tends to cut them early
    #     # when playing them. You shouldn't need to do that in your parsers.
    #     wav = np.concatenate((wav, [0] * 16000))
    #     print("\t%s" % text)
    #     sd.play(wav, 16000, blocking=True)
    # print("")

    return wavs, texts


def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str,
                      skip_existing: bool, hparams):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.


    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    if skip_existing and mel_fpath.exists() and wav_fpath.exists():
        return None

    # Trim silence
    if hparams.trim_silence:
        wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)

    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None

    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text


def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)


def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int):
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)

    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))

def preprocess_custom_dataset(
        datasets_root: Path,
        out_dir: Path,
        n_processes: int = 4,
        skip_existing: bool = False,
        hparams=None,
        no_alignments: bool = False,
        datasets_name: str = "CustomDataset",
        # Любые дополнительные аргументы, специфичные для вашей архитектуры, можно добавить здесь
        **kwargs
):
    import csv
    """
    Предобрабатывает набор данных с кастомной структурой директорий, аналогично preprocess_dataset,
    но адаптировано под вашу архитектуру.
    """
    print(f"Начинается предобработка датасета: {datasets_name}")

    # 1. Путь к корневой директории набора данных
    dataset_root = datasets_root  # в вашем случае директория уже указывает на dataset_root

    # 2. Получаем список всех томов (поддиректорий) в корне набора данных
    volume_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]

    # 3. Создаем необходимые выходные директории
    out_dir.joinpath("mels").mkdir(exist_ok=True, parents=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True, parents=True)

    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_mode = "a" if skip_existing else "w"
    metadata_file = metadata_fpath.open(metadata_mode, encoding="utf-8")

    # 5. Собираем задачи на обработку аудиофайлов
    tasks = []
    for volume_dir in volume_dirs:
        csv_path = volume_dir.joinpath("dataset.csv")
        chunks_dir = volume_dir.joinpath("chunks")
        if csv_path.exists() and chunks_dir.exists():
            with csv_path.open(newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter='|')
                for row in reader:
                    # Предполагаем, что CSV: путь_к_файлу|транскрипция
                    # Удалим лишние пробелы
                    row = [item.strip() for item in row]
                    if len(row) < 2:
                        continue  # Пропуск некорректных строк
                    wav_rel_path, transcript = row[0], row[1]
                    # Вместо использования chunks_dir
                    # base_dir_for_wav = csv_path.parent  # директория, где находится CSV
                    wav_path = wav_rel_path.lstrip("/")
                    wav_path = Path(wav_path)
                    if wav_path.exists():
                        tasks.append((wav_path, transcript))
                    else:
                        print(f"Файл {wav_path} не найден!")


    # Настраиваем частичное применение глобальной функции process_wrapper
    wrapped_process = partial(process_wrapper,
                              out_dir=out_dir,
                              hparams=hparams,
                              skip_existing=skip_existing,
                              no_alignments=no_alignments)

    with Pool(n_processes) as pool:
        for metadatum in tqdm(pool.imap(wrapped_process, tasks), total=len(tasks), desc="Processing files"):
            if metadatum is not None:
                metadata_file.write("|".join(str(x) for x in metadatum) + "\n")

    metadata_file.close()

    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.strip().split("|") for line in metadata_file if line.strip()]

    if not metadata:
        print("Метаданные отсутствуют. Проверьте правильность генерации файла train.txt.")
    else:
        mel_frames = sum(int(m[4]) for m in metadata if len(m) > 4)
        timesteps = sum(int(m[3]) for m in metadata if len(m) > 3)
        sample_rate = hparams.sample_rate
        hours = (timesteps / sample_rate) / 3600

        print(f"The dataset consists of {len(metadata)} utterances, {mel_frames} mel frames, {timesteps} audio timesteps ({hours:.2f} hours).")

        max_text_length = max((len(m[5]) for m in metadata if len(m) > 5), default=0)
        max_mel_frames = max((int(m[4]) for m in metadata if len(m) > 4), default=0)
        max_audio_timesteps = max((int(m[3]) for m in metadata if len(m) > 3), default=0)

        print(f"Max input length (text chars): {max_text_length}")
        print(f"Max mel frames length: {max_mel_frames}")
        print(f"Max audio timesteps length: {max_audio_timesteps}")

import hashlib

def generate_unique_filename(file_name, extension=".npy"):
    # Используем хэш пути или содержимого файла для уникальности
    # Например, хэш пути:
    file_hash = hashlib.md5(file_name.encode()).hexdigest()
    return f"{file_name}_{file_hash}"


# Обертка для передачи нескольких аргументов в Pool.imap
# Вынесем process_wrapper на верхний уровень
def process_wrapper(args, out_dir, hparams, skip_existing, no_alignments):
    # unpack args and call process_file
    wav_path, transcript = args
    return process_file(wav_path, transcript, out_dir, hparams, skip_existing, no_alignments)

def load_audio(file_path, target_sr):
    # Загружаем файл и определяем его частоту дискретизации
    wav, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        # Только если частота отличается, пересэмплируем
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav

# 4. Функция для обработки одного аудиофайла
def process_file(wav_path, transcript, out_dir, hparams, skip_existing, no_alignments):
    # Загрузка аудио
    try:
        # Попытка загрузить аудио
        wav = load_audio(wav_path, hparams.sample_rate)
    except ValueError as e:
        # Если сигнал пустой или не может быть загружен, пропустить этот файл
        print(f"Пропуск файла {wav_path}: {e}")
        return None

    # Проверяем, что сигнал не пустой
    if len(wav) == 0:
        print(f"Пропуск файла {wav_path}: пустой аудиосигнал")
        return None

    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Формирование базового имени файла (basename) для сохранения результатов
    basename = generate_unique_filename(wav_path.stem)

    metadatum = process_utterance(wav, transcript, out_dir, basename,
                                  skip_existing, hparams)
    if metadatum is None:
        print(f"Утверждение для файла {wav_path} было пропущено.")
    else:
        return metadatum


