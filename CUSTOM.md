---

# Руководство по Настройке Проекта

Добро пожаловать в ваш проект! Следуйте приведённым ниже шагам для настройки среды разработки, установки необходимых зависимостей и запуска обучающих скриптов.

## Содержание

1. [Предварительные Требования](#предварительные-требования)
2. [Настройка Anaconda Окружения](#настройка-anaconda-окружения)
3. [Установка Python 3.9](#установка-python-39)
4. [Установка Необходимых Пакетов](#установка-необходимых-пакетов)
5. [Установка Драйверов для Видеокарты](#установка-драйверов-для-видеокарты)
6. [Скачивание Данных](#скачивание-данных)
7. [Запуск Скриптов](#запуск-скриптов)
    - [Предобработка Данных](#предобработка-данных)
    - [Обучение Энкодера](#обучение-энкодера)
    - [Встраивание Mel-Спектрограммы](#встраивание-mel-спектрограммы)
    - [Обучение Синтезатора](#обучение-синтезатора)
    - [Предобработка для Вокодера](#предобработка-для-вокодера)
    - [Обучение Вокодера](#обучение-вокодера)
8. [Устранение Проблем](#устранение-проблем)

---

## Предварительные Требования

Перед началом убедитесь, что на вашем компьютере установлены следующие программы:

- **Anaconda**: [Скачать Anaconda](https://www.anaconda.com/products/distribution)
- **Git**: [Скачать Git](https://git-scm.com/downloads)
- **Драйверы NVIDIA GPU**: В зависимости от вашей видеокарты, следуйте приведённым ниже ссылкам.

---

## Настройка Anaconda Окружения

1. **Откройте терминал или командную строку.**

2. **Создайте новое окружение conda:**

    ```bash
    conda create -n my_project_env python=3.9
    ```

3. **Активируйте окружение:**

    ```bash
    conda activate my_project_env
    ```

---

## Установка Python 3.9

Если вы не указали Python 3.9 при создании окружения, установите его с помощью conda:

```bash
conda install python=3.9
```

---

## Установка Необходимых Пакетов

Убедитесь, что в вашей проектной директории есть файл `requirements.txt`. Установите необходимые пакеты с помощью pip:

```bash
pip install -r requirements.txt
```

> **Совет:** Для избежания проблем с зависимостями рекомендуется управлять пакетами внутри conda окружения.

---

## Установка Драйверов для Видеокарты

Для оптимальной производительности установите соответствующие драйверы для вашей GPU. Следуйте инструкциям в зависимости от производителя вашей видеокарты:

### NVIDIA

1. **Развёртывание NVIDIA AI Enterprise для VMware:**
    - [Руководство по установке](https://docs.nvidia.com/ai-enterprise/deployment/vmware/latest/docker.html)

2. **NVIDIA Container Toolkit:**
    - [Руководство по установке](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)

3. **Скачивание CUDA:**
    - [Скачать CUDA](https://developer.nvidia.com/cuda-downloads)

### AMD

*Обратитесь к официальной документации AMD для установки драйверов.*

---

## Скачивание Данных

Сначала скачайте необходимые данные из предоставленной папки Google Drive:

- [Скачать Данные](https://drive.google.com/drive/folders/16dIdrgRnKGTN2kWPJ30vTTi6syoXf388?usp=sharing)

---

## Запуск Скриптов

### 1. Предобработка Данных

#### Предобработка Энкодера

```bash
python ./encode_preprocessing.py -o ./dataset -d custom ./processed_data/
```

#### Предобработка Синтезатора

```bash
python ./synthesizer_preprocess_audio.py --duration=0.1 --no_alignments --datasets_name "01" --custom
```

### 2. Обучение Энкодера

```bash
python ./encoder_train.py 01 ./dataset/ --no_visdom -b 1500
```

- **Мониторинг Обучения:** Следите за метриками LOSS и ошибками, чтобы убедиться, что модель обучается корректно. Обучение продолжается без ограничения по шагам.

### 3. Встраивание Mel-Спектрограммы

```bash
python ./synthesizer_preprocess_embeds.py ./dataset --encoder_model_fpath ./saved_models/01/encoder.pt --n_processes 4
```

### 4. Обучение Синтезатора

```bash
python ./synthesizer_train.py 01 ./dataset2 -m ./saved_models/ -s 2000
```

### 5. Предобработка для Вокодера

```bash
python ./vocoder_preprocess.py -s ./saved_models/01/synthesizer.pt -i ./dataset/ -o ./dataset/vocoder/
```

### 6. Обучение Вокодера

```bash
python ./vocoder_train.py 01 ./dataset --syn_dir ./dataset/synthesizer --voc_dir ./dataset/vocoder -g ./dataset/synthesizer/mels -b 2000
```

---

## Устранение Проблем

- **Проверка версии драйвера NVIDIA:**

  Используйте `nvidia-smi`, чтобы проверить версию драйвера NVIDIA.

    ```bash
    nvidia-smi
    ```

- **Проблемы с зависимостями:**

  Убедитесь, что все версии пакетов совместимы. Проверьте `requirements.txt` и при необходимости скорректируйте версии.

- **Совместимость GPU:**

  Убедитесь, что ваша GPU поддерживается установленной версией CUDA.

---

## Дополнительные Ресурсы

- [Документация NVIDIA](https://docs.nvidia.com/)
- [Документация Anaconda](https://docs.anaconda.com/)
- [Шпаргалка Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)

---

*Удачной работы!*

---
