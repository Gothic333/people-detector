## People Detector on Video using YOLO

A simple command-line application for detecting people in video using YOLO. This project was developed as a test task.

### Requirements

- Python >= 3.11
- CUDA-enabled GPU (optional, for faster inference)

### Installation

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate # Linux / macOS
   venv\Scripts\activate.bat # Windows
   ```

2. (Optional) If you want to use GPU for inference, install [PyTorch with CUDA support][1]

3. Install the required dependencies.

   ```bash
   pip install -r requirements.txt
   ```

### Using

Run the script from the terminal:

```bash
python3 inference.py --source 'path/to/video' [options]
```

To display help:

```bash
python3 inference.py -h
```

Command-line Options

| Parameter    | Description                                | Default      |
| ------------ | ------------------------------------------ | ------------ |
| `--model`    | Path to YOLO model weights                 | `yolo11n.pt` |
| `--class_id` | Class ID of the object to detect           | `0` (person) |
| `--source`   | Path to the source video file              | **Required** |
| `--save`     | Directory to save the resulting video      | `./output`   |
| `--device`   | Device to run inference on (cpu or cuda:0) | `cpu`        |
| `--conf`     | Confidence threshold for detections        | `0.5`        |

## Детектор людей на видео с использованием YOLO

Простое консольное приложение для детекции людей на видео с использованием модели YOLO. Этот проект был выполнен как тестовое задание.

### Требования

- Python >= 3.11
- GPU с поддержкой CUDA (опционально, для ускорения инференса)

### Установка

1. Создайте и активируйте виртуальное окружение:

   ```bash
   python3 -m venv venv
   source venv/bin/activate # Linux / macOS
   venv\Scripts\activate.bat # Windows
   ```

2. (Опционально) Установите [PyTorch с поддержкой CUDA][1], если хотите использовать инференс на GPU

3. Установите все зависимости из файла requirements.txt:

   ```bash
   pip install -r requirements.txt
   ```

### Использование

Запустите скрипт из терминала:

```bash
python3 inference.py --source 'path/to/video' [options]
```

Чтобы посмотреть справку по параметрам:

```bash
python3 inference.py -h
```

Параметры командной строки

| Параметр     | Описание                                  | По умолчанию   |
| ------------ | ----------------------------------------- | -------------- |
| `--model`    | Путь к весам модели YOLO                  | `yolo11n.pt`   |
| `--class_id` | ID класса объекта для детекции            | `0` (person)   |
| `--source`   | Путь к исходному видеофайлу               | **Обязателен** |
| `--save`     | Папка для сохранения результата           | `./output`     |
| `--device`   | Устройство для инференса (cpu или cuda:0) | `cpu`          |
| `--conf`     | Порог уверенности детекции                | `0.5`          |

[1]: https://pytorch.org/get-started/locally/
