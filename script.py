!rm -r sample_data
!pip uninstall -y tokenizers transformers
!pip install tokenizers==0.19.0
!pip install --upgrade tokenizers
!pip install transformers==4.41.2
!pip install pydub
!pip install cohere openai tiktoken
!pip install openai-whisper
!pip install dostoevsky
!python3 -m dostoevsky download fasttext-social-network-model
!pip install python-telegram-bot
!pip install requests==2.31.0
!pip install google-colab==1.0.0
!pip install pyannote.audio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip show tokenizers transformers

import requests
import os
from pydub import AudioSegment
import whisper
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import gdown
from transformers import BertTokenizerFast, AutoModelForSequenceClassification
from pyannote.audio import Pipeline

# Загрузите файл с токеном с Google Диска
file_id = '1ZY8V2d_W8GJBk7ASEj6NQ3CNahoWsEZg'
gdown.download(f'https://drive.google.com/uc?id={file_id}', 'token.txt', quiet=False)

# Чтение токена из файла
with open('token.txt', 'r') as file:
    TOKEN = file.read().strip()

# URL API Telegram для запросов
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TOKEN}/"

file_path = "/content/token.txt"
os.remove(file_path)

# Глобальный порог чувствительности
negative_threshold = 0.5
bert_negative_threshold = 0.6

# Функция для отправки текстового сообщения
def send_message(chat_id, text):
    url = TELEGRAM_API_URL + "sendMessage"
    data = {"chat_id": chat_id, "text": text}
    response = requests.post(url, data=data)
    return response.json()

# Функция для получения информации о файле по его file_id
def get_file(file_id):
    url = TELEGRAM_API_URL + f"getFile?file_id={file_id}"
    response = requests.get(url)
    result = response.json()
    if result.get("ok"):
        return result.get("result")
    return None

# Функция для загрузки файла по его пути
def download_file(file_path):
    url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
    response = requests.get(url)
    file_name = file_path.split("/")[-1]
    with open(file_name, "wb") as file:
        file.write(response.content)
    return file_name

# Функция для конвертации времени в формат "часы:минуты:секунды"
def convert_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# Функция для анализа тональности текста каждой фразы
def analyze_sentiment_by_phrase(segments, tokenizer, sentiment_model, dostoevsky_model, negative_threshold=0.3, bert_negative_threshold=0.6):
    results = []
    total_phrases = 0
    negative_phrases = 0
    processed_phrases = set()

    for segment in segments:
        phrase = segment['text'].strip()

        # Проверка на дублирование фразы
        if phrase in processed_phrases:
            continue

        # Добавление фразы в обработанные
        processed_phrases.add(phrase)

        # Анализируем тональность с помощью модели Dostoevsky
        phrase_sentiment = dostoevsky_model.predict([phrase])[0]
        negative_score = phrase_sentiment.get('negative', 0)

        # Анализируем тональность с помощью модели Bert
        inputs = tokenizer(phrase, return_tensors="pt")
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
        probabilities = logits.softmax(dim=1)[0]
        max_prob_index = probabilities.argmax().item()
        sentiment_score = probabilities[max_prob_index].item()

        # Проверка порога негативной тональности
        if (max_prob_index == 0 and sentiment_score > bert_negative_threshold) or negative_score > negative_threshold:
            results.append({
                'speaker': segment.get('speaker', 'N/A'),
                'start_time': convert_time(segment['start']),
                'end_time': convert_time(segment['end']),
                'phrase': phrase,
                'dostoevsky_score': negative_score,
                'bert_score': sentiment_score if max_prob_index == 0 else 0.0
            })
            negative_phrases += 1

        total_phrases += 1

    return results, total_phrases, negative_phrases

# Функция для отправки документа
def send_document(chat_id, document_path):
    url = TELEGRAM_API_URL + "sendDocument"
    with open(document_path, "rb") as document:
        files = {"document": document}
        data = {"chat_id": chat_id}
        response = requests.post(url, files=files, data=data)
    return response.json()

# Обработчик аудиофайлов
def handle_audio(chat_id, audio_file_id):
    file_info = get_file(audio_file_id)
    if file_info is not None and "file_path" in file_info:
        file_path = download_file(file_info["file_path"])
        if file_path is not None:
            send_message(chat_id, "Файл получен и будет обработан.")

            # Преобразуем аудиофайл в формат WAV
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(16000)
            output_wav = 'audio_file.wav'
            audio.export(output_wav, format="wav")

            # Используем модель Whisper для транскрибации
            whisper_model = whisper.load_model('large')
            options = dict(language='ru', beam_size=5, best_of=5)
            transcribe_options = dict(task='transcribe', **options)

            result = whisper_model.transcribe(output_wav, **transcribe_options)
            segments = result['segments']
            transcribed_text = result['text']

            # Инициализация моделей для анализа тональности
            tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment')
            sentiment_model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment', return_dict=True)
            dostoevsky_tokenizer = RegexTokenizer()
            dostoevsky_model = FastTextSocialNetworkModel(tokenizer=dostoevsky_tokenizer)

            # Анализ тональности
            results, total_phrases, negative_phrases = analyze_sentiment_by_phrase(segments, tokenizer, sentiment_model, dostoevsky_model, negative_threshold, bert_negative_threshold)

            # Сохранение полной транскрипции в файл
            transcription_file = 'transcription.txt'
            with open(transcription_file, 'w', encoding='utf-8') as f:
                f.write(transcribed_text)

            # Сохранение результатов анализа в файл
            analysis_file = 'analysis.txt'
            with open(analysis_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"Спикер: {result['speaker']}
")
                    f.write(f"Время: {result['start_time']} - {result['end_time']}
")
                    f.write(f"Фраза: '{result['phrase']}'
")
                    f.write(f"Оценка негатива (Dostoevsky): {result['dostoevsky_score']:.2f}
")
                    f.write(f"Оценка негатива (Bert): {result['bert_score']:.2f}

")
                f.write(f"Общее количество фраз: {total_phrases}
")
                f.write(f"Количество негативных фраз: {negative_phrases}
")
                if total_phrases > 0:
                    percentage_negative = (negative_phrases / total_phrases) * 100
                    f.write(f"Процент негативных фраз: {percentage_negative:.2f}%
")

            # Отправка результатов пользователю
            send_document(chat_id, transcription_file)
            send_document(chat_id, analysis_file)
            send_message(chat_id, "Транскрибация и анализ завершены. Результаты отправлены.")
        else:
            send_message(chat_id, "Не удалось загрузить аудиофайл.")
    else:
        send_message(chat_id, "Не удалось получить информацию о файле.")

# Обработчик команды /start
def start(message):
    chat_id = message["chat"]["id"]
    send_message(chat_id, "Привет! Отправьте мне аудиофайл, и я выполню его транскрибацию и анализ тональности. Размер файла не более 20 МБ. Используйте команду /help для получения дополнительной информации.")

# Обработчик команды /help
def help_command(message):
    chat_id = message["chat"]["id"]
    help_text = (
        "Команды:
"
        "/start - Начало работы с ботом
"
        "/help - Получение помощи
"
        "/set_threshold <value> - Установка порога чувствительности определения негатива (по умолчанию 0.5)
"
    )
    send_message(chat_id, help_text)

# Обработчик команды /set_threshold
def set_threshold(message):
    global negative_threshold, bert_negative_threshold
    chat_id = message["chat"]["id"]
    try:
        new_threshold = float(message.get("text").split()[1])
        if 0 <= new_threshold <= 1:
            negative_threshold = new_threshold
            bert_negative_threshold = new_threshold  # Устанавливаем тот же порог для BERT модели
            send_message(chat_id, f"Порог чувствительности успешно изменен на {negative_threshold}.")
        else:
            send_message(chat_id, "Порог чувствительности должен быть в диапазоне от 0 до 1.")
    except (IndexError, ValueError):
        send_message(chat_id, "Использование: /set_threshold <value>")

# Главная функция для обработки входящих обновлений
def main():
    offset = None
    while True:
        # Получаем обновления через Long Polling
        url = TELEGRAM_API_URL + "getUpdates"
        params = {"timeout": 100, "offset": offset}
        response = requests.get(url, params=params)
        updates = response.json().get("result", [])

        if updates:
            for update in updates:
                offset = update["update_id"] + 1  # Обновляем offset для обработки новых сообщений
                message = update.get("message")
                if message:
                    if "text" in message:
                        text = message["text"]
                        if text.startswith("/start"):
                            start(message)
                        elif text.startswith("/help"):
                            help_command(message)
                        elif text.startswith("/set_threshold"):
                            set_threshold(message)
                    elif "audio" in message:
                        audio_file_id = message["audio"]["file_id"]
                        handle_audio(message["chat"]["id"], audio_file_id)
                    elif "voice" in message:
                        voice_file_id = message["voice"]["file_id"]
                        handle_audio(message["chat"]["id"], voice_file_id)

# Запускаем главную функцию
if __name__ == "__main__":
    main()
