import pickle
import random
import sys
import os
import time
import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(1, "./models")
from transformations import geometric2D

def closeApp(exitCode):
    cap.release()
    cv2.destroyAllWindows()
    exit(exitCode)

# Configurações
MIN_PROB = 0.0
QUIT_KEY = 113
WINDOW_NAME = "App"
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(WORKING_DIR, "models/best_model.sav")
model = pickle.load(open(model_path, 'rb'))

# Lista de palavras para o jogo
words = ["AZUL", "VERDE", "ROSA", "AMARELO", "ROXO"]

# Função para escolher uma nova palavra e resetar o progresso
def get_new_word():
    return random.choice(words), 0  # Retorna nova palavra e índice da letra inicial

# Inicializa a primeira palavra e progresso
current_word, current_letter_index = get_new_word()
last_time = time.time()
show_correct = False

# Inicializa captura de vídeo e valida câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("\033[31mNão foi possível abrir a câmera, saindo do programa !!!\033[0m")
    closeApp(1)
else:
    print("Câmera iniciada")

# Cria janela do OpenCV
cv2.namedWindow(WINDOW_NAME)

# Configuração do detector de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

while cv2.waitKeyEx(1) != QUIT_KEY:
    # Captura frame e converte para RGB
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa landmarks da mão
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Coleta dados para predição
        data_aux = []
        x_min, x_max = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].x
        y_min, y_max = hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].y
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            z = hand_landmarks.landmark[i].z
            data_aux.append(x)
            data_aux.append(y)
            data_aux.append(z)

            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        x1 = int(x_min * W) - 10
        y1 = int(y_min * H) - 10
        x2 = int(x_max * W) - 10
        y2 = int(y_max * H) - 10
        data_aux.append(results.multi_handedness[0].classification[0].label == "Left")

        # Predição
        input_values = geometric2D(data_aux)
        predicted_character = model.predict([input_values])[0]

        # Lógica para verificar se a letra está correta na sequência
        current_time = time.time()
        if show_correct:
            # Exibir letra correta por 2 segundos
            if (current_time - last_time) >= 2:
                show_correct = False
                last_time = current_time
                # Avança para a próxima letra ou nova palavra
                if current_letter_index == len(current_word) - 1:
                    current_word, current_letter_index = get_new_word()
                else:
                    current_letter_index += 1
        elif predicted_character == current_word[current_letter_index]:
            show_correct = True
            last_time = current_time
            message = f"Correto! Letra: {predicted_character}"
            color = (0, 255, 0)
        else:
            message = f"Soletrando: {current_word}. Próxima letra: {current_word[current_letter_index]}"
            color = (0, 0, 255)

        # Desenha a caixa e mostra a mensagem
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, message, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

    # Exibe a palavra-alvo e o progresso
    progress_message = f"Palavra: {current_word[:current_letter_index]}_{current_word[current_letter_index:]}"
    cv2.putText(frame, progress_message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Fecha janela se "x" for clicado
    if not cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE):
        break

    # Mostra o frame
    cv2.imshow(WINDOW_NAME, frame)

# Fecha aplicação
closeApp(0)
