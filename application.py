import pickle
import random
import sys
import os
import time
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

sys.path.insert(1, "./models")
from transformations import geometric2D

# Funções de fechamento da aplicação
def closeApp(exitCode):
    cap.release()
    cv2.destroyAllWindows()
    exit(exitCode)

# Configurações iniciais
MIN_PROB = 0.0
QUIT_KEY = 113
WINDOW_NAME = "App"
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(WORKING_DIR, "models/best_model.sav")
model = pickle.load(open(model_path, 'rb'))

# Dicionário de categorias de palavras
categories = {
    "Cores": ["ANIL", "VERDE", "ROSA", "AMARELO", "LILAS"],
    "Animais": ["GALO", "GATO", "PATO", "CAVALO"],
    "Frutas": ["BANANA", "UVA", "MANGA", "MORANGO", "PESSEGO", "FIGO"]
}

# Variáveis para o jogo
selected_category = None
selected_words = []
correct_words = 0
total_words = 0
word_times = []

# Função para obter nova palavra e resetar progresso
def get_new_word():
    return random.choice(selected_words), 0, time.time()  # Adicionando o start_time

# Função principal do jogo
def play_game():
    global selected_words, correct_words, total_words, word_times
    correct_words = 0
    word_times = []
    current_word, current_letter_index, start_time = get_new_word()
    show_correct = False
    wrong_letter_shown = False
    inicio = time.time()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME)

    # Configuração do detector de mãos
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

    while cv2.waitKeyEx(1) != QUIT_KEY and correct_words < total_words:
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
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                data_aux.extend([x, y, z])

            data_aux.append(results.multi_handedness[0].classification[0].label == "Left")

            # Predição
            input_values = geometric2D(data_aux)
            predicted_character = model.predict([input_values])[0]

            # Lógica para trocar a letra após um tempo e detectar erro
            current_time = time.time()
            if show_correct:
                if (current_time - start_time) >= 1:
                    show_correct = False
                    start_time = current_time
                    if current_letter_index == len(current_word) - 1:
                        # Registra o tempo gasto na palavra
                        word_times.append((current_word, time.time() - inicio))
                        correct_words += 1
                        inicio = time.time()
                        if correct_words < total_words:
                            current_word, current_letter_index, start_time = get_new_word()
                        else:
                            break
                    else:
                        current_letter_index += 1
            elif predicted_character == current_word[current_letter_index]:
                show_correct = True
                start_time = current_time
                wrong_letter_shown = False
            else:
                wrong_letter_shown = True

            # Texto da palavra e letra atual
            cv2.putText(
                frame, f"Palavra: {current_word}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, f"Letra Atual: {current_word[current_letter_index]}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if not wrong_letter_shown else (0, 0, 255), 2, cv2.LINE_AA
            )

            # Barra de progresso e contagem de palavras acertadas
            progress_width = int((current_letter_index + 1) / len(current_word) * (W - 40))
            cv2.rectangle(frame, (20, H - 40), (20 + progress_width, H - 20), (0, 255, 255), -1)
            cv2.rectangle(frame, (20, H - 40), (W - 20, H - 20), (255, 255, 255), 2)
            cv2.putText(
                frame, f"Palavras Corretas: {correct_words}/{total_words}", (W - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )

        # Exibir quadro atualizado
        cv2.imshow(WINDOW_NAME, frame)

    # Fecha a captura de vídeo
    cap.release()
    cv2.destroyAllWindows()
    show_results()

# Exibe tela de resultados
def show_results():
    results_window = tk.Toplevel()
    results_window.title("Resultados")
    results_window.geometry("500x400")
    results_text = tk.Text(results_window, height=15, width=50)
    results_text.pack(pady=10)
    
    soma = 0
    for word, elapsed_time in word_times:
        results_text.insert(tk.END, f"Palavra: {word} - Tempo: {elapsed_time:.2f} segundos\n")
        soma += {elapsed_time}
    
    results_text.insert(tk.END, f"TOTAL: {soma:.2f} segundos\n")
    
    # Botão para reiniciar o jogo
    restart_button = tk.Button(results_window, text="Reiniciar", command=lambda: [results_window.destroy(), main()])
    restart_button.pack(pady=10)

    # Botão para fechar o aplicativo
    close_button = tk.Button(results_window, text="Fechar", command=results_window.destroy)
    close_button.pack(pady=10)

# Função para iniciar o jogo após selecionar categoria e quantidade
def start_game():
    global selected_category, selected_words, total_words
    try:
        total_words = int(word_count_entry.get())
        selected_category = category_var.get()
        selected_words = random.sample(categories[selected_category], total_words)
        main_window.destroy()
        play_game()
    except ValueError:
        messagebox.showerror("Erro", "Digite um número válido de palavras.")

# Interface gráfica inicial usando Tkinter
def main():
    global main_window, word_count_entry, category_var
    main_window = tk.Tk()
    main_window.title("Jogo de Letras com Mãos")
    main_window.geometry("500x400")
    main_window.configure(bg="#333333")

    title_label = tk.Label(main_window, text="Selecione a Categoria e Quantidade de Palavras", font=("Helvetica", 16), fg="white", bg="#333333")
    title_label.pack(pady=20)

    # Seleção de categoria
    category_var = tk.StringVar()
    category_label = tk.Label(main_window, text="Categoria:", font=("Helvetica", 12), fg="white", bg="#333333")
    category_label.pack()
    category_dropdown = ttk.Combobox(main_window, textvariable=category_var, values=list(categories.keys()))
    category_dropdown.pack(pady=10)

    # Input para quantidade de palavras
    word_count_label = tk.Label(main_window, text="Quantidade de Palavras:", font=("Helvetica", 12), fg="white", bg="#333333")
    word_count_label.pack()
    word_count_entry = tk.Entry(main_window, font=("Helvetica", 12))
    word_count_entry.pack(pady=10)

    # Botão para iniciar o jogo
    start_button = tk.Button(main_window, text="Start", font=("Helvetica", 14, "bold"), fg="white", bg="#00cc66", command=start_game)
    start_button.pack(pady=20)

    main_window.mainloop()

# Inicia o jogo
if __name__ == "__main__":
    main()
