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

def closeApp(exitCode):
    cap.release()
    cv2.destroyAllWindows()
    exit(exitCode)

# Configurações - MODELO IA
MIN_PROB = 0.0
QUIT_KEY = 113
WINDOW_NAME = "Câmera"
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(WORKING_DIR, "models/best_model.sav")
model = pickle.load(open(model_path, 'rb'))

# Dicionário de categorias de palavras
categories = {
    "Animais": ["GATO", "CAO", "PATO", "GALO", "GAMBA", "CABRA", "COALA", "SAPO", "RATO", "TARTARUGA", 
                "BALEIA", "RAPOSA", "ELEFANTE", "TIGRE", "LOBO", "COIOTE", "URSO", "ALCE", "LONTRA", 
                "ROBALO", "CANARIO", "TATU", "PINGUIM", "POMBA", "VEADO", "TOUPEIRA", "GANSO", "LAGARTO", 
                "LEBRE", "CAPIVARA", "COBRA", "FOCA", "SABIA", "SERIEMA", "MARRECO", "RA", "PARDAL", 
                "POMBO", "FALCAO", "FLAMINGO", "ARARA", "PELICANO", "MORCEGO", "LEAO", "LEMURE", 
                "PAPAGAIO", "BUFALO", "CORVO", "FORMIGA", "LINCE", "GAVIAO", "ORCA", "ANTA", "SALMAO", 
                "PERIQUITO", "SURICATA", "MACACO", "LAGOSTA", "SALAMANDRA", "CODORNA", "CACATUA", "PONEI", 
                "PACA", "GAIVOTA", "BAGRE", "LEOA", "PERERECA", "URUBU", "MORSA", "EMA", 
                "CARPA", "TILAPIA", "MARIPOSA", "GRILO", "ORNITORRINCO", "FURAO", "IGUANA", 
                "TUCANO", "CASCAVEL", "ANEMONA", "GORILA", "ENGUIA", "ARRAIA", "POLVO", 
                "TRUTA", "SIRI", "GAROUPA", "PIRARUCU", "ATUM", "JAGUATIRICA", "PUMA", "CERVO", "ANTILOPE", 
                "BOI", "VACA", "BORBOLETA", "TUBARAO", "LULA", "CAMARAO", "PORCO"],
    "Frutas": ["PERA", "BANANA", "UVA", "LIMAO", "MANGA", "MELAO", "MELANCIA", "AMORA", "ROMA", 
               "GOIABA", "PITANGA", "GRAVIOLA", "ACEROLA", "ABACATE", "FRAMBOESA", "FIGO", "TAMARA", "PESSEGO", 
               "MAMAO", "MORANGO", "CAQUI", "TAMARINDO", "CARAMBOLA", "NESPRA", "GUARANA", 
               "DAMASCO", "LIMA", "CACAU", "TANGERINA", "BUTIA", "BURITI", "MIRTILO", 
               "PEQUI", "TUCUMA", "ATEMOIA", "UMBU", "BACURI", "AVELÃ", 
               "INGA"],
    "Objetos": ["MESA", "CADEIRA", "POTE", "CANETA", "LAPIS", "LIVRO", "TESOURA", "PAPEL", "LAMPADA", 
                "COPO", "PRATO", "GARFO", "PANELA", "SOFA", "TAPETE", "BOLSA", "CINTO", "ANEL", "BRINCO", 
                "RELOGIO", "CAMA", "COBERTOR", "ALMOFADA", "FACA", "CADERNO", "PORTA", "QUADRO", "VELA", 
                "BALDE", "ESCOVA", "PENTE", "SOFA", "COMPUTADOR", "CELULAR", "MONITOR", "TECLADO", "MOUSE", 
                "IMPRESSORA", "FONE", "ANTENA", "RADIO", "BATERIA", "BOLA", "BONECA", "BICICLETA", "TAPETE", 
                "LOUSA", "ALICATE", "MARTELO", "SERRA", "REGUA", "PINCEL", "TINTA", "BALAO", "COLA", 
                "TESOURA", "ENVELOPE", "CLIPES", "ALFINETE", "PREGO", "PARAFUSO", "VASO", "FLOR", "GARRAFA", 
                "BULE", "CANECA", "ESPATULA", "BANCO", "MESA", "MALETA", "TABULEIRO", "ESPADA", "LANTERNA", 
                "SELA", "FLAUTA", "VIOLINO", "GUITARRA", "TAMBOR", "APITO", "RAQUETE", "REDE", "PILAO", 
                "MOEDOR", "RALADOR", "COADOR", "ESCORREDOR", "MORINGA", "FUNIL", "LENTE", "SACO", "BALANCA", 
                "CORDA", "CORDÃO", "RELOGIO", "BUSSOLA", "CESTO", "COFRE", "VASSOURA", "RODO", "TESOURA", 
                "ARADO", "PLAINA", "BROCA", "FIVELA", "TUBO", "MALA", "GEL", "PA", "DISCO", "FITA", "BACIA", 
                "TAMPA", "REDE", "LUVAS", "VARA", "TABUA", "GRAVATA", "FIO", "PAPEL", "LATA", "TIGELA", "CANO", 
                "SABONETE", "TORNEIRA", "PERFUME"],
    "Cores": ["ANIL", "VERDE", "AMARELO", "PRETO", "BRANCO", "ROSA", "MARROM", "BEGE", "VIOLETA", "LILAS", 
              "SALMAO", "DOURADO", "PRATA", "MAGENTA", "TURQUESA", "COBRE", "PURPURA", 
              "MOSTARDA", "SEPIA", "CORAL", "LAVANDA", "CREME", "AREIA", "MARFIM", "OURO", "CARAMELO", "MENTA", 
              "TERRACOTA", "BORDO", "GOIABA", "RUBI", "TURMALINA", "OLIVA", "UVA", "CIANO"]
}

palavras_usadas = []

# Função para obter nova palavra e resetar progresso
def get_new_word():
    while True:
        palavra = random.choice(palavras_selecionadas)
        if palavra not in palavras_usadas:  # Verifica se a palavra já foi usada
            palavras_usadas.append(palavra)  # Adiciona à lista de palavras usadas
            return palavra, 0, time.time()

def play_game():
    global palavras_selecionadas, palavras_corretas, palavras_total, palavra_tempo
    palavras_corretas = 0
    word_times = []
    palavra_atual, letra_index, tempo_inicial_letra = get_new_word()
    mostrar_correta = False
    letra_errada = False

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME)

    # Configuração do detector de mãos
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

    while cv2.waitKeyEx(1) != QUIT_KEY and palavras_corretas < palavras_total:
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

            tempo_atual = time.time()
            if mostrar_correta:
                if(tempo_atual - tempo_inicial_letra) >= 1:
                    mostrar_correta = False
                    tempo_inicial_letra = tempo_atual
                    if letra_index == len(palavra_atual) - 1:
                        # palavra finalizada
                        palavras_corretas += 1
                        tempo_gasto_palavra = time.time() - tempo_inicial_letra  # Calcula o tempo gasto para a palavra atual
                        word_times.append((palavra_atual, round(tempo_gasto_palavra, 2)))  # Salva a palavra e o tempo

                        if palavras_corretas < palavras_total:
                            palavra_atual, letra_index, tempo_inicial_letra = get_new_word()
                        else:
                            break
                    else:
                        letra_index += 1
            elif predicted_character == palavra_atual[letra_index]:
                mostrar_correta = True
                tempo_inicial_letra = tempo_atual
                letra_errada = False
            else:
                letra_errada = True
            
            # Texto da palavra e letra atual
            cv2.putText(
                frame, f"Palavra: {palavra_atual}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, f"Letra Atual: {palavra_atual[letra_index]}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if not letra_errada else (0, 0, 255), 2, cv2.LINE_AA
            )

        # Exibir quadro atualizado
        cv2.imshow(WINDOW_NAME, frame)

    cap.release()
    cv2.destroyAllWindows()
    

'''
def show_results():
    resultados_tela = tk.Toplevel()
    resultados_tela.title("Resultados")
    resultados_tela.geometry("500x400")
    resultados_texto = tk.Text(resultados_tela, height=15, width=50)
    resultados_texto.pack(pady=10)

    # Exibir resultados
    resultados_texto.insert(tk.END, f"Palavras corretas: {palavras_corretas} de {palavras_total}\n\n")    

    recomecar_bt = tk.Button(resultados_tela, text="Reiniciar", command=lambda: [resultados_tela.destroy(), main()])
    recomecar_bt.pack(pady=10)

    fechar_bt = tk.Button(resultados_tela, text="Fechar", command=resultados_tela.destroy)
    fechar_bt.pack(pady=10)

    
'''

def start_game():
    global categoria_selecionada, palavras_selecionadas, palavras_total
    try:
        palavras_total = int(quantidade_palavra.get())
        categoria_selecionada = categoria.get()
        palavras_selecionadas = random.sample(categories[categoria_selecionada], palavras_total)
        tela_principal.destroy()
        play_game()
    except ValueError:
        messagebox.showerror("Erro", "Por favor, insira um valor numérico para a quantidade de palavras.")

        

def main():
    global tela_principal, quantidade_palavra, categoria

    tela_principal = tk.Tk()
    tela_principal.title("Jogo de Letras com Mãos")
    tela_principal.geometry("500x400")
    tela_principal.configure(bg="#333333")

    categoria = tk.StringVar()
    categoria_label = tk.Label(tela_principal, text="Categoria:", font=("Helvetica", 12), fg="white", bg="#333333")
    categoria_label.pack()
    categoria_drop = ttk.Combobox(tela_principal, textvariable=categoria, values=list(categories.keys()))
    categoria_drop.pack(pady=10)

    palavra_label = tk.Label(tela_principal, text="Quantidade de Palavras:", font=("Helvetica", 12), fg="white", bg="#333333")
    palavra_label.pack()
    quantidade_palavra = tk.Entry(tela_principal)
    quantidade_palavra.pack(pady=10)

    botao_start = tk.Button(tela_principal, text="Start", command=start_game)
    botao_start.pack(pady=20)

    tela_principal.mainloop()

if __name__ == "__main__":
    main()
