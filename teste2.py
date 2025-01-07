import os
import cv2
import numpy as np
import datetime
import json
from insightface.app import FaceAnalysis

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'
RECOGNITION_LOG_FILE = 'recognition_log.json'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Inicialização do InsightFace
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Função para carregar embeddings conhecidos
def load_known_faces():
    known_faces = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith('.jpg'):
            name = os.path.splitext(filename)[0]  # Nome sem extensão
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            # Carregar imagem e calcular os embeddings
            image = cv2.imread(img_path)
            faces = app.get(image)
            if faces:  # Verifica se encontrou um rosto
                known_faces.append(faces[0].embedding)
                known_names.append(name)
    return known_faces, known_names

# Função para criar painel com foto e informações
def create_panel(frame, name):
    person_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    if os.path.exists(person_image_path):
        person_image = cv2.imread(person_image_path)
        person_image = cv2.resize(person_image, (200, 200))  # Ajuste o tamanho da foto
        panel_height = max(frame.shape[0], 200)
        panel_width = frame.shape[1] + 200
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:frame.shape[0], :frame.shape[1]] = frame
        panel[:200, frame.shape[1]:] = person_image
        return panel
    return frame

# Função para carregar o histórico de reconhecimentos
def load_recognition_log():
    if os.path.exists(RECOGNITION_LOG_FILE):
        with open(RECOGNITION_LOG_FILE, 'r') as file:
            return json.load(file)
    return {}

# Função para salvar o histórico de reconhecimentos
def save_recognition_log(recognition_log):
    with open(RECOGNITION_LOG_FILE, 'w') as file:
        json.dump(recognition_log, file, indent=4)

# Função para capturar e identificar rostos
def capture_and_identify_faces_no_display():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    # Carrega os rostos conhecidos
    known_faces, known_names = load_known_faces()

    # Carrega o histórico de reconhecimentos
    recognition_log = load_recognition_log()

    frame_count = 0
    start_time = datetime.datetime.now()

    while True:
        ret, frame = video.read()
        if not ret:
            print("Falha ao capturar o vídeo")
            break

        # Localiza e analisa rostos no frame
        faces = app.get(frame)

        for face in faces:
            embedding = face.embedding

            # Comparar com embeddings conhecidos
            distances = [np.linalg.norm(embedding - known_face) for known_face in known_faces]
            if distances:
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < 1.0:  # Limite ajustável
                    name = known_names[best_match_index]
                else:
                    name = "Desconhecido"
            else:
                name = "Desconhecido"

            # Registrar reconhecimento
            if name != "Desconhecido":
                now = datetime.datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")

                if name in recognition_log:
                    last_time_str = recognition_log[name][-1]
                    last_time = datetime.datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")

                    if (now - last_time).total_seconds() >= 60:
                        recognition_log[name].append(current_time)
                        print(f"{name} reconhecido novamente às {current_time}")
                else:
                    recognition_log[name] = [current_time]
                    print(f"{name} reconhecido pela primeira vez às {current_time}")

        frame_count += 1
        if frame_count % 30 == 0:  # Exibe estatísticas a cada 30 quadros
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            print(f"Processados {frame_count} quadros em {elapsed_time:.2f} segundos "
                  f"({frame_count / elapsed_time:.2f} FPS).")

        save_recognition_log(recognition_log)  # Salva o histórico de reconhecimentos

        # Parar o loop após 5 minutos (ou remova este bloco para rodar indefinidamente)
        if (datetime.datetime.now() - start_time).total_seconds() > 300:
            print("Tempo limite atingido. Encerrando o processamento.")
            break

    video.release()
    print("Captura encerrada.")

# Execução principal
if __name__ == "__main__":
    capture_and_identify_faces_no_display()
