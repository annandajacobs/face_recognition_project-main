import os
import cv2
import face_recognition
import numpy as np
import datetime
import json

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'
RECOGNITION_LOG_FILE = 'recognition_log.json'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Função para carregar rostos conhecidos
def load_known_faces():
    known_faces = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith('.jpg'):
            name = filename.split('_')[0]  # Nome antes do "_X.jpg"
            name = os.path.splitext(name)[0]  # Remover a extensão .jpg do nome
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            # Carregar imagem e calcular as "encodings"
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Verifica se encontrou um rosto
                known_faces.append(encodings[0])
                known_names.append(name)
    return known_faces, known_names

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
def capture_and_identify_faces():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    # Carrega os rostos conhecidos
    known_faces, known_names = load_known_faces()

    # Carrega o histórico de reconhecimentos
    recognition_log = load_recognition_log()

    while True:
        ret, frame = video.read()
        if not ret:
            print("Falha ao capturar o vídeo")
            break

        # Convertendo para RGB (necessário para face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Localiza rostos no frame
        face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Comparação com rostos conhecidos
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.5:  # Limite ajustável
                name = known_names[best_match_index]
            else:
                name = "Desconhecido"

            # Desenhar o nome e o bounding box no frame
            top, right, bottom, left = face_location
            color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            frame = create_panel(frame, name)

            # Se a pessoa for reconhecida (não for "Desconhecido")
            if name != "Desconhecido":
                now = datetime.datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")

                # Verifica se já foi registrada anteriormente
                if name in recognition_log:
                    # Obter o tempo do último reconhecimento registrado
                    last_time_str = recognition_log[name][-1]
                    last_time = datetime.datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")

                    # Verificar se já se passaram 60 segundos
                    time_diff = now - last_time
                    if time_diff.total_seconds() >= 60:
                        recognition_log[name].append(current_time)
                        print(f"{name} reconhecido novamente às {current_time}")
                else:
                    # Se for a primeira vez que é reconhecida, registra
                    recognition_log[name] = [current_time]
                    print(f"{name} reconhecido pela primeira vez às {current_time}")

        # Exibe o frame
        cv2.imshow("Reconhecimento Facial", frame)
        save_recognition_log(recognition_log) # Salva o histórico de reconhecimentos no arquivo JSON

        # Parar o loop ao pressionar 'q'
        if cv2.waitKey(1) == ord('q'):
            break  
    
    video.release()
    cv2.destroyAllWindows()

# Execução principal
if __name__ == "__main__":
    capture_and_identify_faces()
