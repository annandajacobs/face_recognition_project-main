import cv2 
import os
import numpy as np
import face_recognition
import mediapipe as mp

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)


# Função para carregar os rostos conhecidos e suas características faciais
def load_known_faces():
    known_faces = {}
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith('.jpg'):
            name = filename.split('_')[0]  # Nome antes do "_X.jpg"
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            img = cv2.imread(img_path)

            # Usando face_recognition para encontrar as características faciais
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img_rgb)

            if face_encodings:
                known_faces[name] = face_encodings[0]  # Usando a primeira face encontrada
            else:
                print(f"Não foi possível encontrar um rosto em {filename}.")
    return known_faces


# Função para capturar rostos e identificá-los
def capture_and_identify_faces():
    mp_face_detection = mp.solutions.face_detection
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    # Carrega os rostos conhecidos
    known_faces = load_known_faces()

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = video.read()
            if not ret:
                print("Falha ao capturar o vídeo")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    # Coordenadas dentro dos limites
                    x, y = max(x, 0), max(y, 0)
                    w_box, h_box = min(w_box, w - x), min(h_box, h - y)

                    # Recorte do rosto
                    crop_img = frame[y:y + h_box, x:x + w_box]
                    if crop_img.size == 0:
                        continue

                    # Usando face_recognition para extrair as características faciais do rosto capturado
                    rgb_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(rgb_crop_img)

                    if face_encodings:
                        captured_face_encoding = face_encodings[0]

                        # Comparar com rostos conhecidos usando a distância cosseno
                        min_distance = float('inf')
                        name = "Desconhecido"
                        for known_name, known_face_encoding in known_faces.items():
                            distance = np.linalg.norm(known_face_encoding - captured_face_encoding)
                            if distance < min_distance:
                                min_distance = distance
                                name = known_name

                        # Se o rosto for desconhecido
                        if min_distance > 0.6:  # Ajuste do limite de distância
                            name = "Desconhecido"

                        color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)

                        # Define a posição do texto (abaixo do rosto)
                        text = f"Nome: {name}"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        text_x, text_y = x, y + h_box + text_height + 10  # Ajuste o deslocamento conforme necessário

                        # Desenha a caixinha (fundo para o texto)
                        cv2.rectangle(
                            frame, 
                            (text_x - 5, text_y - text_height - 5),  # Posição superior esquerda
                            (text_x + text_width + 5, text_y + 5),  # Posição inferior direita
                            color, 
                            cv2.FILLED
                        )

                        # Desenha o texto em cima da caixinha
                        cv2.putText(
                            frame, 
                            text, 
                            (text_x, text_y), 
                            cv2.FONT_HERSHEY_COMPLEX, 
                            1, 
                            (255, 255, 255),  # Cor do texto
                            1
                        )

                        # Desenha o retângulo em volta do rosto
                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)

            # Exibe o frame
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

# Execução principal
if __name__ == "__main__":
    capture_and_identify_faces()