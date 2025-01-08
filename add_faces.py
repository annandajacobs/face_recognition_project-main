import cv2 
import os
import face_recognition

# Configuração dos diretórios
KNOWN_FACES_DIR = 'known_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Testa a câmera antes de abrir
def testar_camera(index=0):
    video = cv2.VideoCapture(index)
    if not video.isOpened():
        print(f"Erro: Não foi possível acessar a câmera no índice {index}.")
        return False
    video.release()
    return True

# Função para carregar as codificações de rostos registrados
def carregar_faces_registradas():
    """Carrega as codificações de rostos salvos no diretório."""
    registered_encodings = {}
    for filename in os.listdir(KNOWN_FACES_DIR):
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        if os.path.isfile(filepath):
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Garante que a face foi detectada
                registered_encodings[filename] = encodings[0]
    return registered_encodings

# Verifica se a face já está registrada
def is_face_registered(new_face_encoding, registered_encodings, tolerance=0.5):
    """Verifica se a face já está registrada no sistema."""
    for registered_encoding in registered_encodings.values():
        match = face_recognition.compare_faces([registered_encoding], new_face_encoding, tolerance)
        if match[0]:  # Se verdadeiro, a face já está registrada
            return True
    return False

# Função para capturar e salvar rostos
def capture_faces_for_person(name):
    """Captura e salva rostos de uma pessoa."""
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    # Definir a resolução desejada da câmera (por exemplo, 1920x1080)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    photo_count = 0
    registered_encodings = carregar_faces_registradas()

    while photo_count < 5:  # Captura 5 fotos por pessoa
        ret, frame = video.read()
        if not ret:
            print("Falha ao capturar o vídeo")
            break

        # Convertendo a imagem para RGB para a detecção de rostos
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecta rostos
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_location, face_encoding in zip(face_locations, face_encodings):
            # Converte as coordenadas para a escala original
            top, right, bottom, left = face_location

            # Desenha o retângulo ao redor da face detectada
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Verifica se a face já está registrada
            if is_face_registered(face_encoding, registered_encodings):
                print("Rosto já registrado. Ignorando.")
                continue

            # Recorta a imagem do rosto
            face_image = frame[top:bottom, left:right]

            # Aplicar pré-processamento
            # 1. Converter para escala de cinza (opcional)
            face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # 3. Redimensionar para o tamanho padrão
            face_image_resized = cv2.resize(face_image_gray, (150, 150))

            # Salvar a imagem processada
            filename = os.path.join(KNOWN_FACES_DIR, f"{name}_{photo_count}.jpg")
            cv2.imwrite(filename, face_image_resized)
            photo_count += 1
            print(f"Foto {photo_count} salva para {name}.")

        # Exibe o frame na tela
        cv2.imshow("Captura de Fotos", frame)

        # Sai ao pressionar 'q'
        if cv2.waitKey(10) == ord('q'):  # Aumentar o tempo para permitir melhor desempenho
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"Captura concluída para {name}. {photo_count} fotos salvas.")

# Execução principal
if __name__ == "__main__":
    if not testar_camera():
        print("Conecte ou configure a câmera corretamente antes de tentar novamente.")
        exit(1)

    # Solicita o nome da pessoa
    name = input("Digite o nome da pessoa para registrar: ")

    # Captura rostos para o novo registro
    capture_faces_for_person(name)

    print("Registro concluído.")
