import cv2 
import os
import face_recognition

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

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
def is_face_registered(new_face_encoding, registered_encodings, tolerance=0.6):
    """Verifica se a face já está registrada no sistema."""
    for registered_encoding in registered_encodings.values():
        match = face_recognition.compare_faces([registered_encoding], new_face_encoding, tolerance)
        if match[0]:  # Se verdadeiro, a face já está registrada
            return True
    return False

# Função para capturar e salvar rostos
def capture_faces_for_person(img, name, cpf):

    registered_encodings = carregar_faces_registradas()

    image = face_recognition.load_image_file(img)
    # Convertendo a imagem para RGB para a detecção de rostos
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detecta rostos
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    if not face_encodings:
        print("Nenhum rosto detectado na imagem.")
        return
    
    photo_count = 1

    for face_location, face_encoding in zip(face_locations, face_encodings):
        # Converte as coordenadas para a escala original
        top, right, bottom, left = face_location
        # Desenha o retângulo ao redor da face detectada
        cv2.rectangle(rgb_img, (left, top), (right, bottom), (0, 255, 0), 2)
        # Verifica se a face já está registrada

        if is_face_registered(face_encoding, registered_encodings):
            print("{name} do CPF {cpf} já foi registrada. Não é possível registrar novamente.")
            break

        # Recorta a imagem do rosto
        face_image = rgb_img[top:bottom, left:right]
        # Aplicar pré-processamento
        # 1. Converter para escala de cinza (opcional)
        face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        # 3. Redimensionar para o tamanho padrão
        face_image_resized = cv2.resize(face_image_gray, (150, 150))

        # Salvar a imagem processada
        filename = os.path.join(KNOWN_FACES_DIR, f"{cpf}_{photo_count}.jpg")
        cv2.imwrite(filename, face_image_resized)
        photo_count += 1
        print(f"Foto {photo_count - 1} salva para {cpf}.")

# Execução principal
if __name__ == "__main__":
    # caminho da imagem 
    img = input("Digite o caminho da imagem: ").strip()

    # Solicita o nome da pessoa
    name = input("Digite o nome da pessoa para registrar: ")

    cpf = input("Digite o cpf da pessoa para registrar: ")
    if len(cpf) != 11:
        print('ERRO. CPF deve ter 11 dígitos.')
    else:
        # Captura rostos para o novo registro
        capture_faces_for_person(img, name, cpf)

    print("Registro concluído.")