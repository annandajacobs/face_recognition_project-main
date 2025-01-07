import os
import cv2
import face_recognition
import numpy as np

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Função para carregar rostos conhecidos
def load_known_faces():
    known_faces = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.png')):  # Suporte a múltiplos formatos
            name = filename.split('_')[0]  # Nome antes do "_X.jpg"
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            # Carregar imagem e calcular os "encodings"
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Verifica se encontrou um rosto
                known_faces.append(encodings[0])
                known_names.append(name)
    return known_faces, known_names

# Função para capturar e identificar rostos
def compare_image_with_known_faces(image_path):
    # Carrega os rostos conhecidos
    known_faces, known_names = load_known_faces()
    if not known_faces:
        return [{"error": "Nenhum rosto conhecido foi carregado."}]  # Retorna lista com um dicionário de erro

    # Carregar imagem de entrada
    if not os.path.exists(image_path):
        return [{"error": f"Arquivo não encontrado: {image_path}"}]

    image = face_recognition.load_image_file(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Localiza rostos no frame
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_encodings:
        return [{"error": "Nenhum rosto foi detectado na imagem fornecida."}]

    # Dicionário de resultados
    results = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Comparação com rostos conhecidos
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < 0.5:  # Limite ajustável
            name = known_names[best_match_index]
            status = "Conhecido"
        else:
            name = "Desconhecido"
            status = "Desconhecido"

        # Salvar o resultado em um dicionário
        results.append({
            "name": name,
            "status": status
        })

        # Desenhar o nome e o bounding box na imagem
        top, right, bottom, left = face_location
        color = (0, 255, 0) if status == "Conhecido" else (0, 0, 255)
        cv2.rectangle(rgb_image, (left, top), (right, bottom), color, 2)
        cv2.putText(rgb_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Mostrar ou salvar a imagem resultante
    return results

# Execução principal
if __name__ == "__main__":
    image_path = input("Digite o caminho da imagem para análise: ").strip()
    result = compare_image_with_known_faces(image_path)
    
    if isinstance(result, list) and "error" in result[0]:  # Verifica se há erro nos resultados
        print(result[0]["error"])
    else:
        print("Resultados da análise:")
        for res in result:
            print(f"Nome: {res['name']}, Status: {res['status']}")

