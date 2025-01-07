import os
from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
from PIL import Image
from io import BytesIO

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
def compare_image_with_known_faces(file):
    # Carrega os rostos conhecidos
    known_faces, known_names = load_known_faces()
    if not known_faces:
        return "Nenhum rosto conhecido foi carregado."

    try:
        # Abrir o arquivo como uma imagem
        image = Image.open(file)
        image = np.array(image)  # Converter para array NumPy
    except Exception as e:
        return f"Erro ao processar a imagem: {str(e)}"

    # Localiza rostos no frame
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_encodings:
        return "Nenhum rosto foi detectado na imagem fornecida."

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

        # Salvar o resultado
        results.append({
            "name": name,
            "status": status,
            "location": face_location,
        })

    return results

# Configuração do Flask
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo foi enviado."}), 400

    file = request.files['file']  # Obter o arquivo enviado

    # Processar a imagem
    result = compare_image_with_known_faces(file)
    if isinstance(result, str):  # Caso tenha ocorrido algum erro
        return jsonify({"error": result}), 400

    return jsonify({"results": result})

# Execução do servidor Flask
if __name__ == '__main__':
    app.run(debug=True)
