import os
import cv2
import face_recognition
import numpy as np
import requests
from io import BytesIO
from PIL import Image

# Configuração dos diretórios
KNOWN_FACES_DIR = 'known_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

file_links = [
    "https://drive.google.com/drive/folders/id=1dVV3I6xfwo_sS8fburWvFVweYF71Ynco"
]

def download_images(links, save_path=KNOWN_FACES_DIR):
    for i, link in enumerate(links):
        try:
            response = requests.get(link, stream=True)
            if response.status_code == 200:
                file_name = os.path.join(save_path, f"image_{i+1}.jpg")
                with open(file_name, 'wb') as f:
                    f.write(response.content)
                print(f"Imagem {i+1} salva como: {file_name}")
            else:
                print(f"Erro ao baixar a imagem {i+1}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Erro ao processar o link {link}: {e}")

# Função para carregar rostos conhecidos
def load_known_faces():
    known_faces = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.png')):  # Suporte a múltiplos formatos
            name = filename.split('_')[0]  # Nome antes do "_X.jpg"
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            # Carregar imagem e calcular os "encodings"
            try:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:  # Verifica se encontrou um rosto
                    known_faces.append(encodings[0])
                    known_names.append(name)
            except Exception as e:
                print(f"Erro ao carregar a imagem {filename}: {e}")
    return known_faces, known_names

# Função para obter a lista de imagens e dados falsos da API Flask
def get_images_from_api():
    try:
        response = requests.get('http://127.0.0.1:5000/images/lista')  # Endereço da API
        if response.status_code == 200:
            return response.json()['images']
        else:
            print(f"Erro ao obter lista de imagens. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
    return []

# Função para capturar e identificar rostos
def compare_image_with_known_faces(image):
    # Carrega os rostos conhecidos
    known_faces, known_names = load_known_faces()
    if not known_faces:
        return [{"error": "Nenhum rosto conhecido foi carregado."}]  # Retorna lista com um dicionário de erro

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Localiza rostos no frame
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_encodings:
        return [{"error": "Nenhum rosto foi detectado na imagem fornecida."}]

    # Dicionário de resultados
    results = []

    # Obter a lista de imagens e dados falsos da API
    images_info = get_images_from_api()

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

        # Buscar dados falsos da API para a imagem reconhecida
        image_info = next((item['data'] for item in images_info if item['image'] == f"{name}.jpg"), {})

        # Salvar o resultado em um dicionário
        results.append({
            "name": name,
            "status": status,
            "data": image_info  # Adiciona os dados falsos
        })

        # Desenhar o nome e o bounding box na imagem
        top, right, bottom, left = face_location
        color = (0, 255, 0) if status == "Conhecido" else (0, 0, 255)
        cv2.rectangle(rgb_image, (left, top), (right, bottom), color, 2)
        cv2.putText(rgb_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Mostrar ou salvar a imagem resultante
    return results

# Função para carregar a imagem da API
def get_image_from_api(image_name):
    try:
        response = requests.get(f'http://127.0.0.1:5000/images/{image_name}')
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return np.array(image)
        else:
            print(f"Erro ao obter imagem {image_name}. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
    return None

# Execução principal
if __name__ == "__main__":
    images_info = get_images_from_api()  # Obtém a lista de imagens da API

    if images_info:
        for image_info in images_info:
            image_name = image_info['image']
            image = get_image_from_api(image_name)  # Obtém a imagem da API
            if image is not None:
                result = compare_image_with_known_faces(image)
                
                if isinstance(result, list) and "error" in result[0]:  # Verifica se há erro nos resultados
                    print(result[0]["error"])
                else:
                    print(f"Resultados da análise para a imagem {image_name}:")
                    for res in result:
                        print(f"Nome: {res['name']}, Status: {res['status']}, Dados: {res['data']}")
            else:
                print(f"Erro ao obter a imagem {image_name}.")
    else:
        print("Erro ao obter a lista de imagens da API.")
