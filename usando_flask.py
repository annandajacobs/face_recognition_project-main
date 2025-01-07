from flask import Flask, jsonify, send_from_directory
import os
import json

app = Flask(__name__)

# Caminho para o diretório de imagens
IMAGES_DIR = 'images'
DATA_FILE = 'image_data.json'  # Arquivo JSON com os dados falsos

# Verifica se o diretório de imagens existe
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Dados falsos para as imagens
image_data = {
    "annanda_5.jpg": {
        "idade": 25,
        "profissao": "Engenheiro",
        "local": "São Paulo",
        "interesses": ["Tecnologia", "Inovação", "Robótica"]
    },
    "joao_3.jpg": {
        "idade": 30,
        "profissao": "Médico",
        "local": "Rio de Janeiro",
        "interesses": ["Saúde", "Tecnologia", "Correr"]
    },
    "ana.jpg": {
        "idade": 22,
        "profissao": "Designer",
        "local": "Belo Horizonte",
        "interesses": ["Design Gráfico", "Arte", "Fotografia"]
    }
}

# Função para carregar os dados falsos
def load_image_data():
    return image_data

# Rota para listar todas as imagens e seus dados falsos
@app.route('/images/lista', methods=['GET'])
def get_image_list():
    try:
        # Listar todas as imagens no diretório
        images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
        
        # Carregar dados falsos
        image_data = load_image_data()

        # Preparar a resposta com os dados falsos
        images_info = []
        for image in images:
            image_info = image_data.get(image, {})
            images_info.append({
                "image": image,
                "data": image_info
            })

        return jsonify({"images": images_info})
    except Exception as e:
        return jsonify({"error": f"Erro ao listar imagens: {str(e)}"}), 500

# Rota para servir uma imagem específica
@app.route('/images/<filename>', methods=['GET'])
def serve_image(filename):
    try:
        # Verificar se o arquivo existe
        image_path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(image_path):
            return send_from_directory(IMAGES_DIR, filename)
        else:
            return jsonify({"error": "Imagem não encontrada"}), 404
    except Exception as e:
        return jsonify({"error": f"Erro ao servir imagem: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
