import os
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#diretorio onde estão as imagens
diretorio_chupeta = r"C:\Users\mateu\Downloads\DATASET\DATASET\chupeta"
diretorio_roendo_unha = r"C:\Users\mateu\Downloads\DATASET\DATASET\roendo_unha"
diretorio_dedo_na_boca = r"C:\Users\mateu\Downloads\DATASET\DATASET\dedo_na_boca"
#concatenar as imagens
image_dirs = [diretorio_chupeta, diretorio_roendo_unha, diretorio_dedo_na_boca]

# Passo 1: Carregar e visualizar as imagens
def load_and_visualize_images(image_dirs):
    image_paths = []
    for dir_path in image_dirs:
        if not os.path.exists(dir_path):
            print(f"Diretório não encontrado: {dir_path}")
            continue
        for fname in os.listdir(dir_path):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(dir_path, fname))

    # Visualizar as imagens carregadas
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(os.path.basename(img_path))
        plt.show()

    return image_paths

# Passo 2: Detecção de Faces usando Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi = (x, y, x + w, y + h)
        return roi
    else:
        return None

# Passo 3: Criação do arquivo CSV
def create_csv(image_paths, csv_path):
    data = []

    for img_path in image_paths:
        roi = detect_face(img_path)
        if roi:
            x1, y1, x2, y2 = roi
            roi_caption = "Face detected in the image"
            image_caption = "Original image without face detection"
            data.append([img_path, f'{x1},{y1},{x2},{y2}', roi_caption, image_caption])

    df = pd.DataFrame(data, columns=['Image Path', 'ROI', 'ROI Caption', 'Image Caption'])
    df.to_csv(csv_path, index=False)
    print(f"Arquivo CSV salvo em: {csv_path}")

# Passo 4: Preparação para treinamento
def prepare_training_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df['Image Path']
    Y = df['ROI Caption']  # ou df['Image Caption']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test

# Execução do pipeline completo
image_paths = load_and_visualize_images(image_dirs)
csv_path = 'dataset.csv'  # Salvar no diretório atual do notebook
create_csv(image_paths, csv_path)
X_train, X_test, Y_train, Y_test = prepare_training_data(csv_path)

# Verificar se o arquivo CSV foi salvo corretamente
if os.path.exists(csv_path):
    print(f"Arquivo CSV salvo com sucesso em: {csv_path}")
else:
    print("Erro ao salvar o arquivo CSV.")
