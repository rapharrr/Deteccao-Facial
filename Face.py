import cv2
import requests
import numpy as np

# Baixar o arquivo XML do classificador de faces
cascade_url = 'https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml'
cascade_file = 'haarcascade_frontalface_default.xml'
response = requests.get(cascade_url)
with open(cascade_file, 'wb') as file:
    file.write(response.content)

# URLs das imagens
url1 = 'https://i.scdn.co/image/ab67616100005174cdce7620dc940db079bf4952'
url2 = 'https://www.eusemfronteiras.com.br/wp-content/uploads/2019/11/shutterstock_1392193913-810x540.jpg'
url3 = 'https://img.freepik.com/fotos-premium/diversas-pessoas-juntas-parceria-de-trabalho-em-equipe_53876-38800.jpg?w=2000'

# Função para exibir imagem usando OpenCV
def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Processamento das imagens
urls = [url1, url2, url3]
for url in urls:
    # Fazer o download da imagem
    response = requests.get(url)
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)

    if img is None:
        print('Erro ao carregar a imagem.')
        continue

    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray is None:
        print('Erro ao converter a imagem para escala de cinza.')
        continue

    # Carregar o classificador de faces
    face_cascade = cv2.CascadeClassifier(cascade_file)

    # Detectar faces na imagem
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Desenhar retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Exibir a imagem com as faces detectadas
    show_image(img)