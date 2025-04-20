

import cv2
import numpy as np

def identify_shape(contour):
    # Aproxima o contorno para identificar a forma
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 3:
        return "Triângulo"
    elif len(approx) == 4:
        # Verifica se é um quadrado ou retângulo
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Quadrado" if 0.95 <= aspect_ratio <= 1.05 else "Retângulo"
    elif len(approx) > 4:
        return "Círculo"
    return "Forma desconhecida"

def identify_color(hsv, x, y):
    # Obtém o valor HSV do pixel
    pixel = hsv[y, x]
    h, s, v = pixel

    if s < 50:
        return "Cinza"
    elif h < 15 or h > 165:
        return "Vermelho"
    elif 15 <= h < 35:
        return "Amarelo"
    elif 35 <= h < 85:
        return "Verde"
    elif 85 <= h < 125:
        return "Azul"
    elif 125 <= h < 165:
        return "Roxo"
    return "Cor desconhecida"

# Carrega a imagem
image = cv2.imread("image.jpg")
resized = cv2.resize(image, (600, 400))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

# Aplica threshold e encontra contornos
_, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Ignora pequenos contornos
    if cv2.contourArea(contour) < 500:
        continue

    # Identifica a forma
    shape = identify_shape(contour)

    # Obtém o centro do contorno para identificar a cor
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        color = identify_color(hsv, cX, cY)

        # Desenha o contorno e escreve a forma e cor
        cv2.drawContours(resized, [contour], -1, (0, 255, 0), 2)
        cv2.putText(resized, f"{shape}, {color}", (cX - 50, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Exibe a imagem
cv2.imshow("Resultado", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()