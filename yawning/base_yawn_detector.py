import os
import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance as dist

# ---------------------------------------------------------
# Cálculo matemático de MAR (Mouth Aspect Ratio)
# ---------------------------------------------------------
def calcular_mar(boca):
    """
    Calcula el Mouth Aspect Ratio (MAR) utilizando los puntos
    de los labios internos del modelo de 68 landmarks.
    """
    # Distancias verticales internas (puntos 61-67, 62-66, 63-65)
    A = dist.euclidean(boca[13], boca[19])
    B = dist.euclidean(boca[14], boca[18])
    C = dist.euclidean(boca[15], boca[17])
    
    # Distancia horizontal interna (puntos 60-64)
    D = dist.euclidean(boca[12], boca[16])
    
    if D == 0:
        return 0.0
    return (A + B + C) / (2.0 * D)

# ---------------------------------------------------------
# Función para aplicar padding configurable a la Bounding Box
# ---------------------------------------------------------
def aplicar_padding_bb(rect, frame_shape, pad_factor_x=0.10, pad_factor_y=0.10):
    """
    Aplica un margen de padding proporcional al ancho y alto de la
    Bounding Box (BB) detectada, asegurando no exceder los límites del fotograma.
    
    :param rect: dlib.rectangle original detectado
    :param frame_shape: dimensiones del fotograma (alto, ancho)
    :param pad_factor_x: porcentaje de padding horizontal (ej. 0.10 = +10% a cada lado)
    :param pad_factor_y: porcentaje de padding vertical (ej. 0.15 = +15% a cada lado)
    :return: (dlib.rectangle con padding, (x1, y1, ancho, alto))
    """
    h_frame, w_frame = frame_shape[:2]
    
    x = rect.left()
    y = rect.top()
    w = rect.width()
    h = rect.height()

    pad_w = int(w * pad_factor_x)
    pad_h = int(h * pad_factor_y)

    # Clamping: delimitar a los límites de la imagen
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_frame, x + w + pad_w)
    y2 = min(h_frame, y + h + pad_h)

    rect_padded = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
    return rect_padded, (x1, y1, x2 - x1, y2 - y1)

# ---------------------------------------------------------
# Configuración inicial y modelos
# ---------------------------------------------------------
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    parent_path = os.path.join(os.path.dirname(__file__), "..", PREDICTOR_PATH)
    if os.path.exists(parent_path):
        PREDICTOR_PATH = parent_path

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Rango de landmarks para la boca (48 a 67)
(mStart, mEnd) = (48, 68)

# ---------------------------------------------------------
# Parámetros configurables
# ---------------------------------------------------------
# Padding para la Bounding Box (BB)
# 0.0 = sin padding, 0.10 = 10% adicional por lado, 0.20 = 20%, etc.
PADDING_FACTOR_X = 0.0  # Margen horizontal (+10% izquierda y derecha)
PADDING_FACTOR_Y = 0.0  # Margen vertical (+10% arriba y abajo)

# Umbrales y tiempos de bostezo
UMBRAL_MAR_BOSTEZO = 0.55        # MAR mínimo considerado como apertura excesiva
DURACION_LIMITE_BOSTEZO = 4.0    # Tiempo mínimo continuo (segundos) para confirmar bostezo
TIEMPO_PERSISTENCIA_ALARMA = 1.5 # Tiempo que permanece visible la alerta tras cerrar la boca

# Variables de estado
en_bostezo = False
contador_bostezo = 0.0           # Tiempo acumulado de bostezo (segundos)
t_ultimo_bostezo = 0.0
alarma_bostezo = False

# ---------------------------------------------------------
# Captura de video en tiempo real
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

print(f"[INFO] Iniciando detección de bostezo con Padding (X: {PADDING_FACTOR_X*100:.0f}%, Y: {PADDING_FACTOR_Y*100:.0f}%)...")
print("[INFO] Presiona 'q' para salir.")

t_prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el fotograma de la cámara.")
        break

    t_actual = time.time()
    dt = max(0.0, min(t_actual - t_prev, 0.5))
    t_prev = t_actual

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector(gray, 0)

    rostro_detectado = False
    mar_actual = 0.0

    for rostro_orig in rostros:
        rostro_detectado = True

        # ---------------------------------------------------------
        # Aplicación de Padding configurable a la Bounding Box
        # ---------------------------------------------------------
        rostro_con_pad, (x, y, w, h) = aplicar_padding_bb(
            rostro_orig, 
            frame.shape, 
            pad_factor_x=PADDING_FACTOR_X, 
            pad_factor_y=PADDING_FACTOR_Y
        )

        # El predictor recibe el rostro con el padding aplicado
        shape = predictor(gray, rostro_con_pad)

        # Extraer puntos faciales
        puntos = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            puntos[i] = (shape.part(i).x, shape.part(i).y)

        boca = puntos[mStart:mEnd]
        mar_actual = calcular_mar(boca)

        # Dibujar BB original (línea punteada/delgada blanca de referencia)
        cv2.rectangle(frame, 
                      (rostro_orig.left(), rostro_orig.top()), 
                      (rostro_orig.right(), rostro_orig.bottom()), 
                      (180, 180, 180), 1)

        # Dibujar BB con padding (color principal según estado de alarma)
        color_box = (0, 0, 255) if alarma_bostezo else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_box, 2)

        # Contorno de los labios
        cv2.drawContours(frame, [cv2.convexHull(boca)], -1, (255, 255, 0), 1)

        # ---------------------------------------------------------
        # Lógica de detección de bostezo
        # ---------------------------------------------------------
        if mar_actual > UMBRAL_MAR_BOSTEZO:
            contador_bostezo = min(DURACION_LIMITE_BOSTEZO, contador_bostezo + dt)
            en_bostezo = True

            if contador_bostezo >= DURACION_LIMITE_BOSTEZO:
                alarma_bostezo = True
                t_ultimo_bostezo = t_actual
        else:
            # Si MAR baja del umbral (parpadeo, movimiento o cierre paulatino):
            # NO resetear súbitamente: decrementar con el mismo paso temporal dt hasta llegar a 0
            if contador_bostezo > 0:
                contador_bostezo = max(0.0, contador_bostezo - dt)
            if contador_bostezo == 0:
                en_bostezo = False

        # ---------------------------------------------------------
        # Información visual sobre el rostro
        # ---------------------------------------------------------
        pos_texto_y = y - 15 if y >= 25 else y + h + 25
        if alarma_bostezo:
            cv2.putText(frame, "ALERTA: BOSTEZO PROLONGADO (>4s)", (x, pos_texto_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2)
        elif contador_bostezo > 0:
            cv2.putText(frame, f"Boca abierta: {contador_bostezo:.1f}s / {DURACION_LIMITE_BOSTEZO:.0f}s", 
                        (x, pos_texto_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        break  # Procesar el primer rostro detectado

    # Persistencia de alarma de bostezo
    if alarma_bostezo and (t_actual - t_ultimo_bostezo) > TIEMPO_PERSISTENCIA_ALARMA:
        alarma_bostezo = False

    # Si no se detecta rostro
    if not rostro_detectado:
        # En lugar de reiniciar a cero de inmediato, decrementar en los mismos pasos temporales (con time) hasta 0
        if contador_bostezo > 0:
            contador_bostezo = max(0.0, contador_bostezo - dt)
        if contador_bostezo == 0:
            en_bostezo = False

        cv2.putText(frame, "Buscando rostro...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if alarma_bostezo:
            cv2.putText(frame, "ALERTA: BOSTEZO PROLONGADO (>4s)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2)
        elif contador_bostezo > 0:
            cv2.putText(frame, f"Boca abierta: {contador_bostezo:.1f}s / {DURACION_LIMITE_BOSTEZO:.0f}s (decrementando)", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    else:
        # Panel superior de métricas
        color_mar = (0, 140, 255) if mar_actual > UMBRAL_MAR_BOSTEZO else (255, 255, 255)
        cv2.putText(frame, f"MAR: {mar_actual:.2f} (Umbral: {UMBRAL_MAR_BOSTEZO:.2f}) | Bostezo: {contador_bostezo:.2f}s / {DURACION_LIMITE_BOSTEZO:.1f}s", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_mar, 2)
        cv2.putText(frame, f"BB Padding: X={PADDING_FACTOR_X*100:.0f}% Y={PADDING_FACTOR_Y*100:.0f}%", 
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Detector de Bostezos (AF - Solo Yawning)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
