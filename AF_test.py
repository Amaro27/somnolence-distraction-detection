import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance as dist
from collections import deque

# ---------------------------------------------------------
# Funciones matemáticas para EAR y MAR
# ---------------------------------------------------------
def calcular_ear(ojo):
    # Calcula la distancia vertical entre los párpados
    A = dist.euclidean(ojo[1], ojo[5])
    B = dist.euclidean(ojo[2], ojo[4])
    # Calcula la distancia horizontal
    C = dist.euclidean(ojo[0], ojo[3])
    # Fórmula del Eye Aspect Ratio
    return (A + B) / (2.0 * C)

def calcular_mar(boca):
    # Distancias verticales de la boca interna
    A = dist.euclidean(boca[13], boca[19])
    B = dist.euclidean(boca[14], boca[18])
    C = dist.euclidean(boca[15], boca[17])
    # Distancia horizontal de la boca interna
    D = dist.euclidean(boca[12], boca[16])
    return (A + B + C) / (2.0 * D)

# ---------------------------------------------------------
# Configuración inicial (Importar librerías e iniciar dlib)
# ---------------------------------------------------------
# NOTA: Necesitas descargar el archivo 'shape_predictor_68_face_landmarks.dat' de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Índices de los puntos faciales (Landmarks)
(lStart, lEnd) = (42, 48) # Ojo izquierdo
(rStart, rEnd) = (36, 42) # Ojo derecho
(mStart, mEnd) = (48, 68) # Boca

# Iniciar cámara
cap = cv2.VideoCapture(0)

# ---------------------------------------------------------
# Variables de Estado y Buffers
# ---------------------------------------------------------
estado = "ESPERA"
tiempo_inicio_calibracion = 0
datos_ear_calibracion = []
datos_mar_calibracion = []

treshold_inicial_ear = 0.0
treshold_adaptativo = 0.0
sensibilidad = 0.85 # Factor para que el threshold no sea exactamente el ojo abierto al 100%

# Buffers para los filtros
buffer_crudo_ear = deque(maxlen=5)      # Ventana para el Median Filter (suaviza picos)
buffer_mediano_ear = deque(maxlen=30)   # Ventana para el Moving Average (tendencia general)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector(gray, 0)

    for rostro in rostros:
        # Detectar puntos faciales
        shape = predictor(gray, rostro)
        puntos = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            puntos[i] = (shape.part(i).x, shape.part(i).y)

        # Extraer coordenadas de ojos y boca
        ojo_izq = puntos[lStart:lEnd]
        ojo_der = puntos[rStart:rEnd]
        boca = puntos[mStart:mEnd]

        # Calcular EAR y MAR actuales
        ear_izq = calcular_ear(ojo_izq)
        ear_der = calcular_ear(ojo_der)
        ear_actual = (ear_izq + ear_der) / 2.0
        mar_actual = calcular_mar(boca)

        # Obtener bounding box para poner los textos
        (x, y, w, h) = (rostro.left(), rostro.top(), rostro.width(), rostro.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ---------------------------------------------------------
        # Lógica de Estados
        # ---------------------------------------------------------
        if estado == "CALIBRACION":
            cv2.putText(frame, "Mantener ojos abiertos y boca cerrada para calibracion", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            tiempo_transcurrido = time.time() - tiempo_inicio_calibracion
            
            if tiempo_transcurrido <= 5.0:
                # Grabar valores durante 5 segundos
                datos_ear_calibracion.append(ear_actual)
                datos_mar_calibracion.append(mar_actual)
                cv2.putText(frame, f"Calibrando... {5 - int(tiempo_transcurrido)}s", 
                            (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Tras acabar los 5 segundos, obtener promedio
                treshold_inicial_ear = np.mean(datos_ear_calibracion)
                promedio_mar = np.mean(datos_mar_calibracion) # Se guarda por si se requiere después
                
                estado = "MONITOREO"
                print(f"Calibración exitosa. Threshold EAR: {treshold_inicial_ear:.3f}")

        elif estado == "MONITOREO":
            # 1. Aplicar Median Filter para evitar cambios repentinos
            buffer_crudo_ear.append(ear_actual)
            
            if len(buffer_crudo_ear) == buffer_crudo_ear.maxlen:
                ear_mediano = np.median(buffer_crudo_ear)
                
                # 2. Aplicar Moving Average Filter al resultado del median
                buffer_mediano_ear.append(ear_mediano)
                
                if len(buffer_mediano_ear) == buffer_mediano_ear.maxlen:
                    # Registrar dato pasado por ambos filtros como threshold adaptativo
                    # Multiplicamos por la sensibilidad para permitir parpadeos normales
                    treshold_adaptativo = np.mean(buffer_mediano_ear) * sensibilidad

            # Calcular el threshold estático con sensibilidad aplicada
            treshold_estatico_efectivo = treshold_inicial_ear * sensibilidad
            
            # 3. Función de Alarma (Compara contra ambos thresholds)
            alarma_activa = False
            if treshold_adaptativo > 0: # Solo si los buffers ya se llenaron
                if ear_actual < treshold_estatico_efectivo or ear_actual < treshold_adaptativo:
                    alarma_activa = True

            # Interfaz de usuario para mostrar advertencia y alarma
            if alarma_activa:
                cv2.putText(frame, "ALARMA: OJOS CERRADOS", (x, y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            # Debug visual en pantalla
            cv2.putText(frame, f"EAR: {ear_actual:.2f} | T_Fijo: {treshold_estatico_efectivo:.2f} | T_Adapt: {treshold_adaptativo:.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif estado == "ESPERA":
            cv2.putText(frame, "Presiona 'c' para iniciar calibracion", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Detector de Fatiga", frame)

    # Controles de teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and estado == "ESPERA":
        estado = "CALIBRACION"
        tiempo_inicio_calibracion = time.time()
        datos_ear_calibracion.clear()
        datos_mar_calibracion.clear()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()