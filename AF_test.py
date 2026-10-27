import cv2
import dlib
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from collections import deque

# ---------------------------------------------------------
# Funciones matemáticas para EAR y MAR
# ---------------------------------------------------------
def calcular_ear(ojo):
    A = dist.euclidean(ojo[1], ojo[5])
    B = dist.euclidean(ojo[2], ojo[4])
    C = dist.euclidean(ojo[0], ojo[3])
    return (A + B) / (2.0 * C)

def calcular_mar(boca):
    A = dist.euclidean(boca[13], boca[19])
    B = dist.euclidean(boca[14], boca[18])
    C = dist.euclidean(boca[15], boca[17])
    D = dist.euclidean(boca[12], boca[16])
    return (A + B + C) / (2.0 * D)

# ---------------------------------------------------------
# Configuración inicial
# ---------------------------------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)  # Ojo izquierdo
(rStart, rEnd) = (36, 42)  # Ojo derecho
(mStart, mEnd) = (48, 68)  # Boca

cap = cv2.VideoCapture(0)

# ---------------------------------------------------------
# Variables de Estado, Umbrales y Buffers
# ---------------------------------------------------------
estado = "ESPERA"
tiempo_inicio_calibracion = 0
tiempo_inicio_monitoreo = 0.0

datos_ear_abierto = []
datos_ear_cerrado = []

ear_base_abierto = 0.0
ear_base_cerrado = 0.0

# Factores de apertura relativa
FACTOR_DESCENSO = 0.70  # 70% de apertura relativa para iniciar parpadeo
sensibilidad = 0.20     # 20% de apertura restante para confirmar cierre efectivo

threshold_estatico_descenso = 0.0
threshold_adaptativo_descenso = 0.0

threshold_estatico_cierre = 0.0
threshold_adaptativo = 0.0

buffer_crudo_ear = deque(maxlen=5)      # Median Filter
buffer_mediano_ear = deque(maxlen=40)   # Moving Average

# Variables para duración de parpadeo con validación de profundidad
en_ciclo_parpadeo = False
tiempo_inicio_parpadeo = 0.0
alcanzo_cierre_efectivo = False
DURACION_MAX_PARPADEO_SEG = 0.50  # 500 ms

alarma_parpadeo = False
tiempo_ultimo_evento_largo = 0.0
duracion_ultimo_parpadeo_ms = 0.0

# Buffer y umbrales para PERCLOS (30s)
buffer_tiempo_ojos = deque()
VENTANA_TIEMPO_SEG = 30.0
perclos_actual = 0.0
UMBRAL_PERCLOS_FATIGA = 15.0  # 15.0% equivale a 0.15
alarma_perclos = False

# Variables de detección de bostezo
en_bostezo = False
t_inicio_bostezo = 0.0
t_ultimo_bostezo = 0.0
alarma_bostezo = False

DURACION_LIMITE_BOSTEZO = 4.0
UMBRAL_MAR_BOSTEZO = 0.55

contador_ojos_cerrados_30s = 0

# Listas de registro para telemetría
log_ear_crudo = []
log_ear_mediana = []
log_ear_mov_avg = []
log_umbral_descenso = []
log_umbral_adaptativo = []
log_umbral_estatico = []
log_alarma_parpadeo = []
log_alarma_perclos = []
log_alarma_bostezo = []
log_mar = []
log_perclos = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector(gray, 0)

    rostro_detectado = False
    ear_actual = 0.0
    mar_actual = 0.0

    for rostro in rostros:
        rostro_detectado = True
        shape = predictor(gray, rostro)
        puntos = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            puntos[i] = (shape.part(i).x, shape.part(i).y)

        ojo_izq = puntos[lStart:lEnd]
        ojo_der = puntos[rStart:rEnd]
        boca = puntos[mStart:mEnd]

        ear_izq = calcular_ear(ojo_izq)
        ear_der = calcular_ear(ojo_der)
        ear_actual = (ear_izq + ear_der) / 2.0
        mar_actual = calcular_mar(boca)

        (x, y, w, h) = (rostro.left(), rostro.top(), rostro.width(), rostro.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        break

    # ---------------------------------------------------------
    # Máquina de Estados
    # ---------------------------------------------------------
    t_actual = time.time()

    # FASE 1: Calibrar ojos abiertos
    if estado == "CALIB_ABIERTO":
        cv2.putText(frame, "FASE 1: Mantener OJOS ABIERTOS", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        tiempo_transcurrido = t_actual - tiempo_inicio_calibracion
        
        if tiempo_transcurrido <= 4.0:
            if rostro_detectado:
                datos_ear_abierto.append(ear_actual)
            cv2.putText(frame, f"Calibrando abiertos... {4 - int(tiempo_transcurrido)}s", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            ear_base_abierto = np.mean(datos_ear_abierto)
            estado = "AVISO_CERRAR"
            tiempo_inicio_calibracion = t_actual

    # Transición guiada entre fases
    elif estado == "AVISO_CERRAR":
        cv2.putText(frame, "CIERRA LOS OJOS Y MANTENLOS CERRADOS...", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        if (t_actual - tiempo_inicio_calibracion) >= 2.0:
            estado = "CALIB_CERRADO"
            tiempo_inicio_calibracion = t_actual

    # FASE 2: Calibrar ojos cerrados
    elif estado == "CALIB_CERRADO":
        cv2.putText(frame, "FASE 2: Mantener OJOS CERRADOS", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        tiempo_transcurrido = t_actual - tiempo_inicio_calibracion
        
        if tiempo_transcurrido <= 4.0:
            if rostro_detectado:
                datos_ear_cerrado.append(ear_actual)
            cv2.putText(frame, f"Calibrando cerrados... {4 - int(tiempo_transcurrido)}s", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            ear_base_cerrado = np.mean(datos_ear_cerrado)
            
            # Cálculo de umbrales exactos a partir de ambas referencias
            rango_apertura = max(0.05, ear_base_abierto - ear_base_cerrado)
            
            # Umbral base de descenso unificado con FACTOR_DESCENSO
            threshold_estatico_descenso = ear_base_cerrado + (FACTOR_DESCENSO * rango_apertura)
            threshold_adaptativo_descenso = threshold_estatico_descenso
            
            # Umbral de cierre efectivo (20% de apertura restante)
            threshold_estatico_cierre = ear_base_cerrado + ((sensibilidad) * rango_apertura)
            threshold_adaptativo = threshold_estatico_cierre

            # Pre-llenado de buffers con el valor base abierto
            buffer_crudo_ear.clear()
            buffer_mediano_ear.clear()
            for _ in range(buffer_crudo_ear.maxlen):
                buffer_crudo_ear.append(ear_base_abierto)
            for _ in range(buffer_mediano_ear.maxlen):
                buffer_mediano_ear.append(ear_base_abierto)

            estado = "MONITOREO"
            buffer_tiempo_ojos.clear()
            tiempo_inicio_monitoreo = t_actual
            print(f"Calibración exitosa | Abierto: {ear_base_abierto:.3f} | Cerrado: {ear_base_cerrado:.3f}")
            print(f"Descenso inicial: {threshold_estatico_descenso:.3f} | Piso Cierre: {threshold_estatico_cierre:.3f}")

    # MONITOREO ACTIVO
    elif estado == "MONITOREO":
        if rostro_detectado:
            # ---------------------------------------------------------
            # FILTROS Y ACTUALIZACIÓN ADAPTATIVA
            # ---------------------------------------------------------
            if ear_actual >= threshold_adaptativo:
                buffer_crudo_ear.append(ear_actual)
                if len(buffer_crudo_ear) == buffer_crudo_ear.maxlen:
                    ear_mediano = np.median(buffer_crudo_ear)
                    buffer_mediano_ear.append(ear_mediano)
                    if len(buffer_mediano_ear) == buffer_mediano_ear.maxlen:
                        rango_movil = max(0.05, np.mean(buffer_mediano_ear) - ear_base_cerrado)
                        
                        # 1. Cierre adaptativo dinámico
                        threshold_adaptativo = ear_base_cerrado + ((sensibilidad) * rango_movil)
                        
                        # 2. Descenso adaptativo dinámico (unificado con FACTOR_DESCENSO)
                        threshold_adaptativo_descenso = ear_base_cerrado + (FACTOR_DESCENSO * rango_movil)

            val_mediana = np.median(buffer_crudo_ear) if len(buffer_crudo_ear) > 0 else ear_actual
            val_mov_avg = np.mean(buffer_mediano_ear) if len(buffer_mediano_ear) > 0 else ear_actual

            # Piso estático para el umbral de cierre
            if threshold_adaptativo < threshold_estatico_cierre:
                threshold_adaptativo = threshold_estatico_cierre

            # ---------------------------------------------------------
            # DETECCIÓN DE PARPADEO CON CRITERIO DE PROFUNDIDAD
            # ---------------------------------------------------------
            if ear_actual < threshold_adaptativo_descenso:
                if not en_ciclo_parpadeo:
                    en_ciclo_parpadeo = True
                    tiempo_inicio_parpadeo = t_actual
                    alcanzo_cierre_efectivo = False

                if ear_actual <= threshold_adaptativo:
                    alcanzo_cierre_efectivo = True

                duracion_actual = t_actual - tiempo_inicio_parpadeo
                if duracion_actual >= DURACION_MAX_PARPADEO_SEG and alcanzo_cierre_efectivo:
                    alarma_parpadeo = True
                    tiempo_ultimo_evento_largo = t_actual
            else:
                if en_ciclo_parpadeo:
                    duracion_parpadeo = t_actual - tiempo_inicio_parpadeo
                    duracion_ultimo_parpadeo_ms = duracion_parpadeo * 1000.0

                    if duracion_parpadeo >= DURACION_MAX_PARPADEO_SEG and alcanzo_cierre_efectivo:
                        alarma_parpadeo = True
                        tiempo_ultimo_evento_largo = t_actual
                        print(f"[MICROSUEÑO] Parpadeo prolongado: {duracion_ultimo_parpadeo_ms:.1f} ms")

                    en_ciclo_parpadeo = False
                    alcanzo_cierre_efectivo = False

                if (t_actual - tiempo_ultimo_evento_largo) > 1.2:
                    alarma_parpadeo = False

            # ---------------------------------------------------------
            # DETECCIÓN DE BOSTEZO
            # ---------------------------------------------------------
            if mar_actual > UMBRAL_MAR_BOSTEZO:
                if not en_bostezo:
                    en_bostezo = True
                    t_inicio_bostezo = t_actual

                duracion_bostezo = t_actual - t_inicio_bostezo

                if duracion_bostezo >= DURACION_LIMITE_BOSTEZO:
                    alarma_bostezo = True
                    t_ultimo_bostezo = t_actual
            else:
                if en_bostezo:
                    if (t_actual - t_inicio_bostezo) >= DURACION_LIMITE_BOSTEZO:
                        alarma_bostezo = True
                        t_ultimo_bostezo = t_actual
                    en_bostezo = False

                if (t_actual - t_ultimo_bostezo) > 1.5:
                    alarma_bostezo = False

            # ---------------------------------------------------------
            # CÁLCULO Y ALARMA DE PERCLOS (Ventana 30s)
            # ---------------------------------------------------------
            ojo_cerrado_binario = 1 if ear_actual <= threshold_adaptativo else 0
            buffer_tiempo_ojos.append((t_actual, ojo_cerrado_binario))
            while buffer_tiempo_ojos and buffer_tiempo_ojos[0][0] < (t_actual - VENTANA_TIEMPO_SEG):
                buffer_tiempo_ojos.popleft()

            tiempo_en_monitoreo = t_actual - tiempo_inicio_monitoreo
            if tiempo_en_monitoreo >= VENTANA_TIEMPO_SEG:
                if len(buffer_tiempo_ojos) > 0:
                    total_c = sum(val[1] for val in buffer_tiempo_ojos)
                    perclos_actual = (total_c / len(buffer_tiempo_ojos)) * 100.0
                    alarma_perclos = perclos_actual >= UMBRAL_PERCLOS_FATIGA
            else:
                perclos_actual = 0.0
                alarma_perclos = False

            # Guardar logs
            log_ear_crudo.append(ear_actual)
            log_ear_mediana.append(val_mediana)
            log_ear_mov_avg.append(val_mov_avg)
            log_umbral_descenso.append(threshold_adaptativo_descenso)
            log_umbral_adaptativo.append(threshold_adaptativo)
            log_umbral_estatico.append(threshold_estatico_cierre)
            log_alarma_parpadeo.append(1 if alarma_parpadeo else 0)
            log_alarma_perclos.append(1 if alarma_perclos else 0)
            log_alarma_bostezo.append(1 if alarma_bostezo else 0)
            log_mar.append(mar_actual)
            log_perclos.append(perclos_actual)

            # ---------------------------------------------------------
            # Visualización en pantalla (Coordenadas ajustadas)
            # ---------------------------------------------------------
            # Alertas sobre el rostro
            if alarma_parpadeo:
                cv2.putText(frame, "ALARMA: OJOS CERRADOS (>500ms)", (x, y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            if alarma_bostezo:
                cv2.putText(frame, "ALERTA: BOSTEZO PROLONGADO (>4s)", (x, y - 48), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2)

            if alarma_perclos:
                cv2.putText(frame, "ALERTA: SOMNOLENCIA (PERCLOS >= 15%)", (x, y - 76), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)

            # Panel de métricas superiores
            cv2.putText(frame, f"EAR: {ear_actual:.2f} | T_Desc Adapt: {threshold_adaptativo_descenso:.2f} | T_Cierre: {threshold_adaptativo:.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar_actual:.2f} | Ultimo Parpadeo: {duracion_ultimo_parpadeo_ms:.0f} ms", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if tiempo_en_monitoreo < VENTANA_TIEMPO_SEG:
                seg_rest = int(VENTANA_TIEMPO_SEG - tiempo_en_monitoreo)
                cv2.putText(frame, f"PERCLOS (30s): 0.0% [Calentamiento: {seg_rest}s]", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            else:
                color_perclos = (0, 0, 255) if alarma_perclos else (255, 100, 255)
                cv2.putText(frame, f"PERCLOS (30s): {perclos_actual:.1f}%", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_perclos, 2)

        else:
            cv2.putText(frame, "Buscando rostro...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    elif estado == "ESPERA":
        cv2.putText(frame, "Presiona 'c' para iniciar calibracion en 2 fases", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Detector de Fatiga", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and estado == "ESPERA":
        estado = "CALIB_ABIERTO"
        tiempo_inicio_calibracion = time.time()
        datos_ear_abierto.clear()
        datos_ear_cerrado.clear()
        en_ciclo_parpadeo = False
        alcanzo_cierre_efectivo = False
        alarma_parpadeo = False
        alarma_perclos = False
        en_bostezo = False
        alarma_bostezo = False
        t_inicio_bostezo = 0.0
        t_ultimo_bostezo = 0.0
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------
# Gráficas al cerrar
# ---------------------------------------------------------
if len(log_ear_crudo) > 0:
    cuadros = range(len(log_ear_crudo))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Gráfica 1: Filtros EAR
    ax1.plot(cuadros, log_ear_crudo, label='EAR Crudo', color='gray', alpha=0.4, linewidth=1)
    ax1.plot(cuadros, log_ear_mediana, label='Median Filter', color='orange', linewidth=1.5)
    ax1.plot(cuadros, log_ear_mov_avg, label='Moving Average', color='blue', linewidth=2)
    ax1.set_title('Filtrado de Señal EAR')
    ax1.set_ylabel('EAR')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Gráfica 2: Disparador por Doble Umbral Adaptativo
    ax2.plot(cuadros, log_ear_crudo, label='EAR Actual', color='black', alpha=0.5, linewidth=1)
    ax2.plot(cuadros, log_umbral_descenso, label='T_Descenso Adaptativo', color='gold', linewidth=1.8)
    ax2.plot(cuadros, log_umbral_adaptativo, label='T_Cierre Adaptativo (20%)', color='purple', linewidth=1.8)
    ax2.plot(cuadros, log_umbral_estatico, label='T_Cierre Piso Estático', color='green', linestyle=':', linewidth=1.5)
    
    alarma_p_arr = np.array(log_alarma_parpadeo)
    ax2.fill_between(cuadros, 0, max(log_ear_crudo) + 0.05, where=(alarma_p_arr == 1), color='red', alpha=0.25, label='Alarma Parpadeo (>500ms)')
    ax2.set_title('Histéresis Adaptativa y Detección de Microsueños')
    ax2.set_ylabel('EAR')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Gráfica 3: MAR con Sombreado de Bostezos
    ax3.plot(cuadros, log_mar, label='MAR (Boca)', color='teal', linewidth=1.8)
    ax3.axhline(y=UMBRAL_MAR_BOSTEZO, color='darkorange', linestyle='--', label=f'Umbral Bostezo ({UMBRAL_MAR_BOSTEZO})')
    
    alarma_bost_arr = np.array(log_alarma_bostezo)
    ax3.fill_between(cuadros, 0, max(log_mar) + 0.1 if log_mar else 1.0, where=(alarma_bost_arr == 1), color='orange', alpha=0.3, label='Alerta Bostezo (>4s)')
    ax3.set_title('Apertura Bucal (MAR) y Detección de Bostezos')
    ax3.set_ylabel('MAR')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Gráfica 4: PERCLOS y Alarma de Somnolencia
    ax4.plot(cuadros, log_perclos, label='% Cierre (30s móvil)', color='magenta', linewidth=1.8)
    ax4.axhline(y=UMBRAL_PERCLOS_FATIGA, color='red', linestyle='--', linewidth=1.5, label=f'Umbral Somnolencia ({UMBRAL_PERCLOS_FATIGA:.0f}%)')
    
    alarma_perc_arr = np.array(log_alarma_perclos)
    ax4.fill_between(cuadros, 0, 100, where=(alarma_perc_arr == 1), color='red', alpha=0.2, label='Zona de Somnolencia Activa')
    ax4.set_title('PERCLOS (Ventana 30s) y Detección de Somnolencia Acumulada')
    ax4.set_xlabel('Fotogramas Monitoreados')
    ax4.set_ylabel('% Tiempo')
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()