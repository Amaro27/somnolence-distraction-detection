import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from collections import deque

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

def cargar_ground_truth_tag(tag_path, fps):
    gt_blinks = []
    current_blink = []
    gt_estados_frame = {}

    with open(tag_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            
            fid = int(parts[0])
            state = int(parts[1])
            is_closed = 1 if state != -1 else 0
            gt_estados_frame[fid] = is_closed

            if is_closed == 1:
                current_blink.append(fid)
            else:
                if current_blink:
                    gt_blinks.append({
                        'start': current_blink[0],
                        'end': current_blink[-1],
                        'frames': len(current_blink),
                        'duration_ms': (len(current_blink) / fps) * 1000.0
                    })
                    current_blink = []

        if current_blink:
            gt_blinks.append({
                'start': current_blink[0],
                'end': current_blink[-1],
                'frames': len(current_blink),
                'duration_ms': (len(current_blink) / fps) * 1000.0
            })

    return gt_blinks, gt_estados_frame

# ---------------------------------------------------------
# Rutas y Modelos
# ---------------------------------------------------------
VIDEO_PATH = "eyeblink8/1/26122013_223310_cam.avi"
TAG_PATH = "eyeblink8/1/26122013_223310_cam.tag"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(rStart, rEnd) = (36, 42)
(lStart, lEnd) = (42, 48)
(mStart, mEnd) = (48, 68)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0

gt_blinks, gt_estados_frame = cargar_ground_truth_tag(TAG_PATH, fps)

# ---------------------------------------------------------
# Parámetros Fisiológicos y Adaptativos
# ---------------------------------------------------------
ear_base_abierto = 0.2981
PISO_FISIOLOGICO_DLIB = 0.1525

FACTOR_DESCENSO = 0.70
SENSIBILIDAD_CIERRE = 0.72
DURACION_MIN_FRAMES = 3

rango_inicial = max(0.08, ear_base_abierto - PISO_FISIOLOGICO_DLIB)
threshold_adaptativo_cierre = PISO_FISIOLOGICO_DLIB + ((1.0 - SENSIBILIDAD_CIERRE) * rango_inicial)
threshold_adaptativo_descenso = PISO_FISIOLOGICO_DLIB + (FACTOR_DESCENSO * rango_inicial)
threshold_estatico_cierre = threshold_adaptativo_cierre

# Buffer adaptativo para la línea base de ojos abiertos
buffer_ojos_abiertos = deque(maxlen=int(4.0 * fps))

pred_blinks = []
en_ciclo_parpadeo = False
frame_inicio_parpadeo = 0
alcanzo_cierre_efectivo = False
ultimo_ear_valido = ear_base_abierto

# PERCLOS
VENTANA_FRAMES = int(30.0 * fps)
buffer_ojos_pred = deque(maxlen=VENTANA_FRAMES)
buffer_ojos_gt = deque(maxlen=VENTANA_FRAMES)

log_frame = []
log_ear_crudo = []
log_umbral_descenso = []
log_umbral_cierre = []
log_mar = []
log_pred_binario = []
log_gt_binario = []
log_perclos_pred = []
log_perclos_gt = []

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector(gray, 0)

    rostro_detectado = False
    ear_frame = 0.0
    mar_actual = 0.0

    for rostro in rostros:
        rostro_detectado = True
        shape = predictor(gray, rostro)
        puntos = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            puntos[i] = (shape.part(i).x, shape.part(i).y)

        ojo_der = puntos[rStart:rEnd]
        ojo_izq = puntos[lStart:lEnd]
        boca = puntos[mStart:mEnd]

        ear_der = calcular_ear(ojo_der)
        ear_izq = calcular_ear(ojo_izq)
        ear_frame = (ear_der + ear_izq) / 2.0
        mar_actual = calcular_mar(boca)
        break

    if rostro_detectado:
        ear_crudo = ear_frame
        ultimo_ear_valido = ear_frame
    else:
        ear_crudo = ultimo_ear_valido

    # ---------------------------------------------------------
    # Actualización Dinámica Protegida (Filtro Adaptativo)
    # ---------------------------------------------------------
    if not en_ciclo_parpadeo and ear_crudo > threshold_adaptativo_descenso:
        buffer_ojos_abiertos.append(ear_crudo)
        if len(buffer_ojos_abiertos) >= 15:
            # Percentil 60 para mantener la línea base sin afectarse por sacadas
            ear_base_abierto = float(np.percentile(buffer_ojos_abiertos, 60))
            rango = max(0.08, ear_base_abierto - PISO_FISIOLOGICO_DLIB)
            threshold_adaptativo_cierre = PISO_FISIOLOGICO_DLIB + ((1.0 - SENSIBILIDAD_CIERRE) * rango)
            threshold_adaptativo_descenso = PISO_FISIOLOGICO_DLIB + (FACTOR_DESCENSO * rango)

    if threshold_adaptativo_cierre < threshold_estatico_cierre:
        threshold_adaptativo_cierre = threshold_estatico_cierre

    # ---------------------------------------------------------
    # Detección de Parpadeos (Evaluando EAR en crudo)
    # ---------------------------------------------------------
    if ear_crudo < threshold_adaptativo_descenso:
        if not en_ciclo_parpadeo:
            en_ciclo_parpadeo = True
            frame_inicio_parpadeo = frame_idx
            alcanzo_cierre_efectivo = False

        if ear_crudo <= threshold_adaptativo_cierre:
            alcanzo_cierre_efectivo = True
    else:
        if en_ciclo_parpadeo:
            dur_f = frame_idx - frame_inicio_parpadeo
            if alcanzo_cierre_efectivo and dur_f >= DURACION_MIN_FRAMES:
                pred_blinks.append({
                    'start': frame_inicio_parpadeo,
                    'end': frame_idx - 1,
                    'frames': dur_f,
                    'duration_ms': (dur_f / fps) * 1000.0
                })
            en_ciclo_parpadeo = False
            alcanzo_cierre_efectivo = False

    # ---------------------------------------------------------
    # PERCLOS Frame a Frame
    # ---------------------------------------------------------
    ojo_cerrado_pred = 1 if ear_crudo <= threshold_adaptativo_cierre else 0
    ojo_cerrado_gt = gt_estados_frame.get(frame_idx, 0)

    buffer_ojos_pred.append(ojo_cerrado_pred)
    buffer_ojos_gt.append(ojo_cerrado_gt)

    perclos_pred = (sum(buffer_ojos_pred) / len(buffer_ojos_pred)) * 100.0 if len(buffer_ojos_pred) > 0 else 0.0
    perclos_gt = (sum(buffer_ojos_gt) / len(buffer_ojos_gt)) * 100.0 if len(buffer_ojos_gt) > 0 else 0.0

    log_frame.append(frame_idx)
    log_ear_crudo.append(ear_crudo)
    log_umbral_descenso.append(threshold_adaptativo_descenso)
    log_umbral_cierre.append(threshold_adaptativo_cierre)
    log_mar.append(mar_actual)
    log_pred_binario.append(ojo_cerrado_pred)
    log_gt_binario.append(ojo_cerrado_gt)
    log_perclos_pred.append(perclos_pred)
    log_perclos_gt.append(perclos_gt)

    frame_idx += 1

cap.release()

# ---------------------------------------------------------
# Cálculo de Métricas Comparativas
# ---------------------------------------------------------
IOU_THRESHOLD = 0.2
matched_gt = set()
tp = 0
fp = 0
errores_duracion_ms = []

for pred in pred_blinks:
    p_set = set(range(pred['start'], pred['end'] + 1))
    mejor_iou = 0.0
    mejor_gt_idx = -1

    for idx, gt in enumerate(gt_blinks):
        if idx in matched_gt:
            continue
        gt_set = set(range(gt['start'], gt['end'] + 1))
        inter = len(p_set.intersection(gt_set))
        union = len(p_set.union(gt_set))

        if union > 0:
            iou = inter / union
            if iou > mejor_iou:
                mejor_iou = iou
                mejor_gt_idx = idx

    if mejor_iou >= IOU_THRESHOLD:
        tp += 1
        matched_gt.add(mejor_gt_idx)
        err = abs(pred['duration_ms'] - gt_blinks[mejor_gt_idx]['duration_ms'])
        errores_duracion_ms.append(err)
    else:
        fp += 1

fn = len(gt_blinks) - len(matched_gt)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
mae_duracion = np.mean(errores_duracion_ms) if errores_duracion_ms else 0.0
mae_perclos = np.mean(np.abs(np.array(log_perclos_pred) - np.array(log_perclos_gt)))

print("\n" + "="*50)
print(" RESULTADOS DEL BENCHMARK (Eyeblink8 vs Adaptativo)")
print("="*50)
print(f"Parpadeos Reales (Ground Truth): {len(gt_blinks)}")
print(f"Parpadeos Detectados:           {len(pred_blinks)}")
print(f"Verdaderos Positivos (TP):       {tp}")
print(f"Falsos Positivos (FP):          {fp}")
print(f"Falsos Negativos (FN):          {fn}")
print(f"Precisión:                      {precision:.4f}")
print(f"Recall (Sensibilidad):          {recall:.4f}")
print(f"F1-Score:                       {f1:.4f}")
print(f"MAE Duración de Parpadeo:       {mae_duracion:.2f} ms")
print(f"MAE PERCLOS (Ventana 30s):      {mae_perclos:.2f} %")
print("="*50 + "\n")