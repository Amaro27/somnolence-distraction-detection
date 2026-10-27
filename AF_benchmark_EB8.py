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
# Configuración y Rutas
# ---------------------------------------------------------
VIDEO_PATH = "eyeblink8/1/26122013_223310_cam.avi"
TAG_PATH = "eyeblink8/1/26122013_223310_cam.tag"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(rStart, rEnd) = (36, 42)
(lStart, lEnd) = (42, 48)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0

gt_blinks, gt_estados_frame = cargar_ground_truth_tag(TAG_PATH, fps)

# ---------------------------------------------------------
# Parámetros Fisiológicos Calibrados
# ---------------------------------------------------------
ear_base_abierto = 0.2981
PISO_FISIOLOGICO_DLIB = 0.1525

FACTOR_DESCENSO = 0.70
SENSIBILIDAD_CIERRE = 0.80
DURACION_MIN_FRAMES = 3
COOLDOWN_FRAMES = 2

rango_inicial = max(0.08, ear_base_abierto - PISO_FISIOLOGICO_DLIB)
threshold_adaptativo_cierre = PISO_FISIOLOGICO_DLIB + ((1.0 - SENSIBILIDAD_CIERRE) * rango_inicial)
threshold_adaptativo_descenso = PISO_FISIOLOGICO_DLIB + (FACTOR_DESCENSO * rango_inicial)
threshold_estatico_cierre = threshold_adaptativo_cierre

buffer_filtro_mediana = deque(maxlen=3)
buffer_ojos_abiertos = deque(maxlen=int(4.0 * fps))

pred_blinks = []
en_ciclo_parpadeo = False
frame_inicio_parpadeo = 0
alcanzo_cierre_efectivo = False
ultimo_frame_fin = -COOLDOWN_FRAMES
ultimo_ear_valido = ear_base_abierto

# PERCLOS (Ventana 30s)
VENTANA_FRAMES = int(30.0 * fps)
buffer_ojos_pred = deque(maxlen=VENTANA_FRAMES)
buffer_ojos_gt = deque(maxlen=VENTANA_FRAMES)

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

    for rostro in rostros:
        rostro_detectado = True
        shape = predictor(gray, rostro)
        puntos = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            puntos[i] = (shape.part(i).x, shape.part(i).y)

        ojo_der = puntos[rStart:rEnd]
        ojo_izq = puntos[lStart:lEnd]

        ear_der = calcular_ear(ojo_der)
        ear_izq = calcular_ear(ojo_izq)
        ear_frame = (ear_der + ear_izq) / 2.0
        break

    if rostro_detectado:
        ear_crudo = ear_frame
        ultimo_ear_valido = ear_frame
    else:
        ear_crudo = ultimo_ear_valido

    buffer_filtro_mediana.append(ear_crudo)
    ear_suavizado = float(np.median(buffer_filtro_mediana))

    # ---------------------------------------------------------
    # Actualización Dinámica Protegida
    # ---------------------------------------------------------
    if not en_ciclo_parpadeo and ear_suavizado > threshold_adaptativo_descenso:
        buffer_ojos_abiertos.append(ear_suavizado)
        if len(buffer_ojos_abiertos) >= 15:
            ear_base_abierto = float(np.percentile(buffer_ojos_abiertos, 60))
            rango = max(0.08, ear_base_abierto - PISO_FISIOLOGICO_DLIB)
            threshold_adaptativo_cierre = PISO_FISIOLOGICO_DLIB + ((1.0 - SENSIBILIDAD_CIERRE) * rango)
            threshold_adaptativo_descenso = PISO_FISIOLOGICO_DLIB + (FACTOR_DESCENSO * rango)

    if threshold_adaptativo_cierre < threshold_estatico_cierre:
        threshold_adaptativo_cierre = threshold_estatico_cierre

    # ---------------------------------------------------------
    # Detección de Parpadeos (Límites Exactos sin Padding)
    # ---------------------------------------------------------
    if ear_suavizado < threshold_adaptativo_descenso:
        if not en_ciclo_parpadeo and (frame_idx - ultimo_frame_fin) > COOLDOWN_FRAMES:
            en_ciclo_parpadeo = True
            frame_inicio_parpadeo = frame_idx
            alcanzo_cierre_efectivo = False

        if en_ciclo_parpadeo and ear_suavizado <= threshold_adaptativo_cierre:
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
                ultimo_frame_fin = frame_idx - 1

            en_ciclo_parpadeo = False
            alcanzo_cierre_efectivo = False

    # ---------------------------------------------------------
    # PERCLOS Fisiológico P80/P72
    # ---------------------------------------------------------
    ojo_cerrado_pred = 1 if ear_suavizado <= threshold_adaptativo_cierre else 0
    ojo_cerrado_gt = gt_estados_frame.get(frame_idx, 0)

    buffer_ojos_pred.append(ojo_cerrado_pred)
    buffer_ojos_gt.append(ojo_cerrado_gt)

    perclos_pred = (sum(buffer_ojos_pred) / len(buffer_ojos_pred)) * 100.0 if len(buffer_ojos_pred) > 0 else 0.0
    perclos_gt = (sum(buffer_ojos_gt) / len(buffer_ojos_gt)) * 100.0 if len(buffer_ojos_gt) > 0 else 0.0

    log_perclos_pred.append(perclos_pred)
    log_perclos_gt.append(perclos_gt)

    frame_idx += 1

cap.release()

# ---------------------------------------------------------
# Métricas Comparativas
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
print(" RESULTADOS RESTAURADOS (Eyeblink8 vs Adaptativo)")
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