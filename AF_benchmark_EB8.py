import cv2
import dlib
import os
import pandas as pd
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
            
            # EVALUACIÓN ESTRICTA: Solo es "cierre" si el anotador marcó 'C' en los atributos
            is_closed = 0
            if len(parts) >= 7:
                if 'C' in parts[2:7]:
                    is_closed = 1
            
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

def extraer_falsos_positivos(video_path, df_eventos, output_dir="eventos_parpadeos/fps_clips1"):
    """
    Recorta y guarda el video de los eventos clasificados como Falsos Positivos.
    Añade un padding de 10 frames antes y después para ver el contexto del error.
    """
    fps_eventos = df_eventos[df_eventos['Tipo'] == 'Falso Positivo']
    
    if len(fps_eventos) == 0:
        print("No se encontraron Falsos Positivos para extraer clips.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video para extracción: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    print(f"\nExtrayendo {len(fps_eventos)} clips de Falsos Positivos...")
    
    for index, row in fps_eventos.iterrows():
        # Padding de 10 frames para entender qué provocó el error (ej. sacada hacia abajo)
        clip_start = max(0, int(row['Pred_Frame_Inicio']) - 10)
        clip_end = min(total_frames - 1, int(row['Pred_Frame_Fin']) + 10)
        
        # Guardar con el ID de la predicción y su duración
        dur = int(row['Pred_Duracion_ms'])
        out_filename = os.path.join(output_dir, f"FP_PredID_{int(row['Pred_ID']):03d}_Dur_{dur}ms.avi")
        out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        
        for f_idx in range(clip_start, clip_end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Dibujar un texto rojo para indicar visualmente que es un FP
            cv2.putText(frame, f"Falso Positivo - Pred {int(row['Pred_ID'])}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
            out.write(frame)
            
        out.release()
        print(f" Guardado: {out_filename}")
    
    cap.release()
    print(f"Extracción de clips completada en: {output_dir}/")

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

# Al medir solo la fase de oclusión profunda ('C'), el evento es más corto en frames
DURACION_MIN_FRAMES = 2
COOLDOWN_FRAMES = 2

rango_inicial = max(0.08, ear_base_abierto - PISO_FISIOLOGICO_DLIB)
threshold_adaptativo_cierre = PISO_FISIOLOGICO_DLIB + ((1.0 - SENSIBILIDAD_CIERRE) * rango_inicial)
threshold_adaptativo_descenso = PISO_FISIOLOGICO_DLIB + (FACTOR_DESCENSO * rango_inicial)
threshold_estatico_cierre = threshold_adaptativo_cierre

# Filtros en cascada
buffer_crudo_ear = deque(maxlen=5)
buffer_mediano_ear = deque(maxlen=int(3.0 * fps))

pred_blinks = []
en_ciclo_parpadeo = False
cierre_efectivo = False
frame_inicio_cierre = -1
frame_fin_cierre = -1
ultimo_frame_fin = -COOLDOWN_FRAMES
ultimo_ear_valido = ear_base_abierto

# PERCLOS
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

    # ---------------------------------------------------------
    # Actualización Dinámica con Doble Filtro en Cascada
    # ---------------------------------------------------------
    if not en_ciclo_parpadeo and ear_crudo > threshold_adaptativo_descenso:
        buffer_crudo_ear.append(ear_crudo)
        
        if len(buffer_crudo_ear) == buffer_crudo_ear.maxlen:
            ear_mediano = float(np.median(buffer_crudo_ear))
            buffer_mediano_ear.append(ear_mediano)
            
            if len(buffer_mediano_ear) >= 15:
                ear_base_abierto = float(np.mean(buffer_mediano_ear))
                rango_movil = max(0.08, ear_base_abierto - PISO_FISIOLOGICO_DLIB)
                
                threshold_adaptativo_cierre = PISO_FISIOLOGICO_DLIB + ((1.0 - SENSIBILIDAD_CIERRE) * rango_movil)
                threshold_adaptativo_descenso = PISO_FISIOLOGICO_DLIB + (FACTOR_DESCENSO * rango_movil)

    if threshold_adaptativo_cierre < threshold_estatico_cierre:
        threshold_adaptativo_cierre = threshold_estatico_cierre

    # ---------------------------------------------------------
    # Detección y Cronometraje de Cierre Profundo ('C')
    # ---------------------------------------------------------
    if ear_crudo < threshold_adaptativo_descenso:
        # 1. Inicia la ventana temporal del parpadeo (sin registrarlo aún)
        if not en_ciclo_parpadeo and (frame_idx - ultimo_frame_fin) > COOLDOWN_FRAMES:
            en_ciclo_parpadeo = True
            cierre_efectivo = False
            frame_inicio_cierre = -1
            frame_fin_cierre = -1

        if en_ciclo_parpadeo:
            if ear_crudo <= threshold_adaptativo_cierre:
                if not cierre_efectivo:
                    # 2. Inicia el cronometraje exacto de la oclusión profunda
                    cierre_efectivo = True
                    frame_inicio_cierre = frame_idx
                # 3. Empuja el fotograma final mientras el ojo siga cerrado
                frame_fin_cierre = frame_idx
    else:
        if en_ciclo_parpadeo:
            # 4. El ojo recuperó la apertura; se consolida el parpadeo
            if cierre_efectivo:
                dur_f = frame_fin_cierre - frame_inicio_cierre + 1
                if dur_f >= DURACION_MIN_FRAMES:
                    pred_blinks.append({
                        'start': frame_inicio_cierre,
                        'end': frame_fin_cierre,
                        'frames': dur_f,
                        'duration_ms': (dur_f / fps) * 1000.0
                    })
                    ultimo_frame_fin = frame_idx - 1

            en_ciclo_parpadeo = False
            cierre_efectivo = False

    # ---------------------------------------------------------
    # PERCLOS Fisiológico Estricto
    # ---------------------------------------------------------
    # Ahora ambos arreglos miden estrictamente la fase de oclusión 'C'
    ojo_cerrado_pred = 1 if ear_crudo <= threshold_adaptativo_cierre else 0
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

# Lista para guardar los datos detallados que irán al archivo CSV
datos_exportacion = []

for pred_idx, pred in enumerate(pred_blinks):
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

    # Construir el registro del evento predicho
    registro = {
        'Pred_ID': pred_idx + 1,
        'Pred_Frame_Inicio': pred['start'],
        'Pred_Frame_Fin': pred['end'],
        'Pred_Duracion_ms': round(pred['duration_ms'], 2),
        'Tipo': 'Falso Positivo', # Por defecto
        'GT_ID': None,
        'GT_Frame_Inicio': None,
        'GT_Frame_Fin': None,
        'GT_Duracion_ms': None,
        'Error_Duracion_ms': None
    }

    if mejor_iou >= IOU_THRESHOLD:
        tp += 1
        matched_gt.add(mejor_gt_idx)
        
        gt_blink = gt_blinks[mejor_gt_idx]
        err = abs(pred['duration_ms'] - gt_blink['duration_ms'])
        errores_duracion_ms.append(err)
        
        # Actualizar el registro si fue un Verdadero Positivo
        registro['Tipo'] = 'Verdadero Positivo'
        registro['GT_ID'] = mejor_gt_idx + 1
        registro['GT_Frame_Inicio'] = gt_blink['start']
        registro['GT_Frame_Fin'] = gt_blink['end']
        registro['GT_Duracion_ms'] = round(gt_blink['duration_ms'], 2)
        registro['Error_Duracion_ms'] = round(err, 2)
    else:
        fp += 1
        
    datos_exportacion.append(registro)

# Añadir los Falsos Negativos (parpadeos reales que el modelo no vio) al final de la tabla
fn = 0
for idx, gt in enumerate(gt_blinks):
    if idx not in matched_gt:
        fn += 1
        datos_exportacion.append({
            'Pred_ID': None,
            'Pred_Frame_Inicio': None,
            'Pred_Frame_Fin': None,
            'Pred_Duracion_ms': None,
            'Tipo': 'Falso Negativo',
            'GT_ID': idx + 1,
            'GT_Frame_Inicio': gt['start'],
            'GT_Frame_Fin': gt['end'],
            'GT_Duracion_ms': round(gt['duration_ms'], 2),
            'Error_Duracion_ms': None
        })

# Cálculo de métricas globales
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
mae_duracion = np.mean(errores_duracion_ms) if errores_duracion_ms else 0.0
mae_perclos = np.mean(np.abs(np.array(log_perclos_pred) - np.array(log_perclos_gt)))

# Exportar datos a CSV
df_eventos = pd.DataFrame(datos_exportacion)
archivo_salida = "eventos_parpadeos/eventos_parpadeos_benchmark1.csv"
os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
df_eventos.to_csv(archivo_salida, index=False, encoding='utf-8-sig')

# Extraer clips de los Falsos Positivos
extraer_falsos_positivos(VIDEO_PATH, df_eventos)

print("\n" + "="*50)
print(" RESULTADOS BENCHMARK (Alineado con 'C')")
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
print("="*50)
print(f"Datos exportados a: {archivo_salida}")
print(f"Clips de FP exportados a: eventos_parpadeos/fps_clips/")