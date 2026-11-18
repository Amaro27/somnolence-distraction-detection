import argparse
import os
from collections import deque

import cv2
import dlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

# ---------------------------------------------------------
# Configuración de los 8 Videos de Eyeblink8 y su Calibración Individual
# ---------------------------------------------------------
VIDEOS_EB8 = {
    1: {
        "video_path": "eyeblink8/1/26122013_223310_cam.avi",
        "tag_path": "eyeblink8/1/26122013_223310_cam.tag",
        "ear_base_abierto": 0.2981,
        "ear_base_cerrado": 0.1525,
    },
    2: {
        "video_path": "eyeblink8/2/26122013_224532_cam.avi",
        "tag_path": "eyeblink8/2/26122013_224532_cam.tag",
        "ear_base_abierto": 0.2855,
        "ear_base_cerrado": 0.1276,
    },
    3: {
        "video_path": "eyeblink8/3/26122013_230103_cam.avi",
        "tag_path": "eyeblink8/3/26122013_230103_cam.tag",
        "ear_base_abierto": 0.3459,
        "ear_base_cerrado": 0.1865,
    },
    4: {
        "video_path": "eyeblink8/4/26122013_230654_cam.avi",
        "tag_path": "eyeblink8/4/26122013_230654_cam.tag",
        "ear_base_abierto": 0.3618,
        "ear_base_cerrado": 0.1852,
    },
    8: {
        "video_path": "eyeblink8/8/27122013_151644_cam.avi",
        "tag_path": "eyeblink8/8/27122013_151644_cam.tag",
        "ear_base_abierto": 0.2792,
        "ear_base_cerrado": 0.1378,
    },
    9: {
        "video_path": "eyeblink8/9/27122013_152435_cam.avi",
        "tag_path": "eyeblink8/9/27122013_152435_cam.tag",
        "ear_base_abierto": 0.2753,
        "ear_base_cerrado": 0.1296,
    },
    10: {
        "video_path": "eyeblink8/10/27122013_153916_cam.avi",
        "tag_path": "eyeblink8/10/27122013_153916_cam.tag",
        "ear_base_abierto": 0.2507,
        "ear_base_cerrado": 0.1208,
    },
    11: {
        "video_path": "eyeblink8/11/27122013_154548_cam.avi",
        "tag_path": "eyeblink8/11/27122013_154548_cam.tag",
        "ear_base_abierto": 0.2368,
        "ear_base_cerrado": 0.1201,
    },
}

# Edge cases designados como (Video_ID, GT_ID) ("n-m") a excluir de la evaluación
EDGE_CASES_GT = [
    (1, 11),
    (1, 12),
    (2, 49),
    (2, 50),
    (2, 51),
    (2, 52),
    (2, 53),
    (11, 14),
    (11, 22),
    (11, 33),
    (11, 34),
]

# Parámetros del algoritmo adaptativo y emparejamiento
FACTOR_DESCENSO = 0.70
SENSIBILIDAD_CIERRE = 0.80
DURACION_MIN_FRAMES = 2
COOLDOWN_FRAMES = 2
IOU_THRESHOLD = 0.20

# Intervalos de duración mínima en pasos de 100 ms (desde 100 ms hacia arriba)
UMBRALES_DURACION_100MS = [100, 200, 300, 400, 500, 600, 700, 800]

(R_START, R_END) = (36, 42)
(L_START, L_END) = (42, 48)


def calcular_ear(ojo):
    A = dist.euclidean(ojo[1], ojo[5])
    B = dist.euclidean(ojo[2], ojo[4])
    C = dist.euclidean(ojo[0], ojo[3])
    return (A + B) / (2.0 * C)


def cargar_ground_truth_tag(tag_path, fps):
    gt_blinks = []
    current_blink = []
    gt_estados_frame = {}

    with open(tag_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) < 2 or not parts[0].isdigit():
                continue

            fid = int(parts[0])

            # EVALUACIÓN ESTRICTA: Solo es "cierre" si el anotador marcó 'C' en los atributos
            is_closed = 0
            if len(parts) >= 7:
                if "C" in parts[2:7]:
                    is_closed = 1

            gt_estados_frame[fid] = is_closed

            if is_closed == 1:
                current_blink.append(fid)
            else:
                if current_blink:
                    gt_blinks.append({
                        "start": current_blink[0],
                        "end": current_blink[-1],
                        "frames": len(current_blink),
                        "duration_ms": (len(current_blink) / fps) * 1000.0,
                    })
                    current_blink = []

        if current_blink:
            gt_blinks.append({
                "start": current_blink[0],
                "end": current_blink[-1],
                "frames": len(current_blink),
                "duration_ms": (len(current_blink) / fps) * 1000.0,
            })

    return gt_blinks, gt_estados_frame


def extraer_falsos_positivos(video_path, df_eventos, output_dir="eventos_parpadeos/fps_clips1"):
    """
    Recorta y guarda el video de los eventos clasificados como Falsos Positivos.
    Añade un padding de 10 frames antes y después para ver el contexto del error.
    """
    fps_eventos = df_eventos[df_eventos["Tipo"] == "Falso Positivo"]

    if len(fps_eventos) == 0:
        print("No se encontraron Falsos Positivos para extraer clips.")
        return

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video para extracción: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    print(f"\nExtrayendo {len(fps_eventos)} clips de Falsos Positivos en {output_dir}...")

    for _, row in fps_eventos.iterrows():
        clip_start = max(0, int(row["Pred_Frame_Inicio"]) - 10)
        clip_end = min(total_frames - 1, int(row["Pred_Frame_Fin"]) + 10)

        dur = int(row["Pred_Duracion_ms"])
        out_filename = os.path.join(
            output_dir, f"FP_PredID_{int(row['Pred_ID']):03d}_Dur_{dur}ms.avi"
        )
        out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)

        for _ in range(clip_start, clip_end + 1):
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(
                frame,
                f"Falso Positivo - Pred {int(row['Pred_ID'])}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            out.write(frame)

        out.release()

    cap.release()
    print(f"Extracción de clips completada en: {output_dir}/")


def calcular_mae_perclos_desde_eventos(df_eventos, gt_estados_frame, total_frames, fps):
    """Calcula el MAE de PERCLOS (ventana móvil de 30s) a partir de los eventos y etiquetas GT."""
    ventana_frames = int(30.0 * fps)
    buffer_ojos_pred = deque(maxlen=ventana_frames)
    buffer_ojos_gt = deque(maxlen=ventana_frames)

    frames_cerrados_pred = set()
    preds = df_eventos[df_eventos["Tipo"].isin(["Verdadero Positivo", "Falso Positivo"])]
    for _, row in preds.iterrows():
        if pd.notna(row["Pred_Frame_Inicio"]) and pd.notna(row["Pred_Frame_Fin"]):
            for f_idx in range(int(row["Pred_Frame_Inicio"]), int(row["Pred_Frame_Fin"]) + 1):
                frames_cerrados_pred.add(f_idx)

    log_perclos_pred = []
    log_perclos_gt = []
    n_frames = max(total_frames, (max(gt_estados_frame.keys()) + 1) if gt_estados_frame else 0)

    for frame_idx in range(n_frames):
        ojo_cerrado_pred = 1 if frame_idx in frames_cerrados_pred else 0
        ojo_cerrado_gt = gt_estados_frame.get(frame_idx, 0)

        buffer_ojos_pred.append(ojo_cerrado_pred)
        buffer_ojos_gt.append(ojo_cerrado_gt)

        perclos_pred = (sum(buffer_ojos_pred) / len(buffer_ojos_pred)) * 100.0
        perclos_gt = (sum(buffer_ojos_gt) / len(buffer_ojos_gt)) * 100.0

        log_perclos_pred.append(perclos_pred)
        log_perclos_gt.append(perclos_gt)

    if not log_perclos_pred:
        return 0.0
    return float(np.mean(np.abs(np.array(log_perclos_pred) - np.array(log_perclos_gt))))


def procesar_video_eb8(
    video_id,
    cfg,
    detector=None,
    predictor=None,
    usar_cache_csv=True,
    extraer_clips=False,
    output_dir="eventos_parpadeos",
):
    """
    Ejecuta el benchmark sobre un video de Eyeblink8 con sus parámetros de calibración
    (o carga el CSV ya generado en `eventos_parpadeos/eventos_parpadeos_benchmark{id}.csv`
    si `usar_cache_csv=True`).
    """
    video_path = cfg["video_path"]
    tag_path = cfg["tag_path"]
    ear_abierto_init = cfg["ear_base_abierto"]
    piso_fisiologico = cfg["ear_base_cerrado"]

    archivo_salida = os.path.join(output_dir, f"eventos_parpadeos_benchmark{video_id}.csv")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    gt_blinks, gt_estados_frame = cargar_ground_truth_tag(tag_path, fps)

    if usar_cache_csv and os.path.exists(archivo_salida):
        cap.release()
        df_eventos = pd.read_csv(archivo_salida)
        mae_perclos = calcular_mae_perclos_desde_eventos(
            df_eventos, gt_estados_frame, total_frames, fps
        )
        if extraer_clips:
            extraer_falsos_positivos(
                video_path, df_eventos, output_dir=os.path.join(output_dir, f"fps_clips{video_id}")
            )
        return df_eventos, gt_blinks, mae_perclos, fps

    if detector is None:
        detector = dlib.get_frontal_face_detector()
    if predictor is None:
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    ear_base_abierto = ear_abierto_init
    rango_inicial = max(0.08, ear_base_abierto - piso_fisiologico)
    threshold_adaptativo_cierre = piso_fisiologico + ((1.0 - SENSIBILIDAD_CIERRE) * rango_inicial)
    threshold_adaptativo_descenso = piso_fisiologico + (FACTOR_DESCENSO * rango_inicial)
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
    ventana_frames = int(30.0 * fps)
    buffer_ojos_pred = deque(maxlen=ventana_frames)
    buffer_ojos_gt = deque(maxlen=ventana_frames)

    log_perclos_pred = []
    log_perclos_gt = []

    frame_idx = 0
    print(f"[Video {video_id}] Procesando {video_path} ({total_frames} frames a {fps:.1f} fps)...")

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

            ojo_der = puntos[R_START:R_END]
            ojo_izq = puntos[L_START:L_END]

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
                    rango_movil = max(0.08, ear_base_abierto - piso_fisiologico)

                    threshold_adaptativo_cierre = piso_fisiologico + (
                        (1.0 - SENSIBILIDAD_CIERRE) * rango_movil
                    )
                    threshold_adaptativo_descenso = piso_fisiologico + (
                        FACTOR_DESCENSO * rango_movil
                    )

        if threshold_adaptativo_cierre < threshold_estatico_cierre:
            threshold_adaptativo_cierre = threshold_estatico_cierre

        # ---------------------------------------------------------
        # Detección y Cronometraje de Cierre Profundo ('C')
        # ---------------------------------------------------------
        if ear_crudo < threshold_adaptativo_descenso:
            if not en_ciclo_parpadeo and (frame_idx - ultimo_frame_fin) > COOLDOWN_FRAMES:
                en_ciclo_parpadeo = True
                cierre_efectivo = False
                frame_inicio_cierre = -1
                frame_fin_cierre = -1

            if en_ciclo_parpadeo:
                if ear_crudo <= threshold_adaptativo_cierre:
                    if not cierre_efectivo:
                        cierre_efectivo = True
                        frame_inicio_cierre = frame_idx
                    frame_fin_cierre = frame_idx
        else:
            if en_ciclo_parpadeo:
                if cierre_efectivo:
                    dur_f = frame_fin_cierre - frame_inicio_cierre + 1
                    if dur_f >= DURACION_MIN_FRAMES:
                        pred_blinks.append({
                            "start": frame_inicio_cierre,
                            "end": frame_fin_cierre,
                            "frames": dur_f,
                            "duration_ms": (dur_f / fps) * 1000.0,
                        })
                        ultimo_frame_fin = frame_idx - 1

                en_ciclo_parpadeo = False
                cierre_efectivo = False

        # ---------------------------------------------------------
        # PERCLOS Fisiológico Estricto
        # ---------------------------------------------------------
        ojo_cerrado_pred = 1 if ear_crudo <= threshold_adaptativo_cierre else 0
        ojo_cerrado_gt = gt_estados_frame.get(frame_idx, 0)

        buffer_ojos_pred.append(ojo_cerrado_pred)
        buffer_ojos_gt.append(ojo_cerrado_gt)

        perclos_pred = (
            (sum(buffer_ojos_pred) / len(buffer_ojos_pred)) * 100.0
            if len(buffer_ojos_pred) > 0
            else 0.0
        )
        perclos_gt = (
            (sum(buffer_ojos_gt) / len(buffer_ojos_gt)) * 100.0
            if len(buffer_ojos_gt) > 0
            else 0.0
        )

        log_perclos_pred.append(perclos_pred)
        log_perclos_gt.append(perclos_gt)

        frame_idx += 1

    cap.release()

    # ---------------------------------------------------------
    # Emparejamiento de Eventos Predichos vs. Ground Truth
    # ---------------------------------------------------------
    matched_gt = set()
    datos_exportacion = []

    for pred_idx, pred in enumerate(pred_blinks):
        p_set = set(range(pred["start"], pred["end"] + 1))
        mejor_iou = 0.0
        mejor_gt_idx = -1

        for idx, gt in enumerate(gt_blinks):
            if idx in matched_gt:
                continue
            gt_set = set(range(gt["start"], gt["end"] + 1))
            inter = len(p_set.intersection(gt_set))
            union = len(p_set.union(gt_set))

            if union > 0:
                iou = inter / union
                if iou > mejor_iou:
                    mejor_iou = iou
                    mejor_gt_idx = idx

        registro = {
            "Pred_ID": pred_idx + 1,
            "Pred_Frame_Inicio": pred["start"],
            "Pred_Frame_Fin": pred["end"],
            "Pred_Duracion_ms": round(pred["duration_ms"], 2),
            "Tipo": "Falso Positivo",
            "GT_ID": None,
            "GT_Frame_Inicio": None,
            "GT_Frame_Fin": None,
            "GT_Duracion_ms": None,
            "Error_Duracion_ms": None,
        }

        if mejor_iou >= IOU_THRESHOLD:
            matched_gt.add(mejor_gt_idx)
            gt_blink = gt_blinks[mejor_gt_idx]
            err = abs(pred["duration_ms"] - gt_blink["duration_ms"])

            registro["Tipo"] = "Verdadero Positivo"
            registro["GT_ID"] = mejor_gt_idx + 1
            registro["GT_Frame_Inicio"] = gt_blink["start"]
            registro["GT_Frame_Fin"] = gt_blink["end"]
            registro["GT_Duracion_ms"] = round(gt_blink["duration_ms"], 2)
            registro["Error_Duracion_ms"] = round(err, 2)

        datos_exportacion.append(registro)

    for idx, gt in enumerate(gt_blinks):
        if idx not in matched_gt:
            datos_exportacion.append({
                "Pred_ID": None,
                "Pred_Frame_Inicio": None,
                "Pred_Frame_Fin": None,
                "Pred_Duracion_ms": None,
                "Tipo": "Falso Negativo",
                "GT_ID": idx + 1,
                "GT_Frame_Inicio": gt["start"],
                "GT_Frame_Fin": gt["end"],
                "GT_Duracion_ms": round(gt["duration_ms"], 2),
                "Error_Duracion_ms": None,
            })

    df_eventos = pd.DataFrame(datos_exportacion)
    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
    df_eventos.to_csv(archivo_salida, index=False, encoding="utf-8-sig")

    if extraer_clips:
        extraer_falsos_positivos(
            video_path, df_eventos, output_dir=os.path.join(output_dir, f"fps_clips{video_id}")
        )

    mae_perclos = (
        float(np.mean(np.abs(np.array(log_perclos_pred) - np.array(log_perclos_gt))))
        if log_perclos_pred
        else 0.0
    )
    return df_eventos, gt_blinks, mae_perclos, fps


def filtrar_edge_cases(df_consolidado, edge_cases=EDGE_CASES_GT):
    """
    Separa el DataFrame consolidado en:
      - `df_filtrado`: eventos excluyendo los parpadeos designados como edge cases `(Video, GT_ID)`.
      - `df_excluidos`: filas correspondientes a los edge cases excluidos.
    """
    set_edge = {(int(v), int(g)) for v, g in edge_cases}
    es_edge = df_consolidado.apply(
        lambda r: pd.notna(r["GT_ID"]) and (int(r["Video"]), int(r["GT_ID"])) in set_edge,
        axis=1,
    )
    df_excluidos = df_consolidado[es_edge].copy().reset_index(drop=True)
    df_filtrado = df_consolidado[~es_edge].copy().reset_index(drop=True)
    return df_filtrado, df_excluidos


def calcular_metricas_subconjunto(df_eventos, min_ms=0.0):
    """
    Calcula TP, FP, FN, Precisión, Recall, F1-Score y MAE/Mediana de error de duración (ms)
    para todos los parpadeos con duración >= min_ms:
      - Si min_ms == 0: evalúa todos los parpadeos (términos generales).
      - Si min_ms > 0 : evalúa los parpadeos reales con `GT_Duracion_ms >= min_ms` (TP y FN)
                        frente a los falsos positivos con `Pred_Duracion_ms >= min_ms` (FP).
    """
    eps = 1e-6
    if min_ms <= 0:
        tp_df = df_eventos[df_eventos["Tipo"] == "Verdadero Positivo"]
        fn_df = df_eventos[df_eventos["Tipo"] == "Falso Negativo"]
        fp_df = df_eventos[df_eventos["Tipo"] == "Falso Positivo"]
    else:
        gt_mask = df_eventos["GT_Duracion_ms"] >= (min_ms - eps)
        tp_df = df_eventos[(df_eventos["Tipo"] == "Verdadero Positivo") & gt_mask]
        fn_df = df_eventos[(df_eventos["Tipo"] == "Falso Negativo") & gt_mask]
        fp_df = df_eventos[
            (df_eventos["Tipo"] == "Falso Positivo")
            & (df_eventos["Pred_Duracion_ms"] >= (min_ms - eps))
        ]

    tp = int(len(tp_df))
    fn = int(len(fn_df))
    fp = int(len(fp_df))
    n_gt = tp + fn
    n_pred = tp + fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    errores = tp_df["Error_Duracion_ms"].dropna()
    mae_duracion = float(errores.mean()) if len(errores) > 0 else 0.0
    mediana_duracion = float(errores.median()) if len(errores) > 0 else 0.0
    std_duracion = float(errores.std(ddof=0)) if len(errores) > 0 else 0.0

    return {
        "GT_Parpadeos": n_gt,
        "Pred_Parpadeos": n_pred,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "MAE_Duracion_ms": round(mae_duracion, 2),
        "Mediana_Error_ms": round(mediana_duracion, 2),
        "Std_Error_ms": round(std_duracion, 2),
    }


def evaluar_por_intervalos_100ms(df_consolidado, umbrales_ms=UMBRALES_DURACION_100MS, fps=30.0):
    """
    Construye la tabla de métricas en términos generales (Todos) y por intervalos de 100 ms
    (>= 100 ms, >= 200 ms, >= 300 ms, >= 400 ms, >= 500 ms, etc.).
    """
    filas = []

    m_all = calcular_metricas_subconjunto(df_consolidado, min_ms=0.0)
    filas.append({
        "Intervalo_Duracion": "General (Todos)",
        "Umbral_Min_ms": 0,
        "Frames_Min_30fps": DURACION_MIN_FRAMES,
        **m_all,
    })

    for min_ms in sorted(umbrales_ms):
        m = calcular_metricas_subconjunto(df_consolidado, min_ms=float(min_ms))
        filas.append({
            "Intervalo_Duracion": f">= {int(min_ms)} ms",
            "Umbral_Min_ms": int(min_ms),
            "Frames_Min_30fps": int(round((min_ms / 1000.0) * fps)),
            **m,
        })

    return pd.DataFrame(filas)


def evaluar_videos_por_intervalos_100ms(df_consolidado, umbrales_ms=(100, 200, 300, 400, 500)):
    """
    Desglosa por cada uno de los 8 videos cuántos parpadeos reales (GT) y detectados (TP)
    existen en cada umbral de duración (>= 100 ms, >= 200 ms, >= 300 ms, >= 400 ms, >= 500 ms).
    """
    filas = []
    for vid_id, sub_df in df_consolidado.groupby("Video", sort=True):
        fila = {"Video": vid_id}
        m_gen = calcular_metricas_subconjunto(sub_df, min_ms=0.0)
        fila["GT_Total"] = m_gen["GT_Parpadeos"]
        fila["TP_Total"] = m_gen["TP"]
        fila["F1_Total"] = m_gen["F1_Score"]
        fila["MAE_Total_ms"] = m_gen["MAE_Duracion_ms"]

        for u in umbrales_ms:
            m_u = calcular_metricas_subconjunto(sub_df, min_ms=float(u))
            n_gt = m_u["GT_Parpadeos"]
            tp = m_u["TP"]
            rec_str = f"{m_u['Recall']*100:.0f}%" if n_gt > 0 else "N/A"
            fila[f">={u}ms (TP/GT)"] = f"{tp}/{n_gt} ({rec_str})"
            fila[f"MAE_>={u}ms"] = m_u["MAE_Duracion_ms"] if tp > 0 else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def graficar_metricas_por_duracion_eb8(df_intervalos, ruta_salida, sufijo_titulo="Sin Edge Cases"):
    """Genera una gráfica con el desempeño general y por intervalos de 100 ms (>= 100 ms en adelante)."""
    df_plot = df_intervalos[df_intervalos["Umbral_Min_ms"] >= 100].copy()
    if df_plot.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    etiquetas = [f">={u} ms" for u in df_plot["Umbral_Min_ms"]]
    x = np.arange(len(etiquetas))
    w = 0.36

    b1 = ax1.bar(
        x - w / 2,
        df_plot["GT_Parpadeos"],
        w,
        label="Parpadeos Reales GT (n)",
        color="tab:gray",
        alpha=0.8,
    )
    b2 = ax1.bar(
        x + w / 2,
        df_plot["TP"],
        w,
        label="Detectados TP (m)",
        color="tab:blue",
    )

    max_y = max(df_plot["GT_Parpadeos"].max(), 1)
    for b in b1:
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max_y * 0.015,
            f"{int(b.get_height())}",
            ha="center",
            fontsize=8.5,
        )
    for b in b2:
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max_y * 0.015,
            f"{int(b.get_height())}",
            ha="center",
            fontsize=8.5,
            color="tab:blue",
            fontweight="bold",
        )

    ax1.set_xticks(x, etiquetas, rotation=25)
    ax1.set_xlabel("Umbral de duración mínima del parpadeo (ms)")
    ax1.set_ylabel("Cantidad de parpadeos")
    ax1.set_title(f"Eyeblink8 ({sufijo_titulo}): GT vs. TP por Intervalo de 100 ms")
    ax1.set_ylim(0, max_y * 1.15)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(
        etiquetas,
        df_plot["Recall"] * 100.0,
        "o-",
        lw=2,
        color="tab:green",
        label="Recall / Sensibilidad (%)",
    )
    ax2.plot(
        etiquetas,
        df_plot["Precision"] * 100.0,
        "s-",
        lw=2,
        color="tab:blue",
        label="Precisión (%)",
    )
    ax2.plot(
        etiquetas,
        df_plot["F1_Score"] * 100.0,
        "^-",
        lw=2,
        color="tab:orange",
        label="F1-Score (%)",
    )

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        etiquetas,
        df_plot["MAE_Duracion_ms"],
        "d--",
        lw=1.8,
        color="tab:red",
        label="MAE Duración (ms)",
    )
    ax2_twin.set_ylabel("MAE Duración (ms)", color="tab:red")
    ax2_twin.tick_params(axis="y", labelcolor="tab:red")
    ax2_twin.set_ylim(0, max(df_plot["MAE_Duracion_ms"].max() * 1.3, 80))

    ax2.set_xticks(x, etiquetas, rotation=25)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Umbral de duración mínima del parpadeo (ms)")
    ax2.set_ylabel("Métrica (%)")
    ax2.set_title(f"Precisión, Recall, F1 y MAE ({sufijo_titulo})")
    ax2.grid(True, alpha=0.3)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc="center right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Eyeblink8 (8 videos calibrados): Métricas generales, sin edge cases y por intervalos de 100 ms"
    )
    parser.add_argument(
        "--videos",
        type=int,
        nargs="*",
        default=list(VIDEOS_EB8.keys()),
        help="IDs de videos a evaluar (por defecto: 1 2 3 4 8 9 10 11)",
    )
    parser.add_argument(
        "--recalcular",
        action="store_true",
        help="Reprocesar los videos .avi cuadro a cuadro con dlib en lugar de usar los CSV precalculados",
    )
    parser.add_argument(
        "--incluir-edge-cases",
        action="store_true",
        help="No excluir los 11 edge cases designados (por defecto se excluyen y se muestra la comparativa)",
    )
    parser.add_argument(
        "--extraer-clips",
        action="store_true",
        help="Extraer clips .avi de los Falsos Positivos por cada video",
    )
    parser.add_argument(
        "--output-dir",
        default="eventos_parpadeos",
        help="Carpeta de salida para los archivos CSV y gráficas",
    )
    args = parser.parse_args()

    videos_seleccionados = [v for v in args.videos if v in VIDEOS_EB8]
    if not videos_seleccionados:
        raise ValueError(f"No se especificaron videos válidos. Opciones: {list(VIDEOS_EB8.keys())}")

    detector = None
    predictor = None
    if args.recalcular:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    dfs_eventos_raw = []
    mae_perclos_por_vid = {}

    for vid_id in videos_seleccionados:
        cfg = VIDEOS_EB8[vid_id]
        df_ev, _, mae_perclos, _ = procesar_video_eb8(
            video_id=vid_id,
            cfg=cfg,
            detector=detector,
            predictor=predictor,
            usar_cache_csv=not args.recalcular,
            extraer_clips=args.extraer_clips,
            output_dir=args.output_dir,
        )
        df_ev = df_ev.copy()
        df_ev.insert(0, "Video", vid_id)
        dfs_eventos_raw.append(df_ev)
        mae_perclos_por_vid[vid_id] = mae_perclos

    df_consolidado_raw = pd.concat(dfs_eventos_raw, ignore_index=True)

    # Filtrar los 11 edge cases designados ("n-m": Video n, GT_ID m)
    if args.incluir_edge_cases:
        df_consolidado = df_consolidado_raw.copy()
        df_excluidos = pd.DataFrame()
    else:
        df_consolidado, df_excluidos = filtrar_edge_cases(df_consolidado_raw, EDGE_CASES_GT)

    # Construir tabla por video sobre el conjunto activo (sin edge cases por defecto)
    filas_por_video = []
    for vid_id in videos_seleccionados:
        cfg = VIDEOS_EB8[vid_id]
        sub_vid = df_consolidado[df_consolidado["Video"] == vid_id]
        n_excl_vid = int((df_excluidos["Video"] == vid_id).sum()) if not df_excluidos.empty else 0
        m_vid = calcular_metricas_subconjunto(sub_vid, min_ms=0.0)
        filas_por_video.append({
            "Video": vid_id,
            "EAR_Abierto_Cal": cfg["ear_base_abierto"],
            "EAR_Cerrado_Cal": cfg["ear_base_cerrado"],
            "Edge_Excluidos": n_excl_vid,
            **m_vid,
            "MAE_PERCLOS_%": round(mae_perclos_por_vid[vid_id], 2),
        })

    df_por_video = pd.DataFrame(filas_por_video)

    # Métricas globales con y sin edge cases
    m_global_raw = calcular_metricas_subconjunto(df_consolidado_raw, min_ms=0.0)
    m_global = calcular_metricas_subconjunto(df_consolidado, min_ms=0.0)
    mae_perclos_global = float(df_por_video["MAE_PERCLOS_%"].mean())

    prec_macro = float(df_por_video["Precision"].mean())
    rec_macro = float(df_por_video["Recall"].mean())
    f1_macro = float(df_por_video["F1_Score"].mean())
    mae_dur_macro = float(df_por_video["MAE_Duracion_ms"].mean())

    # Evaluación por intervalos de 100 ms (con y sin edge cases)
    df_intervalos_raw = evaluar_por_intervalos_100ms(df_consolidado_raw, UMBRALES_DURACION_100MS)
    df_intervalos = evaluar_por_intervalos_100ms(df_consolidado, UMBRALES_DURACION_100MS)
    df_videos_intervalos = evaluar_videos_por_intervalos_100ms(
        df_consolidado, umbrales_ms=[100, 200, 300, 400, 500]
    )

    # ---------------------------------------------------------
    # Impresión de Resultados
    # ---------------------------------------------------------
    if not df_excluidos.empty:
        print("\n" + "=" * 112)
        print(f" EDGE CASES EXCLUIDOS ({len(df_excluidos)} casos 'Video-GT_ID'): {[f'{v}-{g}' for v, g in EDGE_CASES_GT]}")
        print("=" * 112)
        df_excl_vista = df_excluidos[
            ["Video", "GT_ID", "Tipo", "GT_Frame_Inicio", "GT_Frame_Fin", "GT_Duracion_ms"]
        ].copy()
        df_excl_vista["Caso"] = df_excl_vista.apply(
            lambda r: f"{int(r['Video'])}-{int(r['GT_ID'])}", axis=1
        )
        cols_excl = ["Caso", "Video", "GT_ID", "Tipo", "GT_Frame_Inicio", "GT_Frame_Fin", "GT_Duracion_ms"]
        print(df_excl_vista[cols_excl].to_string(index=False))

    etiqueta_modo = "CON EDGE CASES" if args.incluir_edge_cases else "SIN EDGE CASES (11 EXCLUIDOS)"

    print("\n" + "=" * 118)
    print(f" 1. RESULTADOS POR VIDEO INDIVIDUAL ({etiqueta_modo})")
    print("=" * 118)
    cols_video = [
        "Video",
        "EAR_Abierto_Cal",
        "EAR_Cerrado_Cal",
        "Edge_Excluidos",
        "GT_Parpadeos",
        "Pred_Parpadeos",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1_Score",
        "MAE_Duracion_ms",
        "MAE_PERCLOS_%",
    ]
    print(df_por_video[cols_video].to_string(index=False))

    print("\n" + "=" * 95)
    print(f" 2. RESULTADOS EN TÉRMINOS GENERALES ({etiqueta_modo} vs. ORIGINAL)")
    print("=" * 95)
    print(f"{'Métrica':<40} | {'Con Edge Cases':>22} | {'Sin Edge Cases':>22}")
    print("-" * 95)
    print(f"{'Parpadeos Reales Totales (GT)':<40} | {m_global_raw['GT_Parpadeos']:>22d} | {m_global['GT_Parpadeos']:>22d}")
    print(f"{'Parpadeos Detectados Totales':<40} | {m_global_raw['Pred_Parpadeos']:>22d} | {m_global['Pred_Parpadeos']:>22d}")
    print(f"{'Verdaderos Positivos (TP)':<40} | {m_global_raw['TP']:>22d} | {m_global['TP']:>22d}")
    print(f"{'Falsos Positivos (FP)':<40} | {m_global_raw['FP']:>22d} | {m_global['FP']:>22d}")
    print(f"{'Falsos Negativos (FN)':<40} | {m_global_raw['FN']:>22d} | {m_global['FN']:>22d}")
    print("-" * 95)
    print(
        f"{'Precisión Global (Micro)':<40} | "
        f"{m_global_raw['Precision']:>12.4f} ({m_global_raw['Precision']*100:5.2f}%) | "
        f"{m_global['Precision']:>12.4f} ({m_global['Precision']*100:5.2f}%)"
    )
    print(
        f"{'Recall / Sensibilidad Global (Micro)':<40} | "
        f"{m_global_raw['Recall']:>12.4f} ({m_global_raw['Recall']*100:5.2f}%) | "
        f"{m_global['Recall']:>12.4f} ({m_global['Recall']*100:5.2f}%)"
    )
    print(
        f"{'F1-Score Global (Micro)':<40} | "
        f"{m_global_raw['F1_Score']:>12.4f} ({m_global_raw['F1_Score']*100:5.2f}%) | "
        f"{m_global['F1_Score']:>12.4f} ({m_global['F1_Score']*100:5.2f}%)"
    )
    print(
        f"{'MAE Duración de Parpadeo Global':<40} | "
        f"{m_global_raw['MAE_Duracion_ms']:>19.2f} ms | "
        f"{m_global['MAE_Duracion_ms']:>19.2f} ms"
    )
    print(
        f"{'Mediana Error Duración Global':<40} | "
        f"{m_global_raw['Mediana_Error_ms']:>19.2f} ms | "
        f"{m_global['Mediana_Error_ms']:>19.2f} ms"
    )
    print("-" * 95)
    print(
        f"Promedio Macro por Video ({etiqueta_modo}) -> "
        f"Precisión: {prec_macro:.4f} | Recall: {rec_macro:.4f} | F1: {f1_macro:.4f} | MAE Duración: {mae_dur_macro:.2f} ms"
    )
    print("=" * 95)

    print("\n" + "=" * 118)
    print(f" 3. DETECCIÓN POR INTERVALOS DE DURACIÓN DE 100 ms ({etiqueta_modo})")
    print("=" * 118)
    cols_int = [
        "Intervalo_Duracion",
        "Frames_Min_30fps",
        "GT_Parpadeos",
        "TP",
        "FN",
        "FP",
        "Precision",
        "Recall",
        "F1_Score",
        "MAE_Duracion_ms",
        "Mediana_Error_ms",
    ]
    print(df_intervalos[cols_int].to_string(index=False))

    print("\n" + "-" * 118)
    print(f" RESUMEN DE INTERVALOS PRINCIPALES (500 ms PARA ARRIBA -> 100 ms PARA ARRIBA, {etiqueta_modo})")
    print("-" * 118)
    df_500_a_100 = (
        df_intervalos[df_intervalos["Umbral_Min_ms"].isin([500, 400, 300, 200, 100])]
        .sort_values("Umbral_Min_ms", ascending=False)
    )
    for _, row in df_500_a_100.iterrows():
        u = int(row["Umbral_Min_ms"])
        row_raw = df_intervalos_raw[df_intervalos_raw["Umbral_Min_ms"] == u].iloc[0]
        print(
            f"  * Parpadeos >= {u:3d} ms | GT: {int(row['GT_Parpadeos']):3d} (antes {int(row_raw['GT_Parpadeos']):3d}) | "
            f"TP: {int(row['TP']):3d} | FN: {int(row['FN']):2d} (antes {int(row_raw['FN']):2d}) | FP: {int(row['FP']):2d} | "
            f"Precisión: {row['Precision']:.4f} | Recall: {row['Recall']:.4f} (antes {row_raw['Recall']:.4f}) | "
            f"F1: {row['F1_Score']:.4f} (antes {row_raw['F1_Score']:.4f}) | MAE: {row['MAE_Duracion_ms']:5.2f} ms"
        )
    print("-" * 118)

    print("\n" + "=" * 118)
    print(f" 4. DESGLOSE POR VIDEO EN INTERVALOS DE 100 ms (>= 100 ms A >= 500 ms, {etiqueta_modo})")
    print("=" * 118)
    cols_vid_int = [
        "Video",
        "GT_Total",
        "TP_Total",
        ">=100ms (TP/GT)",
        ">=200ms (TP/GT)",
        ">=300ms (TP/GT)",
        ">=400ms (TP/GT)",
        ">=500ms (TP/GT)",
    ]
    print(df_videos_intervalos[cols_vid_int].to_string(index=False))
    print("=" * 118)

    # Exportar resultados a CSV y gráfica
    os.makedirs(args.output_dir, exist_ok=True)
    suf_archivo = "" if args.incluir_edge_cases else "_sin_edge_cases"
    ruta_csv_videos = os.path.join(args.output_dir, f"metricas_por_video{suf_archivo}_eb8.csv")
    ruta_csv_intervalos = os.path.join(args.output_dir, f"metricas_por_duracion_100ms{suf_archivo}_eb8.csv")
    ruta_csv_consolidado = os.path.join(args.output_dir, f"eventos_parpadeos_benchmark_consolidado{suf_archivo}.csv")
    ruta_grafica = os.path.join(args.output_dir, f"deteccion_por_duracion{suf_archivo}_eb8.png")

    def _guardar_csv_seguro(df, ruta):
        try:
            df.to_csv(ruta, index=False, encoding="utf-8-sig")
            return ruta
        except PermissionError:
            base, ext = os.path.splitext(ruta)
            alt = f"{base}_actualizado{ext}"
            df.to_csv(alt, index=False, encoding="utf-8-sig")
            print(f"[AVISO] '{ruta}' está abierto en otro programa; se guardó en '{alt}'.")
            return alt

    ruta_csv_videos = _guardar_csv_seguro(df_por_video, ruta_csv_videos)
    ruta_csv_intervalos = _guardar_csv_seguro(df_intervalos, ruta_csv_intervalos)
    ruta_csv_consolidado = _guardar_csv_seguro(df_consolidado, ruta_csv_consolidado)
    graficar_metricas_por_duracion_eb8(
        df_intervalos,
        ruta_grafica,
        sufijo_titulo="Con Edge Cases" if args.incluir_edge_cases else "Sin Edge Cases",
    )

    print(f"\nArchivos exportados exitosamente en '{args.output_dir}/':")
    print(f"  - {ruta_csv_videos}")
    print(f"  - {ruta_csv_intervalos}")
    print(f"  - {ruta_csv_consolidado}")
    print(f"  - {ruta_grafica}")


if __name__ == "__main__":
    main()
