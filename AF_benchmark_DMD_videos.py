"""
AF_benchmark_DMD_videos.py
==========================
Evaluación basada en EVENTOS Y DURACIÓN DE PARPADEOS (alineada con `AF_benchmark_EB8.py`,
`AF_test.py` y `AF_benchmark_DMD.py`) directamente sobre los VIDEOS ORIGINALES EN HD (1280x720,
`*_rgb_face.mp4`) y sus anotaciones oficiales OpenLABEL (`*_rgb_ann_drowsiness.json`) en la
carpeta `dataset_dmd/`.

Características clave:
  * SIN REDIMENSIONAMIENTO DE IMAGEN: Procesa los fotogramas de `*_rgb_face.mp4` a su resolución
    nativa 1280x720 (igual que `AF_test.py` y `AF_benchmark_EB8.py`), sin usar imágenes de
    224x224 ni requerir herramientas externas del DMD.
  * Lee las etiquetas (`eyes_state/open`, `eyes_state/close`, `eyes_state/closing`,
    `eyes_state/opening`) directamente de los archivos `*_rgb_ann_drowsiness.json`.
  * Extrae los 285 eventos reales de cierre (`eyes_state/close`) con su duración en milisegundos
    (a 30 fps) y evalúa cuántos parpadeos de `n` ms en adelante (p. ej. >= 500 ms) son detectados
    exitosamente por el algoritmo, generando las mismas tablas, gráficas y clips `.avi` que
    `AF_benchmark_DMD.py` dentro de `resultados_dmd_videos/`.
"""
import argparse
from collections import deque
import glob
import json
import os
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import dlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Configuración y Parámetros Fisiológicos (AF_benchmark_EB8 / AF_test)
# ---------------------------------------------------------
DIR_DATASET_DMD = BASE_DIR / "dataset_dmd"
DIR_SALIDA = BASE_DIR / "resultados_dmd_videos"
MODELO_DLIB = BASE_DIR / "shape_predictor_68_face_landmarks.dat"

FPS_DMD = 30.0
EAR_ABIERTO_EB8 = 0.2981
PISO_FISIOLOGICO_DLIB = 0.1525
FACTOR_DESCENSO = 0.70
SENSIBILIDAD_CIERRE = 0.80   # 1 - 0.80 = 0.20 (20% del rango de apertura = ojo cerrado)
DURACION_MIN_FRAMES = 2      # mínimo 2 frames (66.7 ms a 30 fps)
COOLDOWN_FRAMES = 2          # tolerancia de unión entre frames cerrados consecutivos (66.7 ms)
IOU_THRESHOLD = 0.20         # umbral estricto de solapamiento temporal de AF_benchmark_EB8.py

UMBRALES_DURACION_MS = [100, 200, 300, 400, 500, 600, 750, 1000, 1500, 2000]
CALIBRACIONES = ["grabacion", "dataset", "fija_eb8", "fija_estatica"]
MODOS_FILTRO = ["crudo_estatico", "adaptativo_eb8", "adaptativo_eb8_suav"]

(R_START, R_END) = (36, 42)
(L_START, L_END) = (42, 48)

_detector = None
_predictor = None


def calcular_ear(ojo):
    A = dist.euclidean(ojo[1], ojo[5])
    B = dist.euclidean(ojo[2], ojo[4])
    C = dist.euclidean(ojo[0], ojo[3])
    return (A + B) / (2.0 * C)


def _guardar_csv(df, ruta, **kwargs):
    try:
        df.to_csv(ruta, **kwargs)
    except PermissionError:
        alt = Path(ruta).with_name(f"{Path(ruta).stem}_actualizado{Path(ruta).suffix}")
        df.to_csv(alt, **kwargs)
        print(f"[AVISO] '{Path(ruta).name}' está abierto en otro programa; se guardó en '{alt.name}'.")


# ---------------------------------------------------------
# Descubrimiento de Videos y Lectura de Anotaciones JSON (OpenLABEL)
# ---------------------------------------------------------
_RE_TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2};\d{2};\d{2})")


def id_grabacion_desde_ruta(ruta):
    """Convierte '..._2019-03-14T14;26;17+01;00_...' en '2019-03-14-14;26;17' (mismo ID que AF_benchmark_DMD.py)."""
    m = _RE_TIMESTAMP.search(os.path.basename(ruta))
    return f"{m.group(1)}-{m.group(2)}" if m else Path(ruta).stem


def listar_videos_dmd(dir_dataset, max_grabaciones=0):
    patron = str(Path(dir_dataset) / "**" / "*_rgb_ann_drowsiness.json")
    jsons = sorted(glob.glob(patron, recursive=True))
    pares = []
    for jpath in jsons:
        vpath = jpath.replace("_rgb_ann_drowsiness.json", "_rgb_face.mp4")
        if not os.path.exists(vpath):
            print(f"[AVISO] No se encontró el video '{vpath}' para '{jpath}'")
            continue
        rec_id = id_grabacion_desde_ruta(vpath)
        pares.append((rec_id, vpath, jpath))
    pares.sort(key=lambda t: t[0])
    if max_grabaciones > 0:
        pares = pares[:max_grabaciones]
    return pares


def cargar_etiquetas_json(json_path):
    """Lee el archivo OpenLABEL `*_rgb_ann_drowsiness.json` y devuelve:
      - mapa_clases: dict {frame_idx: 'open' | 'close' | 'closing' | 'opening'}
      - max_frame  : último frame anotado en `eyes_state/*`
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    root = data.get("openlabel", data.get("vcd", data))
    actions = root.get("actions", {})
    mapa_clases = {}
    max_frame = -1
    for act in actions.values():
        tipo = act.get("type", "")
        if not tipo.startswith("eyes_state/"):
            continue
        clase = tipo.split("/", 1)[1]
        if clase not in ("open", "close", "closing", "opening"):
            continue
        for iv in act.get("frame_intervals", []):
            f0, f1 = int(iv["frame_start"]), int(iv["frame_end"])
            if f1 > max_frame:
                max_frame = f1
            for f_idx in range(f0, f1 + 1):
                mapa_clases[f_idx] = clase
    return mapa_clases, max_frame


# ---------------------------------------------------------
# Etapa 1: Procesamiento Nativo 1280x720 con dlib (Sin Redimensionar)
# ---------------------------------------------------------
def _init_worker(modelo_path):
    global _detector, _predictor
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(str(modelo_path))


def _fraccion_piel(img_bgr, x1, y1, x2, y2):
    h_im, w_im = img_bgr.shape[:2]
    w, h = x2 - x1, y2 - y1
    xa, xb = max(0, int(x1 + 0.2 * w)), min(w_im, int(x2 - 0.2 * w))
    ya, yb = max(0, int(y1 + 0.2 * h)), min(h_im, int(y2 - 0.2 * h))
    if xb <= xa or yb <= ya:
        return 0.0
    roi = img_bgr[ya:yb, xa:xb]
    mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb), (0, 130, 75), (255, 175, 130))
    return float(mask.mean() / 255.0)


def _ear_desde_caja(gray, rect, shrink=0.88):
    cx = (rect.left() + rect.right()) / 2.0
    cy = (rect.top() + rect.bottom()) / 2.0 - (1.0 - shrink) * 0.30 * rect.height()
    w, h = rect.width() * shrink, rect.height() * shrink
    r_adj = dlib.rectangle(int(round(cx - w / 2.0)), int(round(cy - h / 2.0)),
                           int(round(cx + w / 2.0)), int(round(cy + h / 2.0)))
    shape = _predictor(gray, r_adj)
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype="float")
    ear_der = calcular_ear(pts[R_START:R_END])
    ear_izq = calcular_ear(pts[L_START:L_END])
    return (ear_der + ear_izq) / 2.0, ear_der, ear_izq, pts


def _detectar_en_ventana(gray, frame_bgr, rx1, ry1, rx2, ry2, thr_score=-0.8):
    roi = np.ascontiguousarray(gray[ry1:ry2, rx1:rx2])
    dets, scores, _ = _detector.run(roi, 0, thr_score)
    candidatos = []
    for d, s in zip(dets, scores):
        w = d.width()
        if not (140 <= w <= 360):
            continue
        x1, y1, x2, y2 = d.left() + rx1, d.top() + ry1, d.right() + rx1, d.bottom() + ry1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Zona plausible del conductor en 1280x720 (descarta cámara del tablero a la izquierda y pecho abajo)
        if not (340 <= cx <= 940 and 100 <= cy <= 530):
            continue
        piel = _fraccion_piel(frame_bgr, x1, y1, x2, y2)
        if piel < 0.20:
            continue
        candidatos.append((s, piel, dlib.rectangle(x1, y1, x2, y2)))
    if not candidatos:
        return None, np.nan, np.nan
    s, piel, mejor = max(candidatos, key=lambda t: t[0])
    return mejor, float(s), float(piel)


def procesar_video_completo(arg):
    """Procesa un video completo `_rgb_face.mp4` a su resolución nativa 1280x720 para todos los
    frames etiquetados en `eyes_state/*`, incluyendo estabilización de trayectoria de caja."""
    rec_id, vpath, jpath = arg
    t0 = time.time()
    mapa_clases, max_frame_ann = cargar_etiquetas_json(jpath)
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {vpath}")

    registros = []
    buf_cajas = deque(maxlen=15)  # Historial reciente (cx, cy, w, h) que sigue suavemente al rostro
    f_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx > max_frame_ann and max_frame_ann > 0:
            break

        clase = mapa_clases.get(f_idx)
        if clase is None:
            f_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Búsqueda 100% nativa en 1280x720 (sin ningún redimensionamiento) en la región del conductor
        rect, s, piel = _detectar_en_ventana(gray, frame, 320, 80, 980, 650, thr_score=-1.0)

        metodo = "sin_rostro"
        if rect is not None:
            cx = (rect.left() + rect.right()) / 2.0
            cy = (rect.top() + rect.bottom()) / 2.0
            w, h = float(rect.width()), float(rect.height())
            # Si el detector devuelve una caja válida con piel, actualizar siempre el historial
            # y amortiguar solo saltos aislados de 1 frame (> 90 px en 1280x720)
            if len(buf_cajas) >= 5:
                arr_c = np.array(buf_cajas)
                cx_m, cy_m, w_m, h_m = np.median(arr_c, axis=0)
                if np.hypot(cx - cx_m, cy - cy_m) > 90.0 or abs(w - w_m) / max(w_m, 1.0) > 0.25:
                    # Amortiguar hacia la mediana pero actualizar el buffer para nunca congelarse
                    buf_cajas.append((0.5 * cx + 0.5 * cx_m, 0.5 * cy + 0.5 * cy_m, w_m, h_m))
                    bx1 = int(round(cx_m - w_m / 2.0))
                    by1 = int(round(cy_m - h_m / 2.0))
                    bx2 = int(round(cx_m + w_m / 2.0))
                    by2 = int(round(cy_m + h_m / 2.0))
                    rect = dlib.rectangle(bx1, by1, bx2, by2)
                    metodo = "caja_trayectoria"
                else:
                    buf_cajas.append((cx, cy, w, h))
                    metodo = "dlib_estandar" if s >= 0.0 else "dlib_relajado"
            else:
                buf_cajas.append((cx, cy, w, h))
                metodo = "dlib_estandar" if s >= 0.0 else "dlib_relajado"
        elif len(buf_cajas) > 0:
            arr_c = np.array(buf_cajas)
            cx_m, cy_m, w_m, h_m = np.median(arr_c, axis=0)
            bx1 = int(round(cx_m - w_m / 2.0))
            by1 = int(round(cy_m - h_m / 2.0))
            bx2 = int(round(cx_m + w_m / 2.0))
            by2 = int(round(cy_m + h_m / 2.0))
            rect = dlib.rectangle(bx1, by1, bx2, by2)
            metodo = "caja_trayectoria"

        if rect is not None:
            ear, ear_der, ear_izq, _ = _ear_desde_caja(gray, rect)
            registros.append({
                "grabacion": rec_id,
                "video_path": vpath,
                "frame": f_idx,
                "clase": clase,
                "ear": round(float(ear), 5),
                "ear_der": round(float(ear_der), 5),
                "ear_izq": round(float(ear_izq), 5),
                "x1": int(rect.left()),
                "y1": int(rect.top()),
                "x2": int(rect.right()),
                "y2": int(rect.bottom()),
                "score": round(s, 3) if not np.isnan(s) else np.nan,
                "piel": round(piel, 3) if not np.isnan(piel) else np.nan,
                "metodo": metodo,
            })
        else:
            registros.append({
                "grabacion": rec_id,
                "video_path": vpath,
                "frame": f_idx,
                "clase": clase,
                "ear": np.nan, "ear_der": np.nan, "ear_izq": np.nan,
                "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan,
                "score": np.nan, "piel": np.nan,
                "metodo": "sin_rostro",
            })
        f_idx += 1

    cap.release()
    df_v = pd.DataFrame(registros)
    dt = time.time() - t0
    return rec_id, len(df_v), dt, df_v


def extraer_ear_videos(args, ruta_cache):
    if ruta_cache.exists() and not args.recalcular:
        print(f"Usando caché de EAR de videos originales (1280x720): {ruta_cache}")
        df = pd.read_csv(ruta_cache)
        return df.sort_values(["grabacion", "frame"]).reset_index(drop=True)

    pares = listar_videos_dmd(args.dir, args.max_grabaciones)
    if not pares:
        sys.exit(f"No se encontraron videos `*_rgb_face.mp4` con su `.json` en {args.dir}")

    partes = []
    if ruta_cache.exists():
        df_prev = pd.read_csv(ruta_cache)
        ct = df_prev[df_prev["metodo"] == "caja_trayectoria"].groupby("grabacion").size()
        limpias = {g for g in df_prev["grabacion"].unique() if ct.get(g, 0) < 1200}
        if limpias:
            print(f"Conservando {len(limpias)} videos ya procesados con alta cobertura directa: {sorted(limpias)}")
            partes.append(df_prev[df_prev["grabacion"].isin(limpias)])
            pares = [p for p in pares if p[0] not in limpias]

    workers = args.workers or max(1, min(8, len(pares), cpu_count() - 1))
    print(f"\nProcesando {len(pares)} videos originales HD (1280x720, sin redimensionar) con {workers} procesos...")
    t0 = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(args.modelo,)) as pool:
        for i, (rec_id, n_f, dt, df_v) in enumerate(pool.imap_unordered(procesar_video_completo, pares), 1):
            partes.append(df_v)
            print(f"  [{i:2d}/{len(pares)}] {rec_id}: {n_f} frames etiquetados procesados en {dt:.1f}s ({n_f/max(dt,0.1):.1f} fps)")

    df = pd.concat(partes, ignore_index=True).sort_values(["grabacion", "frame"]).reset_index(drop=True)
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_cache, index=False, encoding="utf-8-sig")
    print(f"Tiempo total de extracción HD: {time.time() - t0:.1f}s | Caché guardada en: {ruta_cache}")
    return df


# ---------------------------------------------------------
# Etapa 2: Reutilización del Pipeline de Eventos y Duración de AF_benchmark_DMD.py
# ---------------------------------------------------------
from AF_benchmark_DMD import (
    extraer_eventos_gt,
    obtener_referencias_calibracion,
    simular_y_extraer_eventos,
    emparejar_eventos_eb8,
    evaluar_por_umbrales_duracion,
    evaluar_por_grabacion_eventos,
    graficar_deteccion_por_duracion,
    graficar_dispersion_duraciones,
    graficar_serie_temporal,
)


def extraer_clips_parpadeos_hd(df_sim, df_eventos, dir_salida, modelo_path, n_clips=16, fps=FPS_DMD):
    """Extrae clips .avi desde los videos originales 1280x720 (`_rgb_face.mp4`) para los eventos
    con mayor error de duración."""
    vp = df_eventos[df_eventos["Tipo"] == "Verdadero Positivo"].copy()
    if vp.empty or n_clips <= 0:
        return

    _init_worker(modelo_path)
    out_dir = Path(dir_salida) / "clips_parpadeos"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_avi in out_dir.glob("*.avi"):
        try:
            old_avi.unlink()
        except OSError:
            pass

    vp["Diff_ms"] = vp["Pred_Duracion_ms"] - vp["GT_Duracion_ms"]
    n_sobre = max(1, n_clips // 2)
    n_sub = max(1, n_clips - n_sobre)
    top_sobre = vp[vp["Diff_ms"] > 0].sort_values("Error_Duracion_ms", ascending=False).head(n_sobre)
    top_sub = vp[vp["Diff_ms"] < 0].sort_values("Error_Duracion_ms", ascending=False).head(n_sub)
    seleccion = pd.concat([top_sobre, top_sub]).drop_duplicates(subset=["Grabacion", "GT_ID"])

    colores_clase = {
        "open": (60, 180, 60),
        "closing": (0, 165, 255),
        "opening": (0, 200, 255),
        "close": (40, 40, 230),
    }

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    ancho_out, alto_img = 640, 360
    alto_panel_inf = 90
    alto_out = alto_img + alto_panel_inf

    print(f"\nExtrayendo {len(seleccion)} clips de video (.avi) desde los videos originales 1280x720...")
    for _, row in seleccion.iterrows():
        g = row["Grabacion"]
        gt_id = int(row["GT_ID"])
        gt_ini, gt_fin = int(row["GT_Frame_Inicio"]), int(row["GT_Frame_Fin"])
        pr_ini, pr_fin = int(row["Pred_Frame_Inicio"]), int(row["Pred_Frame_Fin"])
        gt_ms, pr_ms = float(row["GT_Duracion_ms"]), float(row["Pred_Duracion_ms"])
        diff_ms = pr_ms - gt_ms
        err_ms = float(row["Error_Duracion_ms"])

        sub_g = df_sim[df_sim["grabacion"] == g].sort_values("frame").reset_index(drop=True)
        if sub_g.empty or "video_path" not in sub_g.columns:
            continue
        vpath = str(sub_g["video_path"].iloc[0])
        if not os.path.exists(vpath):
            continue

        f_min_clip = max(int(sub_g["frame"].min()), min(gt_ini, pr_ini) - 20)
        f_max_clip = min(int(sub_g["frame"].max()), max(gt_fin, pr_fin) + 20)
        sub_clip = sub_g[(sub_g["frame"] >= f_min_clip) & (sub_g["frame"] <= f_max_clip)].reset_index(drop=True)
        if sub_clip.empty:
            continue

        tipo_err = "SOBREESTIMADO" if diff_ms >= 0 else "SUBESTIMADO"
        nombre_avi = (f"{tipo_err}_Err_{int(round(err_ms)):04d}ms_GT{gt_id:03d}_"
                      f"{g.replace(';', '-')}_f{gt_ini}-{gt_fin}.avi")
        writer = cv2.VideoWriter(str(out_dir / nombre_avi), fourcc, fps, (ancho_out, alto_out))
        cap = cv2.VideoCapture(vpath)

        frames_arr = sub_clip["frame"].to_numpy()
        ears_arr = sub_clip["ear_eval"].to_numpy()
        thrs_arr = sub_clip["umbral"].to_numpy()
        clases_arr = sub_clip["clase"].to_numpy()
        ear_min_plot = max(0.05, float(np.nanmin(ears_arr)) - 0.04)
        ear_max_plot = min(0.50, float(np.nanmax(ears_arr)) + 0.05)
        if ear_max_plot <= ear_min_plot + 0.05:
            ear_max_plot = ear_min_plot + 0.15

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames_arr[0]))
        cur_pos = int(frames_arr[0])

        for k, r_f in sub_clip.iterrows():
            f_cur = int(r_f["frame"])
            while cur_pos < f_cur:
                cap.grab()
                cur_pos += 1
            ret, frame_hd = cap.read()
            cur_pos += 1
            if not ret:
                continue

            gray_hd = cv2.cvtColor(frame_hd, cv2.COLOR_BGR2GRAY)
            zoom = None
            if not np.isnan(r_f["x1"]):
                rect = dlib.rectangle(int(r_f["x1"]), int(r_f["y1"]), int(r_f["x2"]), int(r_f["y2"]))
                shrink = 0.88
                cx = (rect.left() + rect.right()) / 2.0
                cy = (rect.top() + rect.bottom()) / 2.0 - (1.0 - shrink) * 0.30 * rect.height()
                w, h = rect.width() * shrink, rect.height() * shrink
                r_adj = dlib.rectangle(int(round(cx - w / 2.0)), int(round(cy - h / 2.0)),
                                       int(round(cx + w / 2.0)), int(round(cy + h / 2.0)))
                _, _, _, pts = _ear_desde_caja(gray_hd, rect)

                ex1 = max(0, int(np.min(pts[36:48, 0])) - 35)
                ex2 = min(1280, int(np.max(pts[36:48, 0])) + 35)
                ey1 = max(0, int(np.min(pts[36:48, 1])) - 28)
                ey2 = min(720, int(np.max(pts[36:48, 1])) + 28)

                cv2.rectangle(frame_hd, (r_adj.left(), r_adj.top()), (r_adj.right(), r_adj.bottom()), (255, 140, 0), 2)
                for i_pt in range(36, 48):
                    px, py = int(round(pts[i_pt, 0])), int(round(pts[i_pt, 1]))
                    cv2.circle(frame_hd, (px, py), 3, (0, 0, 255), -1)
                cv2.polylines(frame_hd, [pts[36:42].astype(int)], True, (0, 255, 255), 1)
                cv2.polylines(frame_hd, [pts[42:48].astype(int)], True, (0, 255, 255), 1)

                if ex2 > ex1 + 20 and ey2 > ey1 + 12:
                    zoom = cv2.resize(frame_hd[ey1:ey2, ex1:ex2], (190, 75), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(zoom, (0, 0), (189, 74), (255, 255, 255), 1)

            vis = cv2.resize(frame_hd, (ancho_out, alto_img), interpolation=cv2.INTER_AREA)
            if zoom is not None:
                vis[8:83, ancho_out - 198:ancho_out - 8] = zoom

            c_cur = str(r_f["clase"])
            ear_cur = float(r_f["ear_eval"])
            thr_cur = float(r_f["umbral"])
            pred_cur = int(r_f["pred"])
            en_pred_span = (pr_ini <= f_cur <= pr_fin)

            cv2.rectangle(vis, (0, 0), (ancho_out - 205, 86), (20, 20, 20), -1)
            cv2.putText(vis, f"{g} (HD 1280x720) | Frame: {f_cur}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
            cv2.putText(vis, f"GT JSON: {c_cur.upper()}", (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, colores_clase.get(c_cur, (255, 255, 255)), 2)
            txt_p = "CERRADO (<=20%)" if pred_cur == 1 else ("EN EVENTO" if en_pred_span else "ABIERTO")
            col_p = (50, 50, 255) if pred_cur == 1 else ((0, 165, 255) if en_pred_span else (80, 220, 80))
            cv2.putText(vis, f"Pred: {txt_p}", (215, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col_p, 2)
            cv2.putText(vis, f"EAR: {ear_cur:.3f} | Umbral(20%): {thr_cur:.3f} | Abierto: {r_f['ear_abierto_ref']:.3f}",
                        (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(vis, f"GT 'close': {gt_ms:.0f}ms ({gt_ini}-{gt_fin}) | Pred: {pr_ms:.0f}ms ({pr_ini}-{pr_fin}) | Diff: {diff_ms:+.0f}ms",
                        (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            panel = np.full((alto_panel_inf, ancho_out, 3), 25, dtype=np.uint8)
            m_izq, m_der = 40, ancho_out - 15
            m_top, m_bot = 12, alto_panel_inf - 18
            n_f = max(1, len(frames_arr) - 1)

            def fx_pos(idx_k):
                return int(round(m_izq + (idx_k / n_f) * (m_der - m_izq)))

            def fy_pos(val_e):
                t = (val_e - ear_min_plot) / (ear_max_plot - ear_min_plot)
                t = float(np.clip(t, 0.0, 1.0))
                return int(round(m_bot - t * (m_bot - m_top)))

            for idx_k in range(len(frames_arr)):
                x_a = fx_pos(idx_k)
                x_b = fx_pos(min(len(frames_arr) - 1, idx_k + 1))
                c_k = clases_arr[idx_k]
                if c_k == "close":
                    cv2.rectangle(panel, (x_a, m_top), (max(x_a + 1, x_b), m_bot), (35, 35, 110), -1)
                elif c_k in ("closing", "opening"):
                    cv2.rectangle(panel, (x_a, m_top), (max(x_a + 1, x_b), m_bot), (25, 70, 95), -1)

            pts_thr = np.array([[fx_pos(i), fy_pos(thrs_arr[i])] for i in range(len(frames_arr))], dtype=int)
            pts_ear = np.array([[fx_pos(i), fy_pos(ears_arr[i])] for i in range(len(frames_arr))], dtype=int)
            cv2.polylines(panel, [pts_thr], False, (0, 200, 255), 1)
            cv2.polylines(panel, [pts_ear], False, (255, 220, 0), 2)

            xk = fx_pos(k)
            cv2.line(panel, (xk, 4), (xk, alto_panel_inf - 4), (255, 255, 255), 1)
            cv2.circle(panel, (xk, fy_pos(ear_cur)), 3, (0, 0, 255) if pred_cur == 1 else (0, 255, 0), -1)
            cv2.putText(panel, "Fondo Rojo=close | Naranja=closing/opening | Linea Azul=EAR | Amarilla=Umbral 20%",
                        (m_izq, alto_panel_inf - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

            writer.write(np.vstack([vis, panel]))

        cap.release()
        writer.release()
        print(f"  Guardado: {nombre_avi}")
    print(f"Clips HD exportados en: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Benchmark DMD sobre VIDEOS ORIGINALES HD (1280x720) por eventos y duración")
    ap.add_argument("--dir", default=str(DIR_DATASET_DMD), help="Carpeta dataset_dmd con los videos y JSONs")
    ap.add_argument("--salida", default=str(DIR_SALIDA), help="Carpeta de salida para resultados de videos HD")
    ap.add_argument("--modelo", default=str(MODELO_DLIB), help="Ruta a shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--workers", type=int, default=0, help="Procesos en paralelo (0 = auto)")
    ap.add_argument("--max-grabaciones", type=int, default=0, help="Limitar a N videos (0 = los 16 completos)")
    ap.add_argument("--calibracion", choices=CALIBRACIONES, default="grabacion",
                    help="Modo de calibración principal: grabacion | dataset | fija_eb8 | fija_estatica")
    ap.add_argument("--filtro", choices=MODOS_FILTRO, default="adaptativo_eb8_suav",
                    help="Modo de filtrado principal: crudo_estatico | adaptativo_eb8 | adaptativo_eb8_suav")
    ap.add_argument("--sensibilidad", type=float, default=SENSIBILIDAD_CIERRE,
                    help="0.80 -> umbral al 20%% inferior del rango de apertura")
    ap.add_argument("--umbral-ms", type=float, default=500.0,
                    help="Umbral de duración mínima de interés en ms (por defecto 500 ms)")
    ap.add_argument("--clips-error", type=int, default=16,
                    help="Cantidad de clips .avi de parpadeos con mayor error de duración a extraer")
    ap.add_argument("--recalcular", action="store_true", help="Re-extraer EAR de los videos ignorando la caché")
    args = ap.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    suf = f"_g{args.max_grabaciones}" if args.max_grabaciones else ""
    ruta_cache = salida / f"ear_videos_1280x720{suf}.csv"

    # 1. Extraer EAR de los videos originales 1280x720 sin redimensionar
    df = extraer_ear_videos(args, ruta_cache)
    print("\nCobertura de detección de rostro en videos HD (1280x720):")
    print(df["metodo"].value_counts().to_string())

    # 2. Extraer los eventos reales de cierre ('eyes_state/close') del Ground Truth JSON
    gt_blinks = extraer_eventos_gt(df, FPS_DMD)
    n_gt_foco = sum(1 for g in gt_blinks if g["duration_ms"] >= (args.umbral_ms - 1e-6))
    print(f"\nEventos reales de cierre ('eyes_state/close') en JSON (30 fps): {len(gt_blinks)} totales "
          f"| {n_gt_foco} eventos con duración >= {int(args.umbral_ms)} ms")

    # 3. Comparativa de configuraciones (Calibración x Filtro)
    filas_comp = []
    for cal in CALIBRACIONES:
        refs = obtener_referencias_calibracion(df, cal)
        for flt in MODOS_FILTRO:
            if cal == "fija_estatica" and flt == "adaptativo_eb8":
                continue
            _, pred_b = simular_y_extraer_eventos(df, refs, cal, flt, args.sensibilidad, FPS_DMD)
            df_ev, df_gt_i = emparejar_eventos_eb8(gt_blinks, pred_b, IOU_THRESHOLD, FPS_DMD)
            tabla_u = evaluar_por_umbrales_duracion(df_ev, df_gt_i, [100, int(args.umbral_ms), 1000])
            r_all, r_foco, r_1000 = tabla_u.iloc[0], tabla_u.iloc[1], tabla_u.iloc[2]
            filas_comp.append({
                "calibracion": cal,
                "filtro": flt,
                "GT_Totales": int(r_all["GT_Parpadeos (n)"]),
                "Det_Totales": int(r_all["Detectados_TP (m)"]),
                "Recall_Total": r_all["Recall_Sensibilidad"],
                f"GT_>={int(args.umbral_ms)}ms (n)": int(r_foco["GT_Parpadeos (n)"]),
                f"Det_>={int(args.umbral_ms)}ms (m)": int(r_foco["Detectados_TP (m)"]),
                f"Recall_>={int(args.umbral_ms)}ms": r_foco["Recall_Sensibilidad"],
                f"Prec_>={int(args.umbral_ms)}ms": r_foco["Precision (vs_open)"],
                f"F1_>={int(args.umbral_ms)}ms": r_foco["F1_Score (vs_open)"],
                f"MAE_>={int(args.umbral_ms)}ms": r_foco["MAE_Duracion_ms"],
                "Recall_>=1000ms": r_1000["Recall_Sensibilidad"],
            })

    df_comp = pd.DataFrame(filas_comp)
    _guardar_csv(df_comp, salida / "comparacion_configuraciones.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print(f" COMPARATIVA EN VIDEOS ORIGINALES HD (1280x720) - PARPADEOS LARGOS (>= {int(args.umbral_ms)} ms, 30 fps)")
    print("=" * 110)
    print(df_comp.to_string(index=False))

    # 4. Evaluación detallada de la configuración principal
    refs_main = obtener_referencias_calibracion(df, args.calibracion)
    df_main, pred_blinks_main = simular_y_extraer_eventos(
        df, refs_main, args.calibracion, args.filtro, args.sensibilidad, FPS_DMD
    )
    df_eventos, df_gt_info = emparejar_eventos_eb8(gt_blinks, pred_blinks_main, IOU_THRESHOLD, FPS_DMD)

    umbrales_eval = sorted(set(UMBRALES_DURACION_MS + [int(args.umbral_ms)]))
    df_umbrales = evaluar_por_umbrales_duracion(df_eventos, df_gt_info, umbrales_eval)
    df_por_grab = evaluar_por_grabacion_eventos(df_eventos, df_gt_info, args.umbral_ms)

    print("\n" + "=" * 110)
    print(f" RESULTADOS EN VIDEOS HD (1280x720) POR DURACIÓN DE PARPADEO - cal={args.calibracion} | filtro={args.filtro}")
    print("=" * 110)
    cols_u = [
        "Umbral_Min_ms", "Frames_Min_30fps", "GT_Parpadeos (n)", "Detectados_TP (m)",
        "Falsos_Negativos (FN)", "Recall_Sensibilidad", "FP_En_Ojo_Abierto",
        "Precision (vs_open)", "F1_Score (vs_open)", "MAE_Duracion_ms"
    ]
    print(df_umbrales[cols_u].to_string(index=False))

    r_foco = df_umbrales[df_umbrales["Umbral_Min_ms"] == int(args.umbral_ms)].iloc[0]
    print("\n" + "=" * 75)
    print(f" RESUMEN VIDEOS HD (1280x720): PARPADEOS PROLONGADOS (>= {int(args.umbral_ms)} ms)")
    print("=" * 75)
    print(f"Parpadeos Reales >= {int(args.umbral_ms)} ms en el dataset (n): {int(r_foco['GT_Parpadeos (n)'])}")
    print(f"Parpadeos Detectados exitosamente (m):            {int(r_foco['Detectados_TP (m)'])}")
    print(f"Falsos Negativos (no detectados):                 {int(r_foco['Falsos_Negativos (FN)'])}")
    print(f"Falsos Positivos >= {int(args.umbral_ms)} ms (en ojo abierto):     {int(r_foco['FP_En_Ojo_Abierto'])}")
    print(f"Tasa de Detección / Recall (m / n):               {r_foco['Recall_Sensibilidad']:.4f} ({r_foco['Recall_Sensibilidad']*100:.2f} %)")
    print(f"Precisión (vs ojo abierto):                       {r_foco['Precision (vs_open)']:.4f} ({r_foco['Precision (vs_open)']*100:.2f} %)")
    print(f"F1-Score:                                         {r_foco['F1_Score (vs_open)']:.4f}")
    print(f"MAE Duración de Parpadeo:                         {r_foco['MAE_Duracion_ms']:.2f} ms")
    print("=" * 75)

    print(f"\nDesglose por grabación/sujeto en videos HD (1280x720) para parpadeos >= {int(args.umbral_ms)} ms:")
    print(df_por_grab.to_string(index=False))

    _guardar_csv(df_eventos, salida / "eventos_parpadeos_dmd.csv", index=False, encoding="utf-8-sig")
    _guardar_csv(df_gt_info, salida / "ground_truth_parpadeos_dmd.csv", index=False, encoding="utf-8-sig")
    _guardar_csv(df_umbrales, salida / "metricas_por_duracion_ms.csv", index=False, encoding="utf-8-sig")
    _guardar_csv(df_por_grab, salida / "metricas_por_grabacion_eventos.csv", index=False, encoding="utf-8-sig")
    _guardar_csv(df_umbrales, salida / "metricas_resumen.csv", index=False, encoding="utf-8-sig")

    graficar_deteccion_por_duracion(df_umbrales, salida / "deteccion_por_duracion_ms.png")
    graficar_dispersion_duraciones(df_gt_info, args.umbral_ms, salida / "dispersion_duraciones_ms.png")
    graficar_serie_temporal(df_main, salida / "serie_temporal_ejemplo.png")
    if args.clips_error > 0:
        extraer_clips_parpadeos_hd(df_main, df_eventos, salida, args.modelo, args.clips_error, FPS_DMD)

    print(f"\nResultados de videos HD exportados en: {salida}")


if __name__ == "__main__":
    main()
