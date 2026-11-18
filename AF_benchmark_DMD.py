"""
AF_benchmark_DMD.py
===================
Evaluación basada en EVENTOS Y DURACIÓN DE PARPADEOS (alineada con `AF_benchmark_EB8.py` y
`AF_test.py`) sobre la carpeta `eyes_state` del Driver Monitoring Dataset (DMD), filmado a 30 fps.

Objetivo y Criterio de Evaluación:
  * Solo se considera como cierre ocular efectivo la carpeta `close` (oclusión profunda, igual
    que el atributo 'C' en `AF_benchmark_EB8.py`). No se combina `close` con `closing`.
  * A partir de las imágenes ordenadas cronológicamente por `(grabacion, frame)` a 30 fps:
      1) Se extraen todos los eventos reales de cierre (`gt_blinks`) con su frame de inicio,
         frame de fin y duración en milisegundos: `duration_ms = (frames / 30.0) * 1000.0`.
      2) Se ejecuta el algoritmo de `AF_test.py` / `AF_benchmark_EB8.py` (umbral al 20% del
         rango de apertura + doble filtro adaptativo en cascada Mediana -> Media Móvil) para
         extraer los eventos de cierre detectados (`pred_blinks`) y su duración en milisegundos.
      3) Se evalúa cuántos parpadeos de `n` milisegundos en adelante (p. ej. >= 500 ms, que es
         el umbral de parpadeo prolongado / microsueño en `AF_test.py`) existen en el dataset y
         cuántos fueron detectados exitosamente, calculando TP, FN, FP, Recall, Precisión,
         F1-Score y MAE de duración (ms) para cada umbral de duración.
"""
import argparse
from collections import deque
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
DIR_EYES_STATE = BASE_DIR / "output" / "dmd_rgb" / "s5" / "eyes_state"
DIR_SALIDA = BASE_DIR / "resultados_dmd"
MODELO_DLIB = BASE_DIR / "shape_predictor_68_face_landmarks.dat"

CLASES = ["close", "closing", "open", "opening"]

# Resolución 16:9 restaurada (mitad de 1280x720 original del DMD)
ANCHO_16_9 = 640
ALTO_16_9 = 360

# Parámetros fisiológicos de AF_benchmark_EB8.py y AF_test.py
FPS_DMD = 30.0
EAR_ABIERTO_EB8 = 0.2981
PISO_FISIOLOGICO_DLIB = 0.1525
FACTOR_DESCENSO = 0.70
SENSIBILIDAD_CIERRE = 0.80   # 1 - 0.80 = 0.20 (20% del rango de apertura = ojo cerrado)
DURACION_MIN_FRAMES = 2      # mínimo 2 frames (66.7 ms a 30 fps), igual que AF_benchmark_EB8.py
COOLDOWN_FRAMES = 2          # tolerancia de unión entre frames cerrados consecutivos (66.7 ms)
IOU_THRESHOLD = 0.20         # umbral de solapamiento temporal de AF_benchmark_EB8.py

# Cortes de duración mínima (ms) para evaluar detección de parpadeos de n ms en adelante
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


# ---------------------------------------------------------
# Utilidades de lectura y exportación
# ---------------------------------------------------------
_RE_NOMBRE = re.compile(r"face_(.+?)_(\d+)_(\d+)_(\d+)\.\w+$")


def meta_nombre(ruta):
    """Extrae (grabacion, frame) del nombre face_<timestamp>_<a>_<b>_<frame>.jpg."""
    m = _RE_NOMBRE.search(os.path.basename(ruta))
    return (m.group(1), int(m.group(4))) if m else ("desconocida", -1)


def _es_puntero_lfs(ruta):
    try:
        if os.path.getsize(ruta) > 1024:
            return False
        with open(ruta, "rb") as f:
            return f.read(24).startswith(b"version https://git-")
    except OSError:
        return False


def _leer_imagen(ruta):
    try:
        return cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def _guardar_csv(df, ruta, **kwargs):
    """Guarda un DataFrame en CSV; si el archivo está abierto en otro programa, usa sufijo."""
    try:
        df.to_csv(ruta, **kwargs)
    except PermissionError:
        alt = Path(ruta).with_name(f"{Path(ruta).stem}_actualizado{Path(ruta).suffix}")
        df.to_csv(alt, **kwargs)
        print(f"[AVISO] '{Path(ruta).name}' está abierto en otro programa; se guardó en '{alt.name}'.")


def listar_imagenes(dir_eyes_state, max_grabaciones=0):
    tareas = []
    for clase in CLASES:
        carpeta = Path(dir_eyes_state) / clase
        if not carpeta.is_dir():
            print(f"[AVISO] No existe la carpeta: {carpeta}")
            continue
        tareas += [(e.path, clase) for e in os.scandir(carpeta)
                   if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg", ".png"))]
    if max_grabaciones:
        recs = sorted({meta_nombre(r)[0] for r, _ in tareas})[:max_grabaciones]
        tareas = [t for t in tareas if meta_nombre(t[0])[0] in recs]
    tareas.sort(key=lambda t: t[0])
    print("  " + ", ".join(f"{c}: {sum(1 for _, k in tareas if k == c)}" for c in CLASES))
    return tareas


# ---------------------------------------------------------
# Etapa 1: Detección de rostro 16:9 con dlib y cálculo de EAR
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


def procesar_imagen_16_9(arg):
    ruta, clase = arg
    reg = {
        "ruta": ruta, "clase": clase,
        "ear": np.nan, "ear_der": np.nan, "ear_izq": np.nan,
        "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan,
        "score": np.nan, "piel": np.nan, "metodo": "sin_rostro"
    }
    if _es_puntero_lfs(ruta):
        reg["metodo"] = "puntero_lfs"
        return reg
    img = _leer_imagen(ruta)
    if img is None:
        reg["metodo"] = "error_lectura"
        return reg

    img_16_9 = cv2.resize(img, (ANCHO_16_9, ALTO_16_9), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_16_9, cv2.COLOR_BGR2GRAY)

    dets, scores, _ = _detector.run(gray, 0, -1.0)
    candidatos = []
    for d, s in zip(dets, scores):
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        w = d.width()
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not (180 <= cx <= 460 and 55 <= cy <= 255 and 80 <= w <= 220):
            continue
        piel = _fraccion_piel(img_16_9, x1, y1, x2, y2)
        if piel < 0.20:
            continue
        candidatos.append((s, piel, d))

    if not candidatos:
        return reg

    s, piel, mejor_rect = max(candidatos, key=lambda t: t[0])
    ear, ear_der, ear_izq, _ = _ear_desde_caja(gray, mejor_rect)
    reg.update({
        "ear": round(float(ear), 5),
        "ear_der": round(float(ear_der), 5),
        "ear_izq": round(float(ear_izq), 5),
        "x1": int(mejor_rect.left()),
        "y1": int(mejor_rect.top()),
        "x2": int(mejor_rect.right()),
        "y2": int(mejor_rect.bottom()),
        "score": round(float(s), 3),
        "piel": round(float(piel), 3),
        "metodo": "dlib_estandar" if s >= 0.0 else "dlib_relajado",
    })
    return reg


def _recalcular_caja_fija(arg):
    ruta, x1, y1, x2, y2 = arg
    img = _leer_imagen(ruta)
    if img is None:
        return ruta, np.nan, np.nan, np.nan
    gray = cv2.cvtColor(cv2.resize(img, (ANCHO_16_9, ALTO_16_9), interpolation=cv2.INTER_CUBIC),
                        cv2.COLOR_BGR2GRAY)
    rect = dlib.rectangle(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    ear, ear_der, ear_izq, _ = _ear_desde_caja(gray, rect)
    return ruta, round(float(ear), 5), round(float(ear_der), 5), round(float(ear_izq), 5)


def extraer_ear_dataset(args, ruta_cache):
    if ruta_cache.exists() and not args.recalcular:
        print(f"Usando caché de EAR 16:9: {ruta_cache}")
        df = pd.read_csv(ruta_cache)
        return df.sort_values(["grabacion", "frame"]).reset_index(drop=True)

    print(f"Listando imágenes en {args.dir}...")
    tareas = listar_imagenes(args.dir, args.max_grabaciones)
    if not tareas:
        sys.exit("No se encontraron imágenes.")

    muestra = tareas[:: max(1, len(tareas) // 200)]
    if sum(_es_puntero_lfs(r) for r, _ in muestra) > 0.5 * len(muestra):
        sys.exit(
            "\n[ERROR] Las imágenes son punteros de Git LFS (~130 bytes), no JPG reales.\n"
            "        Descárgalas con `git lfs pull` desde el repo original."
        )

    workers = args.workers or max(1, min(8, cpu_count() - 1))
    print(f"\nEtapa 1a: Extrayendo rostro y EAR (16:9 + dlib) de {len(tareas)} imágenes con {workers} procesos...")
    t0 = time.time()
    registros = []
    with Pool(workers, initializer=_init_worker, initargs=(args.modelo,)) as pool:
        for i, reg in enumerate(pool.imap(procesar_imagen_16_9, tareas, chunksize=32), 1):
            registros.append(reg)
            if i % 5000 == 0 or i == len(tareas):
                v = time.time() - t0
                print(f"  {i}/{len(tareas)}  ({i / v:.1f} img/s, ETA {(len(tareas) - i) / (i / v):.0f}s)")

    df = pd.DataFrame(registros)
    meta = df["ruta"].map(meta_nombre)
    df["grabacion"] = meta.map(lambda t: t[0])
    df["frame"] = meta.map(lambda t: t[1])
    df = df.sort_values(["grabacion", "frame"]).reset_index(drop=True)

    tareas_recalc = []
    indices_recalc = []
    for _, sub in df.groupby("grabacion"):
        cx = (sub["x1"] + sub["x2"]) / 2.0
        cy = (sub["y1"] + sub["y2"]) / 2.0
        w = sub["x2"] - sub["x1"]
        h = sub["y2"] - sub["y1"]

        cx_med = cx.rolling(31, min_periods=3, center=True).median().bfill().ffill()
        cy_med = cy.rolling(31, min_periods=3, center=True).median().bfill().ffill()
        w_med = w.rolling(31, min_periods=3, center=True).median().bfill().ffill()
        h_med = h.rolling(31, min_periods=3, center=True).median().bfill().ffill()

        dist_centro = np.hypot(cx - cx_med, cy - cy_med)
        cambio_tam = (w - w_med).abs() / w_med.clip(lower=1.0)
        invalida = sub["ear"].isna() | (dist_centro > 20.0) | (cambio_tam > 0.15)

        for idx in sub.index[invalida]:
            if np.isnan(cx_med.loc[idx]):
                continue
            bx1 = cx_med.loc[idx] - w_med.loc[idx] / 2.0
            by1 = cy_med.loc[idx] - h_med.loc[idx] / 2.0
            bx2 = cx_med.loc[idx] + w_med.loc[idx] / 2.0
            by2 = cy_med.loc[idx] + h_med.loc[idx] / 2.0
            tareas_recalc.append((df.loc[idx, "ruta"], bx1, by1, bx2, by2))
            indices_recalc.append((idx, int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))))

    if tareas_recalc:
        print(f"Etapa 1b: Corrigiendo {len(tareas_recalc)} fotogramas con caja fuera de trayectoria o sin detección...")
        with Pool(workers, initializer=_init_worker, initargs=(args.modelo,)) as pool:
            res_recalc = list(pool.imap(_recalcular_caja_fija, tareas_recalc, chunksize=32))
        for (idx, bx1, by1, bx2, by2), (_, ear, ear_der, ear_izq) in zip(indices_recalc, res_recalc):
            df.loc[idx, ["ear", "ear_der", "ear_izq"]] = [ear, ear_der, ear_izq]
            df.loc[idx, ["x1", "y1", "x2", "y2"]] = [bx1, by1, bx2, by2]
            df.loc[idx, "metodo"] = "caja_trayectoria"

    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_cache, index=False, encoding="utf-8-sig")
    print(f"Caché guardada en: {ruta_cache}")
    return df


# ---------------------------------------------------------
# Etapa 2: Extracción de Eventos de Ground Truth (`close`) y Simulación Temporal
# ---------------------------------------------------------
def extraer_eventos_gt(df, fps=FPS_DMD):
    """Extrae los eventos individuales de parpadeo/cierre del Ground Truth (estrictamente la
    carpeta `close`) y calcula su duración en frames y en milisegundos a 30 fps."""
    gt_blinks = []
    for g, sub in df.groupby("grabacion", sort=False):
        frames = sub["frame"].to_numpy()
        clases = sub["clase"].to_numpy()
        actual = []
        for f_idx, c in zip(frames, clases):
            f_idx = int(f_idx)
            if c == "close":
                if actual and f_idx != actual[-1] + 1:
                    dur_f = len(actual)
                    gt_blinks.append({
                        "grabacion": g,
                        "start": actual[0],
                        "end": actual[-1],
                        "frames": dur_f,
                        "duration_ms": (dur_f / fps) * 1000.0,
                    })
                    actual = [f_idx]
                else:
                    actual.append(f_idx)
            else:
                if actual:
                    dur_f = len(actual)
                    gt_blinks.append({
                        "grabacion": g,
                        "start": actual[0],
                        "end": actual[-1],
                        "frames": dur_f,
                        "duration_ms": (dur_f / fps) * 1000.0,
                    })
                    actual = []
        if actual:
            dur_f = len(actual)
            gt_blinks.append({
                "grabacion": g,
                "start": actual[0],
                "end": actual[-1],
                "frames": dur_f,
                "duration_ms": (dur_f / fps) * 1000.0,
            })
    return gt_blinks


def obtener_referencias_calibracion(df, modo_calibracion):
    """Obtiene (ear_abierto_init, piso_init) por grabación usando el video completo."""
    val = df[df["ear"].notna()]
    abierto_global = float(val.loc[val["clase"] == "open", "ear"].median())
    piso_global = float(val.loc[val["clase"] == "close", "ear"].median())

    refs = {}
    for g, sub in df.groupby("grabacion"):
        if modo_calibracion in ("fija_eb8", "fija_estatica"):
            refs[g] = (EAR_ABIERTO_EB8, PISO_FISIOLOGICO_DLIB)
        elif modo_calibracion == "dataset":
            refs[g] = (abierto_global, piso_global)
        else:  # "grabacion" (calibrado por individuo en su video)
            sub_v = sub[sub["ear"].notna()]
            o = sub_v.loc[sub_v["clase"] == "open", "ear"]
            c = sub_v.loc[sub_v["clase"] == "close", "ear"]
            ab = float(o.median()) if len(o) >= 5 else abierto_global
            pi = float(c.median()) if len(c) >= 5 else piso_global
            refs[g] = (ab, pi)
    return refs


def simular_y_extraer_eventos(df, refs, modo_calibracion, modo_filtro,
                              sensibilidad=SENSIBILIDAD_CIERRE, fps=FPS_DMD):
    """Simula el algoritmo de `AF_test.py` / `AF_benchmark_EB8.py` sobre cada grabación continua
    y extrae la lista de parpadeos/cierres predichos (`pred_blinks`) con su duración en ms."""
    n = len(df)
    ear_eval_arr = np.zeros(n, dtype=float)
    umbral_cierre_arr = np.zeros(n, dtype=float)
    umbral_desc_arr = np.zeros(n, dtype=float)
    ear_base_arr = np.zeros(n, dtype=float)
    piso_arr = np.zeros(n, dtype=float)
    pred_arr = np.zeros(n, dtype=int)

    usar_adaptativo = (modo_filtro in ("adaptativo_eb8", "adaptativo_eb8_suav")) and (modo_calibracion != "fija_estatica")
    suavizar_ear = (modo_filtro == "adaptativo_eb8_suav")

    pred_blinks = []

    for g, sub in df.groupby("grabacion", sort=False):
        ear_abierto_init, piso_fijo = refs[g]
        ear_base_abierto = ear_abierto_init

        rango_min = 0.05
        rango_inicial = max(rango_min, ear_base_abierto - piso_fijo)
        thr_cierre = piso_fijo + ((1.0 - sensibilidad) * rango_inicial)
        thr_descenso = piso_fijo + (FACTOR_DESCENSO * rango_inicial)
        thr_estatico_cierre = thr_cierre

        # Filtros en cascada idénticos a AF_test.py (Mediana=5, Media móvil=40)
        buffer_crudo_ear = deque([ear_base_abierto] * 5, maxlen=5)
        buffer_mediano_ear = deque(maxlen=40)
        if modo_calibracion in ("grabacion", "dataset"):
            for _ in range(buffer_mediano_ear.maxlen):
                buffer_mediano_ear.append(ear_base_abierto)
        buffer_suav_corto = deque(maxlen=5)

        en_ciclo_parpadeo = False
        ultimo_ear_valido = ear_base_abierto

        idxs = sub.index.to_numpy()
        ears = sub["ear"].to_numpy()
        frames = sub["frame"].to_numpy()
        clases = sub["clase"].to_numpy()
        f_to_clase = dict(zip(frames.astype(int), clases))

        frames_cerrados = []

        for k in range(len(idxs)):
            idx = idxs[k]
            f_idx = int(frames[k])
            e_frame = ears[k]

            if not np.isnan(e_frame) and e_frame > 0.0:
                ear_crudo = float(e_frame)
                ultimo_ear_valido = ear_crudo
            else:
                ear_crudo = ultimo_ear_valido

            buffer_suav_corto.append(ear_crudo)
            ear_eval = float(np.median(buffer_suav_corto)) if suavizar_ear else ear_crudo

            # Actualización dinámica con doble filtro en cascada (AF_test / AF_benchmark_EB8)
            if usar_adaptativo:
                if not en_ciclo_parpadeo and ear_crudo > thr_descenso:
                    buffer_crudo_ear.append(ear_crudo)
                    if len(buffer_crudo_ear) == buffer_crudo_ear.maxlen:
                        ear_mediano = float(np.median(buffer_crudo_ear))
                        buffer_mediano_ear.append(ear_mediano)
                        if len(buffer_mediano_ear) >= 15:
                            ear_base_abierto = float(np.mean(buffer_mediano_ear))
                            rango_movil = max(rango_min, ear_base_abierto - piso_fijo)
                            thr_cierre = piso_fijo + ((1.0 - sensibilidad) * rango_movil)
                            thr_descenso = piso_fijo + (FACTOR_DESCENSO * rango_movil)

                if modo_calibracion in ("grabacion", "dataset") and thr_cierre < thr_estatico_cierre:
                    thr_cierre = thr_estatico_cierre

            en_ciclo_parpadeo = (ear_eval < thr_descenso)
            ojo_cerrado_pred = 1 if ear_eval <= thr_cierre else 0

            if ojo_cerrado_pred == 1:
                frames_cerrados.append(f_idx)

            ear_eval_arr[idx] = ear_eval
            umbral_cierre_arr[idx] = thr_cierre
            umbral_desc_arr[idx] = thr_descenso
            ear_base_arr[idx] = ear_base_abierto
            piso_arr[idx] = piso_fijo
            pred_arr[idx] = ojo_cerrado_pred

        # Consolidar fotogramas cerrados en eventos individuales (uniendo huecos <= COOLDOWN_FRAMES)
        if frames_cerrados:
            ini = fin = frames_cerrados[0]
            for f_c in frames_cerrados[1:]:
                if (f_c - fin - 1) <= COOLDOWN_FRAMES:
                    fin = f_c
                else:
                    dur_f = fin - ini + 1
                    if dur_f >= DURACION_MIN_FRAMES:
                        clases_ev = [f_to_clase.get(f, "open") for f in range(ini, fin + 1)]
                        pred_blinks.append({
                            "grabacion": g,
                            "start": ini,
                            "end": fin,
                            "frames": dur_f,
                            "duration_ms": (dur_f / fps) * 1000.0,
                            "frac_open": sum(1 for c in clases_ev if c == "open") / len(clases_ev),
                        })
                    ini = fin = f_c
            dur_f = fin - ini + 1
            if dur_f >= DURACION_MIN_FRAMES:
                clases_ev = [f_to_clase.get(f, "open") for f in range(ini, fin + 1)]
                pred_blinks.append({
                    "grabacion": g,
                    "start": ini,
                    "end": fin,
                    "frames": dur_f,
                    "duration_ms": (dur_f / fps) * 1000.0,
                    "frac_open": sum(1 for c in clases_ev if c == "open") / len(clases_ev),
                })

    out = df.copy()
    out["ear_eval"] = ear_eval_arr
    out["ear_abierto_ref"] = ear_base_arr
    out["piso_ref"] = piso_arr
    out["umbral"] = umbral_cierre_arr
    out["umbral_descenso"] = umbral_desc_arr
    rango = np.clip(out["ear_abierto_ref"] - out["piso_ref"], 0.02, None)
    out["ear_norm"] = (out["ear_eval"] - out["piso_ref"]) / rango
    out["pred"] = pred_arr
    return out, pred_blinks


# ---------------------------------------------------------
# Etapa 3: Emparejamiento de Eventos y Evaluación por Duración (>= n ms)
# ---------------------------------------------------------
def emparejar_eventos_eb8(gt_blinks, pred_blinks, iou_threshold=IOU_THRESHOLD, fps=FPS_DMD):
    """Empareja los eventos predichos con los eventos reales de `close` dentro de cada grabación:
      - Punto 2 (Consolidación de fragmentos): Si un mismo cierre real (GT) es solapado por 2 o más
        sub-segmentos predichos (por temblor momentáneo del EAR en cierres largos), se consolidan
        desde el primer fotograma cerrado (`frame_inicio_cierre`) hasta el último (`frame_fin_cierre`),
        tal como `dur_f = frame_fin_cierre - frame_inicio_cierre + 1` en `AF_benchmark_EB8.py`.
      - Punto 3 (Criterio estricto IoU >= 0.20): Se exige estrictamente `iou >= iou_threshold` (0.20),
        sin aceptar emparejamientos con IoU bajo por cobertura.
    """
    datos_exportacion = []
    gt_info = []
    preds_usados = set()

    # Agrupar predicciones por grabación para búsqueda rápida
    preds_por_grab = {}
    for p_idx, p in enumerate(pred_blinks):
        preds_por_grab.setdefault(p["grabacion"], []).append((p_idx, p))

    rec_id_counter = 1
    for gt_idx, gt in enumerate(gt_blinks):
        g = gt["grabacion"]
        gt_set = set(range(gt["start"], gt["end"] + 1))
        candidatos = preds_por_grab.get(g, [])

        # Buscar todos los segmentos predichos no usados que se solapen con este evento GT
        solapados = []
        for p_idx, p in candidatos:
            if p_idx in preds_usados:
                continue
            p_set = set(range(p["start"], p["end"] + 1))
            if len(gt_set & p_set) > 0:
                solapados.append((p_idx, p, p_set))

        if solapados:
            # Consolidar fragmentos que caen en el mismo cierre real (inicio del primero al fin del último)
            span_start = min(p["start"] for _, p, _ in solapados)
            span_end = max(p["end"] for _, p, _ in solapados)
            span_frames = span_end - span_start + 1
            span_dur_ms = (span_frames / fps) * 1000.0
            span_set = set(range(span_start, span_end + 1))

            inter = len(gt_set & span_set)
            union = len(gt_set | span_set)
            iou = inter / union if union > 0 else 0.0

            # Punto 3: Criterio estricto IoU >= 0.20 (idéntico a AF_benchmark_EB8.py)
            if iou >= iou_threshold:
                for p_idx, _, _ in solapados:
                    preds_usados.add(p_idx)
                err = abs(span_dur_ms - gt["duration_ms"])
                datos_exportacion.append({
                    "Grabacion": g,
                    "Pred_ID": rec_id_counter,
                    "Pred_Frame_Inicio": span_start,
                    "Pred_Frame_Fin": span_end,
                    "Pred_Frames": span_frames,
                    "Pred_Duracion_ms": round(span_dur_ms, 2),
                    "Fragmentos_Unidos": len(solapados),
                    "Tipo": "Verdadero Positivo",
                    "GT_ID": gt_idx + 1,
                    "GT_Frame_Inicio": gt["start"],
                    "GT_Frame_Fin": gt["end"],
                    "GT_Frames": gt["frames"],
                    "GT_Duracion_ms": round(gt["duration_ms"], 2),
                    "Error_Duracion_ms": round(err, 2),
                    "IoU": round(iou, 4),
                    "En_Transicion_Sin_Close": False,
                })
                rec_id_counter += 1
                gt_info.append({
                    **gt,
                    "gt_id": gt_idx + 1,
                    "detectado": True,
                    "pred_duration_ms": span_dur_ms,
                    "error_ms": err,
                    "iou": iou,
                    "fragmentos_unidos": len(solapados),
                })
                continue

        # Si no hubo predicción con IoU >= 0.20, es un Falso Negativo
        gt_info.append({
            **gt,
            "gt_id": gt_idx + 1,
            "detectado": False,
            "pred_duration_ms": np.nan,
            "error_ms": np.nan,
            "iou": 0.0,
            "fragmentos_unidos": 0,
        })
        datos_exportacion.append({
            "Grabacion": g,
            "Pred_ID": None,
            "Pred_Frame_Inicio": None,
            "Pred_Frame_Fin": None,
            "Pred_Frames": None,
            "Pred_Duracion_ms": None,
            "Fragmentos_Unidos": 0,
            "Tipo": "Falso Negativo",
            "GT_ID": gt_idx + 1,
            "GT_Frame_Inicio": gt["start"],
            "GT_Frame_Fin": gt["end"],
            "GT_Frames": gt["frames"],
            "GT_Duracion_ms": round(gt["duration_ms"], 2),
            "Error_Duracion_ms": None,
            "IoU": 0.0,
            "En_Transicion_Sin_Close": False,
        })

    # Registrar las predicciones no emparejadas como Falsos Positivos
    for p_idx, pred in enumerate(pred_blinks):
        if p_idx in preds_usados:
            continue
        datos_exportacion.append({
            "Grabacion": pred["grabacion"],
            "Pred_ID": rec_id_counter,
            "Pred_Frame_Inicio": pred["start"],
            "Pred_Frame_Fin": pred["end"],
            "Pred_Frames": pred["frames"],
            "Pred_Duracion_ms": round(pred["duration_ms"], 2),
            "Fragmentos_Unidos": 1,
            "Tipo": "Falso Positivo",
            "GT_ID": None,
            "GT_Frame_Inicio": None,
            "GT_Frame_Fin": None,
            "GT_Frames": None,
            "GT_Duracion_ms": None,
            "Error_Duracion_ms": None,
            "IoU": None,
            "En_Transicion_Sin_Close": pred.get("frac_open", 1.0) < 0.5,
        })
        rec_id_counter += 1

    return pd.DataFrame(datos_exportacion), pd.DataFrame(gt_info)


def evaluar_por_umbrales_duracion(df_eventos, df_gt_info, umbrales_ms=UMBRALES_DURACION_MS):
    """Para cada umbral `min_ms` (ej. 500 ms en adelante), calcula:
      - GT_Eventos (n)       : cantidad de parpadeos reales en `close` con duración >= min_ms
      - Detectados_TP (m)    : cuántos de esos n parpadeos reales fueron detectados exitosamente
      - No_Detectados_FN     : cuántos de esos n parpadeos reales NO fueron detectados
      - FP_Totales           : eventos predichos >= min_ms que no corresponden a ningún `close`
      - FP_En_Ojo_Abierto    : eventos predichos >= min_ms ocurridos en carpeta `open` (excluyendo
                               parpadeos reales etiquetados solo como `closing`/`opening`)
      - Recall (m / n), Precisión, F1-Score y MAE de duración (ms).
    """
    filas = []
    fps_df = df_eventos[df_eventos["Tipo"] == "Falso Positivo"]

    for min_ms in umbrales_ms:
        # 1e-6 evita problemas de punto flotante (ej. 500.0 ms exactos = 15/30*1000)
        sub_gt = df_gt_info[df_gt_info["duration_ms"] >= (min_ms - 1e-6)]
        n_gt = len(sub_gt)
        tp = int(sub_gt["detectado"].sum())
        fn = n_gt - tp

        sub_fp = fps_df[fps_df["Pred_Duracion_ms"] >= (min_ms - 1e-6)]
        fp_total = len(sub_fp)
        fp_open = int((~sub_fp["En_Transicion_Sin_Close"]).sum())

        recall = tp / n_gt if n_gt > 0 else 0.0
        prec_estricta = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
        f1_estricto = (2 * prec_estricta * recall / (prec_estricta + recall)
                       if (prec_estricta + recall) > 0 else 0.0)

        prec_open = tp / (tp + fp_open) if (tp + fp_open) > 0 else 0.0
        f1_open = (2 * prec_open * recall / (prec_open + recall)
                   if (prec_open + recall) > 0 else 0.0)

        errores = sub_gt.loc[sub_gt["detectado"], "error_ms"].dropna()
        mae_ms = float(errores.mean()) if len(errores) > 0 else 0.0
        med_err_ms = float(errores.median()) if len(errores) > 0 else 0.0

        filas.append({
            "Umbral_Min_ms": int(min_ms),
            "Frames_Min_30fps": int(round(min_ms / 1000.0 * FPS_DMD)),
            "GT_Parpadeos (n)": n_gt,
            "Detectados_TP (m)": tp,
            "Falsos_Negativos (FN)": fn,
            "Recall_Sensibilidad": round(recall, 4),
            "FP_En_Ojo_Abierto": fp_open,
            "Precision (vs_open)": round(prec_open, 4),
            "F1_Score (vs_open)": round(f1_open, 4),
            "FP_Incluyendo_Transicion": fp_total,
            "Precision_Estricta": round(prec_estricta, 4),
            "F1_Estricto": round(f1_estricto, 4),
            "MAE_Duracion_ms": round(mae_ms, 2),
            "Mediana_Error_Duracion_ms": round(med_err_ms, 2),
        })
    return pd.DataFrame(filas)


def evaluar_por_grabacion_eventos(df_eventos, df_gt_info, umbral_ms=500):
    """Desglosa por cada una de las 16 grabaciones cuántos parpadeos >= umbral_ms hubo y cuántos
    se detectaron."""
    fps_df = df_eventos[(df_eventos["Tipo"] == "Falso Positivo") &
                        (df_eventos["Pred_Duracion_ms"] >= (umbral_ms - 1e-6))]
    filas = []
    for g, sub_gt_all in df_gt_info.groupby("grabacion"):
        sub_gt = sub_gt_all[sub_gt_all["duration_ms"] >= (umbral_ms - 1e-6)]
        n_gt = len(sub_gt)
        tp = int(sub_gt["detectado"].sum())
        fn = n_gt - tp
        sub_fp = fps_df[fps_df["Grabacion"] == g]
        fp_open = int((~sub_fp["En_Transicion_Sin_Close"]).sum())
        fp_tot = len(sub_fp)

        rec = tp / n_gt if n_gt > 0 else 0.0
        prec = tp / (tp + fp_open) if (tp + fp_open) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        errs = sub_gt.loc[sub_gt["detectado"], "error_ms"].dropna()
        filas.append({
            "grabacion": g,
            f"GT_>={int(umbral_ms)}ms (n)": n_gt,
            "Detectados_TP (m)": tp,
            "FN": fn,
            "FP_open": fp_open,
            "FP_total": fp_tot,
            "Recall": round(rec, 4),
            "Precision": round(prec, 4),
            "F1_Score": round(f1, 4),
            "MAE_Duracion_ms": round(float(errs.mean()), 1) if len(errs) else 0.0,
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------
# Gráficas de Eventos y Duración
# ---------------------------------------------------------
def graficar_deteccion_por_duracion(df_umbrales, ruta):
    """Grafica barras con n (parpadeos reales >= T ms) vs m (detectados) y la curva de Recall/F1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    etiquetas = [f">={u}" for u in df_umbrales["Umbral_Min_ms"]]
    x = np.arange(len(etiquetas))
    w = 0.38

    b1 = ax1.bar(x - w / 2, df_umbrales["GT_Parpadeos (n)"], w, label="Parpadeos Reales GT (n)", color="tab:gray", alpha=0.75)
    b2 = ax1.bar(x + w / 2, df_umbrales["Detectados_TP (m)"], w, label="Detectados con Éxito (m)", color="tab:blue")
    for b in b1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"{int(b.get_height())}",
                 ha="center", fontsize=8)
    for b in b2:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"{int(b.get_height())}",
                 ha="center", fontsize=8, color="tab:blue", fontweight="bold")
    ax1.set_xticks(x, etiquetas, rotation=30)
    ax1.set_xlabel("Duración mínima del parpadeo (ms)")
    ax1.set_ylabel("Cantidad de eventos de parpadeo")
    ax1.set_title("Parpadeos Reales (n) vs. Detectados (m) por Duración")
    ax1.set_ylim(0, max(df_umbrales["GT_Parpadeos (n)"]) * 1.15)
    ax1.legend(fontsize=8.5)

    ax2.plot(etiquetas, df_umbrales["Recall_Sensibilidad"] * 100.0, "o-", lw=2, color="tab:green",
             label="Recall / Tasa de Detección (m/n %)")
    ax2.plot(etiquetas, df_umbrales["Precision (vs_open)"] * 100.0, "s-", lw=2, color="tab:blue",
             label="Precisión (vs open %)")
    ax2.plot(etiquetas, df_umbrales["F1_Score (vs_open)"] * 100.0, "^-", lw=2, color="tab:orange",
             label="F1-Score (%)")
    ax2.set_xticks(x, etiquetas, rotation=30)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Duración mínima del parpadeo (ms)")
    ax2.set_ylabel("Porcentaje (%)")
    ax2.set_title("Métricas por Umbral de Duración Mínima")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_dispersion_duraciones(df_gt_info, umbral_ms, ruta):
    """Compara la duración real (ms) vs la duración medida (ms) en los parpadeos detectados."""
    det = df_gt_info[df_gt_info["detectado"]].copy()
    if det.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    cortos = det[det["duration_ms"] < umbral_ms]
    largos = det[det["duration_ms"] >= umbral_ms]

    if len(cortos):
        ax.scatter(cortos["duration_ms"], cortos["pred_duration_ms"], alpha=0.5, color="tab:gray",
                   label=f"< {int(umbral_ms)} ms (n={len(cortos)})")
    if len(largos):
        ax.scatter(largos["duration_ms"], largos["pred_duration_ms"], alpha=0.7, color="tab:red",
                   label=f">= {int(umbral_ms)} ms (n={len(largos)})")

    lim = max(det["duration_ms"].max(), det["pred_duration_ms"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="Ideal (Pred = GT)")
    ax.axvline(umbral_ms, color="tab:red", ls=":", lw=1.2, label=f"Umbral {int(umbral_ms)} ms")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Duración Real en Ground Truth (ms)")
    ax.set_ylabel("Duración Detectada por el Algoritmo (ms)")
    ax.set_title("Duración Real vs. Detectada por Evento de Parpadeo")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_serie_temporal(df_sim, ruta, ventana=600):
    mejor = None
    for g, sub in df_sim.groupby("grabacion"):
        cerr = (sub["clase"] == "close").astype(int).to_numpy()
        if len(cerr) < ventana:
            continue
        s = np.convolve(cerr, np.ones(ventana), mode="valid")
        if mejor is None or s.max() > mejor[0]:
            mejor = (s.max(), g, int(s.argmax()))
    if mejor is None:
        return
    _, g, ini = mejor
    sub = df_sim[df_sim["grabacion"] == g].iloc[ini:ini + ventana]
    fr = sub["frame"].to_numpy()

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.plot(fr, sub["ear"].to_numpy(), color="lightgray", lw=0.9, label="EAR crudo")
    ax.plot(fr, sub["ear_eval"].to_numpy(), color="tab:blue", lw=1.4, label="EAR evaluado")
    ax.plot(fr, sub["ear_abierto_ref"].to_numpy(), color="tab:green", ls=":", lw=1.2,
            label="EAR base abierto (adaptativo)")
    ax.plot(fr, sub["umbral"].to_numpy(), "k--", lw=1.3, label="Umbral adaptativo cierre (20%)")
    cerr = (sub["clase"] == "close").to_numpy()
    ax.fill_between(fr, 0, 1, where=cerr, transform=ax.get_xaxis_transform(),
                    color="tab:red", alpha=0.22, label="GT: close (cierre efectivo)")
    ax.set_xlabel("Frame (30 fps)")
    ax.set_ylabel("EAR")
    ax.set_title(f"Secuencia Temporal y Duración de Parpadeos (Grabación {g})")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------
# Ejecución Principal
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Benchmark DMD eyes_state por eventos y duración de parpadeos (30 fps)")
    ap.add_argument("--dir", default=str(DIR_EYES_STATE), help="Carpeta eyes_state del DMD")
    ap.add_argument("--salida", default=str(DIR_SALIDA), help="Carpeta de salida para resultados")
    ap.add_argument("--modelo", default=str(MODELO_DLIB), help="Ruta a shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--workers", type=int, default=0, help="Procesos en paralelo (0 = auto)")
    ap.add_argument("--max-grabaciones", type=int, default=0, help="Limitar a N grabaciones (0 = las 16 completas)")
    ap.add_argument("--calibracion", choices=CALIBRACIONES, default="grabacion",
                    help="Modo de calibración principal: grabacion | dataset | fija_eb8 | fija_estatica")
    ap.add_argument("--filtro", choices=MODOS_FILTRO, default="adaptativo_eb8_suav",
                    help="Modo de filtrado principal: crudo_estatico | adaptativo_eb8 | adaptativo_eb8_suav")
    ap.add_argument("--sensibilidad", type=float, default=SENSIBILIDAD_CIERRE,
                    help="0.80 -> umbral al 20%% inferior del rango de apertura")
    ap.add_argument("--umbral-ms", type=float, default=500.0,
                    help="Umbral de duración mínima de interés en ms (por defecto 500 ms, como en AF_test.py)")
    ap.add_argument("--recalcular", action="store_true", help="Re-extraer EAR ignorando la caché")
    args = ap.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    suf = f"_g{args.max_grabaciones}" if args.max_grabaciones else ""
    ruta_cache = salida / f"ear_16_9_dlib{suf}.csv"

    # 1. Cargar/extraer EAR en secuencia temporal continua (30 fps)
    df = extraer_ear_dataset(args, ruta_cache)

    # 2. Extraer todos los eventos reales de cierre ('close') del Ground Truth
    gt_blinks = extraer_eventos_gt(df, FPS_DMD)
    n_gt_foco = sum(1 for g in gt_blinks if g["duration_ms"] >= (args.umbral_ms - 1e-6))
    print(f"\nEventos reales de cierre ('close') en Ground Truth (30 fps): {len(gt_blinks)} totales "
          f"| {n_gt_foco} eventos con duración >= {int(args.umbral_ms)} ms")

    # 3. Comparativa de configuraciones (Calibración x Filtro) para parpadeos >= umbral_ms
    filas_comp = []
    for cal in CALIBRACIONES:
        refs = obtener_referencias_calibracion(df, cal)
        for flt in MODOS_FILTRO:
            if cal == "fija_estatica" and flt == "adaptativo_eb8":
                continue
            _, pred_b = simular_y_extraer_eventos(df, refs, cal, flt, args.sensibilidad, FPS_DMD)
            df_ev, df_gt_i = emparejar_eventos_eb8(gt_blinks, pred_b, IOU_THRESHOLD)
            tabla_u = evaluar_por_umbrales_duracion(df_ev, df_gt_i, [100, int(args.umbral_ms), 1000])
            r_all = tabla_u.iloc[0]
            r_foco = tabla_u.iloc[1]
            r_1000 = tabla_u.iloc[2]
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
    print(f" COMPARATIVA DE CONFIGURACIONES EN DETECCIÓN DE PARPADEOS LARGOS (>= {int(args.umbral_ms)} ms, 30 fps)")
    print("=" * 110)
    print(df_comp.to_string(index=False))

    # 4. Evaluación detallada de la configuración principal
    refs_main = obtener_referencias_calibracion(df, args.calibracion)
    df_main, pred_blinks_main = simular_y_extraer_eventos(
        df, refs_main, args.calibracion, args.filtro, args.sensibilidad, FPS_DMD
    )
    df_eventos, df_gt_info = emparejar_eventos_eb8(gt_blinks, pred_blinks_main, IOU_THRESHOLD)

    umbrales_eval = sorted(set(UMBRALES_DURACION_MS + [int(args.umbral_ms)]))
    df_umbrales = evaluar_por_umbrales_duracion(df_eventos, df_gt_info, umbrales_eval)
    df_por_grab = evaluar_por_grabacion_eventos(df_eventos, df_gt_info, args.umbral_ms)

    print("\n" + "=" * 110)
    print(f" RESULTADOS POR DURACIÓN DE PARPADEO (de n ms en adelante) - cal={args.calibracion} | filtro={args.filtro}")
    print("=" * 110)
    cols_u = [
        "Umbral_Min_ms", "Frames_Min_30fps", "GT_Parpadeos (n)", "Detectados_TP (m)",
        "Falsos_Negativos (FN)", "Recall_Sensibilidad", "FP_En_Ojo_Abierto",
        "Precision (vs_open)", "F1_Score (vs_open)", "MAE_Duracion_ms"
    ]
    print(df_umbrales[cols_u].to_string(index=False))

    # Resumen específico para el umbral de interés (por defecto >= 500 ms)
    r_foco = df_umbrales[df_umbrales["Umbral_Min_ms"] == int(args.umbral_ms)].iloc[0]
    print("\n" + "=" * 75)
    print(f" RESUMEN FOCO: PARPADEOS PROLONGADOS (>= {int(args.umbral_ms)} ms, equivalente a AF_test.py)")
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

    print(f"\nDesglose por grabación/sujeto para parpadeos >= {int(args.umbral_ms)} ms:")
    print(df_por_grab.to_string(index=False))

    # Exportar tablas CSV y gráficas
    _guardar_csv(df_eventos, salida / "eventos_parpadeos_dmd.csv", index=False, encoding="utf-8-sig")
    _guardar_csv(df_gt_info, salida / "ground_truth_parpadeos_dmd.csv", index=False, encoding="utf-8-sig")
    _guardar_csv(df_umbrales, salida / "metricas_por_duracion_ms.csv", index=False, encoding="utf-8-sig")
    _guardar_csv(df_por_grab, salida / "metricas_por_grabacion_eventos.csv", index=False, encoding="utf-8-sig")
    _guardar_csv(df_umbrales, salida / "metricas_resumen.csv", index=False, encoding="utf-8-sig")

    graficar_deteccion_por_duracion(df_umbrales, salida / "deteccion_por_duracion_ms.png")
    graficar_dispersion_duraciones(df_gt_info, args.umbral_ms, salida / "dispersion_duraciones_ms.png")
    graficar_serie_temporal(df_main, salida / "serie_temporal_ejemplo.png")

    print(f"\nArchivos exportados en: {salida}")
    print("  - metricas_por_duracion_ms.csv      (tabla de detección desde n ms en adelante)")
    print("  - eventos_parpadeos_dmd.csv         (lista detallada de cada evento estilo EB8)")
    print("  - ground_truth_parpadeos_dmd.csv    (los 285 eventos reales de 'close' y si se detectaron)")
    print("  - metricas_por_grabacion_eventos.csv(desempeño por sujeto en >= 500 ms)")
    print("  - deteccion_por_duracion_ms.png y dispersion_duraciones_ms.png")


if __name__ == "__main__":
    main()
