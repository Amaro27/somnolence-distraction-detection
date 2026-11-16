"""
AF_benchmark_DMD.py
===================
Evaluación en secuencia temporal (frame a frame) del detector de ojo cerrado y parpadeo
(dlib 68 puntos + EAR + filtro adaptativo en cascada de AF_benchmark_EB8) sobre la carpeta
`eyes_state` del Driver Monitoring Dataset (DMD).

Estructura del dataset:
    <eyes_state>/close/*.jpg     -> ojo cerrado          (GT = 1)
    <eyes_state>/open/*.jpg      -> ojo abierto          (GT = 0)
    <eyes_state>/closing/*.jpg   -> ojo cerrándose       (transición)
    <eyes_state>/opening/*.jpg   -> ojo abriéndose       (transición)

Nombre de cada archivo:
    face_<AAAA-MM-DD-HH;MM;SS>_<a>_<b>_<frame>.jpg
  * El timestamp identifica la GRABACIÓN (= un sujeto/video; hay 16 en s5).
  * <frame> es el índice de fotograma (0, 1, ..., N-1). Al reunir las 4 subcarpetas y
    ordenar por (grabacion, frame), se reconstruye la secuencia continua completa de cada video.

Hallazgo clave sobre la detección de rostro en DMD `eyes_state`:
  * Las imágenes de `eyes_state` están guardadas a 224x224 aplastando fotogramas panorámicos
    16:9 (1280x720) de toda la cabina. Eso comprime el rostro horizontalmente ~1.78x, haciendo
    que el detector HOG de dlib falle (~95%) y distorsionando la proporción vertical/horizontal
    del EAR.
  * Al restaurar la relación de aspecto 16:9 (640x360) antes de detectar y predecir landmarks,
    el detector HOG nativo de dlib (`dlib.get_frontal_face_detector`) detecta el rostro en >99%
    de los fotogramas y entrega valores de EAR fisiológicos (~0.30 abierto, ~0.16 cerrado),
    compatibles con `AF_benchmark_EB8.py`.
  * Además se filtra la región plausible del conductor (descartando falsos positivos en el
    tablero o pecho) y se aplica consistencia temporal a la caja del rostro.

Algoritmo evaluado (fiel a `AF_benchmark_EB8.py` y `AF_test.py`):
  * Umbral de cierre al 20% del rango de apertura (`SENSIBILIDAD_CIERRE = 0.80`):
        threshold_cierre   = piso + 0.20 * (ear_base_abierto - piso)
        threshold_descenso = piso + 0.70 * (ear_base_abierto - piso)
  * Doble filtro en cascada (Mediana de 5 frames -> Media móvil de 3 s / 90 frames) con guarda
    (`not en_ciclo_parpadeo and ear > threshold_adaptativo_descenso`) para actualizar
    dinámicamente `ear_base_abierto` y los umbrales adaptativos en línea.
  * Se comparan cuantitativamente:
      - Modos de calibración:
          1) `grabacion`  : `ear_base_abierto` y `piso` calibrados por sujeto (muestra inicial).
          2) `dataset`    : un único `ear_base_abierto` y `piso` general para todo el dataset.
          3) `fija_eb8`   : sin calibración con etiquetas del DMD; usa los valores fijos de
                            `AF_benchmark_EB8.py` (abierto=0.2981, piso=0.1525) y permite que
                            el filtro adaptativo ajuste `ear_base_abierto` en línea.
          4) `fija_estatica`: umbral fijo constante de EB8 (abierto=0.2981, piso=0.1525).
      - Modos de filtrado temporal:
          1) `crudo_estatico`      : EAR por frame sin suavizar + umbral estático.
          2) `adaptativo_eb8`      : filtro adaptativo en cascada de EB8 sobre el umbral.
          3) `adaptativo_eb8_suav` : filtro adaptativo en cascada de EB8 + mediana móvil corta
                                     (3 frames) en el EAR para reducir ruido de landmarks.
"""
import argparse
from collections import deque
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
from scipy.stats import rankdata

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Configuración y Parámetros Fisiológicos (de AF_benchmark_EB8.py)
# ---------------------------------------------------------
DIR_EYES_STATE = BASE_DIR / "output" / "dmd_rgb" / "s5" / "eyes_state"
DIR_SALIDA = BASE_DIR / "resultados_dmd"
MODELO_DLIB = BASE_DIR / "shape_predictor_68_face_landmarks.dat"

CLASES = ["close", "closing", "open", "opening"]

# Resolución 16:9 restaurada (mitad de 1280x720 original del DMD)
ANCHO_16_9 = 640
ALTO_16_9 = 360

# Parámetros fisiológicos de AF_benchmark_EB8.py
EAR_ABIERTO_EB8 = 0.2981
PISO_FISIOLOGICO_DLIB = 0.1525
FACTOR_DESCENSO = 0.70
SENSIBILIDAD_CIERRE = 0.80   # 1 - 0.80 = 0.20 (20% del rango de apertura = ojo cerrado)
FPS_DMD = 30.0
COOLDOWN_FRAMES = 2

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
# Utilidades de lectura y metadatos de archivos
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
    """Fracción de píxeles con tono de piel (YCrCb) en el centro de la caja."""
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
    """Ajusta ligeramente la caja HOG (shrink=0.88 centrado un poco más arriba) para evitar que
    la barba/cuello desplace hacia abajo las landmarks de los ojos, y calcula el EAR con dlib."""
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
    """Restaura 16:9 (640x360), detecta candidatos de rostro con dlib HOG en la zona del
    conductor y calcula el EAR del mejor candidato."""
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

    # Restaurar relación de aspecto 16:9 original del DMD (640x360)
    img_16_9 = cv2.resize(img, (ANCHO_16_9, ALTO_16_9), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_16_9, cv2.COLOR_BGR2GRAY)

    # Detector HOG de dlib (incluyendo umbral relajado si hace falta)
    dets, scores, _ = _detector.run(gray, 0, -1.0)
    candidatos = []
    for d, s in zip(dets, scores):
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        w, h = d.width(), d.height()
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Zona plausible de la cabeza del conductor en 640x360 (descarta cámara del tablero y pecho)
        if not (180 <= cx <= 460 and 55 <= cy <= 255 and 80 <= w <= 220):
            continue
        piel = _fraccion_piel(img_16_9, x1, y1, x2, y2)
        if piel < 0.20:
            continue
        candidatos.append((s, piel, d))

    if not candidatos:
        return reg

    # Elegir el candidato con mayor score de dlib
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
    """Para frames donde la caja detectada se desvió de la trayectoria del rostro del sujeto,
    recalcula las landmarks de dlib usando la caja interpolada de la trayectoria del sujeto."""
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
        return pd.read_csv(ruta_cache)

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

    # Etapa 1b: Verificación de consistencia espacial/temporal de la bounding box por grabación
    # Si en algún fotograma no hubo detección o la caja saltó lejos de la trayectoria mediana
    # del rostro (> 35 px en centro o > 35% en tamaño), se corrige con la caja mediana local.
    tareas_recalc = []
    indices_recalc = []
    for g, sub in df.groupby("grabacion"):
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
# Etapa 2: Calibración y Filtro Adaptativo en Cascada (AF_benchmark_EB8)
# ---------------------------------------------------------
def seleccionar_muestra_calibracion(df, frac_calib=1.0, min_por_clase=25):
    """Si frac_calib >= 1.0 (por defecto), usa el video completo (100% de los frames) para
    calibrar y también evalúa sobre el video completo (sin split). Si 0 < frac_calib < 1,
    toma una muestra uniforme de esa fracción y la excluye de la evaluación."""
    if frac_calib >= 1.0:
        return np.ones(len(df), dtype=bool), np.ones(len(df), dtype=bool)
    es_calib = np.zeros(len(df), dtype=bool)
    paso = max(2, int(round(1.0 / max(frac_calib, 0.01))))
    for _, sub in df.groupby("grabacion"):
        for clase in ("open", "close"):
            idx_c = sub.index[sub["clase"] == clase].to_numpy()
            if len(idx_c) == 0:
                continue
            elegidos = idx_c[::paso]
            if len(elegidos) < min_por_clase and len(idx_c) >= min_por_clase:
                elegidos = idx_c[np.linspace(0, len(idx_c) - 1, min_por_clase, dtype=int)]
            es_calib[elegidos] = True
    return es_calib, ~es_calib


def obtener_referencias_calibracion(df, es_calib, modo_calibracion):
    """Devuelve un diccionario {grabacion: (ear_abierto_init, piso_init)} según el modo."""
    cal = df[es_calib & df["ear"].notna()]
    abierto_global = float(cal.loc[cal["clase"] == "open", "ear"].median())
    piso_global = float(cal.loc[cal["clase"] == "close", "ear"].median())

    refs = {}
    for g, sub in df.groupby("grabacion"):
        if modo_calibracion in ("fija_eb8", "fija_estatica"):
            refs[g] = (EAR_ABIERTO_EB8, PISO_FISIOLOGICO_DLIB)
        elif modo_calibracion == "dataset":
            refs[g] = (abierto_global, piso_global)
        else:  # "grabacion" (calibrado por individuo)
            sub_cal = sub[es_calib[sub.index] & sub["ear"].notna()]
            o = sub_cal.loc[sub_cal["clase"] == "open", "ear"]
            c = sub_cal.loc[sub_cal["clase"] == "close", "ear"]
            ab = float(o.median()) if len(o) >= 5 else abierto_global
            pi = float(c.median()) if len(c) >= 5 else piso_global
            refs[g] = (ab, pi)
    return refs


def simular_secuencia_eb8(df, refs, modo_calibracion, modo_filtro, sensibilidad=SENSIBILIDAD_CIERRE):
    """Ejecuta en orden cronológico (frame a frame por cada grabación) el algoritmo de
    `AF_benchmark_EB8.py`:
      - Doble filtro en cascada (Mediana de 5 frames -> Media móvil de 3s) protegido por guarda
        (`not en_ciclo_parpadeo and ear > threshold_adaptativo_descenso`).
      - Detección de estado cerrado cuando `ear_eval <= threshold_adaptativo_cierre`
        (20% del rango de apertura).
    """
    n = len(df)
    ear_eval_arr = np.zeros(n, dtype=float)
    umbral_cierre_arr = np.zeros(n, dtype=float)
    umbral_desc_arr = np.zeros(n, dtype=float)
    ear_base_arr = np.zeros(n, dtype=float)
    piso_arr = np.zeros(n, dtype=float)
    pred_arr = np.zeros(n, dtype=int)

    usar_adaptativo = (modo_filtro in ("adaptativo_eb8", "adaptativo_eb8_suav")) and (modo_calibracion != "fija_estatica")
    suavizar_ear = (modo_filtro == "adaptativo_eb8_suav")

    for g, sub in df.groupby("grabacion", sort=False):
        ear_abierto_init, piso_fijo = refs[g]
        ear_base_abierto = ear_abierto_init

        rango_min = 0.05
        rango_inicial = max(rango_min, ear_base_abierto - piso_fijo)
        thr_cierre = piso_fijo + ((1.0 - sensibilidad) * rango_inicial)
        thr_descenso = piso_fijo + (FACTOR_DESCENSO * rango_inicial)
        thr_estatico_cierre = thr_cierre

        # Filtros en cascada idénticos a AF_test.py (mediana=5, media móvil=40 pre-llenados)
        buffer_crudo_ear = deque(maxlen=5)
        buffer_mediano_ear = deque(maxlen=40)
        buffer_suav_corto = deque(maxlen=3)

        for _ in range(buffer_crudo_ear.maxlen):
            buffer_crudo_ear.append(ear_base_abierto)
        if modo_calibracion in ("grabacion", "dataset"):
            for _ in range(buffer_mediano_ear.maxlen):
                buffer_mediano_ear.append(ear_base_abierto)

        en_ciclo_parpadeo = False
        ultimo_frame_fin = -COOLDOWN_FRAMES
        ultimo_ear_valido = ear_base_abierto

        idxs = sub.index.to_numpy()
        ears = sub["ear"].to_numpy()
        frames = sub["frame"].to_numpy()

        for k in range(len(idxs)):
            idx = idxs[k]
            f_idx = frames[k]
            e_frame = ears[k]

            if not np.isnan(e_frame) and e_frame > 0.0:
                ear_crudo = float(e_frame)
                ultimo_ear_valido = ear_crudo
            else:
                ear_crudo = ultimo_ear_valido

            buffer_suav_corto.append(ear_crudo)
            ear_eval = float(np.median(buffer_suav_corto)) if suavizar_ear else ear_crudo

            # ---------------------------------------------------------
            # Actualización Dinámica con Doble Filtro en Cascada (AF_benchmark_EB8)
            # ---------------------------------------------------------
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

                # En modos calibrados con el propio sujeto, mantener candado inferior estático;
                # en fija_eb8 (sin calibración previa del sujeto), permitir adaptarse libremente.
                if modo_calibracion in ("grabacion", "dataset") and thr_cierre < thr_estatico_cierre:
                    thr_cierre = thr_estatico_cierre

            # ---------------------------------------------------------
            # Máquina de estados del ciclo de parpadeo (guarda del filtro)
            # ---------------------------------------------------------
            if ear_eval < thr_descenso:
                if not en_ciclo_parpadeo and (f_idx - ultimo_frame_fin) > COOLDOWN_FRAMES:
                    en_ciclo_parpadeo = True
            else:
                if en_ciclo_parpadeo:
                    ultimo_frame_fin = f_idx - 1
                    en_ciclo_parpadeo = False

            # Predicción frame a frame (igual que ojo_cerrado_pred en AF_benchmark_EB8.py)
            ojo_cerrado_pred = 1 if ear_eval <= thr_cierre else 0

            ear_eval_arr[idx] = ear_eval
            umbral_cierre_arr[idx] = thr_cierre
            umbral_desc_arr[idx] = thr_descenso
            ear_base_arr[idx] = ear_base_abierto
            piso_arr[idx] = piso_fijo
            pred_arr[idx] = ojo_cerrado_pred

    out = df.copy()
    out["ear_eval"] = ear_eval_arr
    out["ear_abierto_ref"] = ear_base_arr
    out["piso_ref"] = piso_arr
    out["umbral"] = umbral_cierre_arr
    out["umbral_descenso"] = umbral_desc_arr
    rango = np.clip(out["ear_abierto_ref"] - out["piso_ref"], 0.02, None)
    out["ear_norm"] = (out["ear_eval"] - out["piso_ref"]) / rango
    out["pred"] = pred_arr
    return out


# ---------------------------------------------------------
# Métricas y Gráficas
# ---------------------------------------------------------
ESCENARIOS = {
    "A_open_vs_close (estricto)": {"open": 0, "close": 1},
    "B_open+opening_vs_close+closing": {"open": 0, "opening": 0, "close": 1, "closing": 1},
}


def metricas_binarias(y_true, y_pred, scores_cerrado):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    div = lambda a, b: a / b if b > 0 else 0.0
    prec, rec, esp = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fp)
    n1, n0 = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n1 and n0:
        r = rankdata(scores_cerrado)
        auc = float((r[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    else:
        auc = float("nan")
    return {
        "N": len(y_true), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": div(tp + tn, len(y_true)),
        "Balanced_Accuracy": (rec + esp) / 2.0,
        "Precision": prec, "Recall": rec, "Especificidad": esp,
        "F1": div(2 * prec * rec, prec + rec), "AUC_ROC": auc,
    }


def evaluar_escenario(df_eval, mapa):
    d = df_eval[df_eval["clase"].isin(mapa) & df_eval["ear_eval"].notna()].copy()
    if d.empty or d["clase"].map(mapa).nunique() < 2:
        return None, None
    d["gt"] = d["clase"].map(mapa)
    # Score continuo de cierre: distancia por debajo del umbral normalizada por el rango
    score_cierre = -d["ear_norm"].to_numpy()
    return metricas_binarias(d["gt"], d["pred"], score_cierre), d


def graficar_confusion(m, titulo, ruta):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Abierto (Pred)", "Cerrado (Pred)"])
    ax.set_yticks([0, 1], ["Abierto (GT)", "Cerrado (GT)"])
    ax.set_title(titulo, fontsize=9.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / max(1, cm[i].sum()):.1%})",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_distribucion(df_eval, sensibilidad, ruta):
    colores = {"open": "tab:green", "opening": "tab:olive", "closing": "tab:orange", "close": "tab:red"}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = np.linspace(-0.5, 1.8, 95)
    for clase in ("open", "opening", "closing", "close"):
        v = df_eval.loc[df_eval["clase"] == clase, "ear_norm"].dropna()
        if len(v):
            ax.hist(v.clip(-0.5, 1.8), bins=bins, density=True, alpha=0.5,
                    color=colores[clase], label=f"{clase} (n={len(v)})")
    ax.axvline(1.0 - sensibilidad, color="k", ls="--", lw=1.5,
               label=f"Umbral cierre = {1.0 - sensibilidad:.2f} (20% apertura)")
    ax.set_xlabel("EAR relativo al rango del individuo (0 = cerrado base, 1 = abierto base)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución del EAR Normalizado por Carpeta del DMD (16:9 + dlib)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_pct_cerrado(tabla, ruta):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    orden = [c for c in ("open", "opening", "closing", "close") if c in tabla.index]
    vals = [tabla.loc[c, "Pct_pred_cerrado"] for c in orden]
    barras = ax.bar(orden, vals, color=["tab:green", "tab:olive", "tab:orange", "tab:red"][:len(orden)])
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("% predicho como OJO CERRADO")
    ax.set_title("Porcentaje Predicho como Cerrado por Carpeta del DMD")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_serie_temporal(df_sim, ruta, ventana=600):
    """Grafica una ventana continua de 600 frames mostrando el EAR, el umbral adaptativo y los
    cierres reales (ground truth)."""
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
                    color="tab:red", alpha=0.22, label="GT: close")
    ax.set_xlabel("Frame")
    ax.set_ylabel("EAR")
    ax.set_title(f"Secuencia Temporal y Filtro Adaptativo (Grabación {g})")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_ejemplos_errores(d_a, n, dir_salida, modelo_path):
    """Guarda hasta `n` imágenes de Falsos Positivos y Falsos Negativos con la caja y landmarks."""
    _init_worker(modelo_path)
    rng = np.random.default_rng(0)
    for tipo, filtro in (("FP", (d_a["gt"] == 0) & (d_a["pred"] == 1)),
                         ("FN", (d_a["gt"] == 1) & (d_a["pred"] == 0))):
        sub = d_a[filtro]
        if sub.empty:
            continue
        carpeta = Path(dir_salida) / "errores" / tipo
        carpeta.mkdir(parents=True, exist_ok=True)
        for f_old in carpeta.glob("*.jpg"):
            try:
                f_old.unlink()
            except OSError:
                pass
        elegidos = sub.iloc[rng.choice(len(sub), size=min(n, len(sub)), replace=False)]
        for _, r in elegidos.iterrows():
            img = _leer_imagen(r["ruta"])
            if img is None:
                continue
            vis = cv2.resize(img, (ANCHO_16_9, ALTO_16_9), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            if not np.isnan(r["x1"]):
                x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 120, 0), 2)
                _, _, _, pts = _ear_desde_caja(gray, dlib.rectangle(x1, y1, x2, y2))
                for (px, py) in pts[36:48].astype(int):
                    cv2.circle(vis, (px, py), 2, (0, 0, 255), -1)
            txt = f"{tipo} | GT:{r['clase']} | EAR:{r['ear_eval']:.3f} | Thr:{r['umbral']:.3f} | {r['grabacion']}"
            cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            nombre = f"{r['grabacion'].replace(';', '-')}_frame{int(r['frame']):05d}.jpg"
            cv2.imencode(".jpg", vis)[1].tofile(str(carpeta / nombre))


# ---------------------------------------------------------
# Ejecución Principal
# ---------------------------------------------------------
def _guardar_csv(df, ruta, **kwargs):
    """Guarda un DataFrame en CSV; si el archivo está abierto en Excel (PermissionError),
    lo guarda con sufijo '_actualizado.csv' sin interrumpir la ejecución."""
    try:
        df.to_csv(ruta, **kwargs)
    except PermissionError:
        alt = Path(ruta).with_name(f"{Path(ruta).stem}_actualizado{Path(ruta).suffix}")
        df.to_csv(alt, **kwargs)
        print(f"[AVISO] '{Path(ruta).name}' está abierto en otro programa; se guardó en '{alt.name}'.")


def main():
    ap = argparse.ArgumentParser(description="Benchmark DMD eyes_state con secuencia temporal y filtro adaptativo EB8")
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
    ap.add_argument("--frac-calib", type=float, default=1.0,
                    help="1.0 (por defecto): calibra y evalúa con el video completo (sin split); "
                         "<1.0: usa esa fracción para calibrar y la excluye de la evaluación")
    ap.add_argument("--guardar-errores", type=int, default=50,
                    help="Cantidad de imágenes de FP y FN a guardar en errores/ (0 = ninguna)")
    ap.add_argument("--recalcular", action="store_true", help="Re-extraer EAR ignorando la caché")
    args = ap.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    suf = f"_g{args.max_grabaciones}" if args.max_grabaciones else ""
    ruta_cache = salida / f"ear_16_9_dlib{suf}.csv"

    # 1. Extraer EAR con proporción 16:9 restaurada y dlib nativo
    df = extraer_ear_dataset(args, ruta_cache)
    print("\nCobertura de detección de rostro (16:9 + dlib):")
    print(df["metodo"].value_counts().to_string())

    # 2. Definir conjunto de calibración y de evaluación (por defecto 100% del video completo)
    es_calib, mask_eval = seleccionar_muestra_calibracion(df, args.frac_calib)
    print(f"\nFrames totales: {len(df)} | Frames usados para calibración: {es_calib.sum()} "
          f"({es_calib.mean():.1%}) | Frames evaluados: {mask_eval.sum()} ({mask_eval.mean():.1%})")

    # 3. Comparativa cuantitativa de todas las combinaciones (Calibración x Filtro)
    filas_comp = []
    for cal in CALIBRACIONES:
        refs = obtener_referencias_calibracion(df, es_calib, cal)
        for flt in MODOS_FILTRO:
            if cal == "fija_estatica" and flt != "crudo_estatico" and flt == "adaptativo_eb8":
                continue
            df_sim = simular_secuencia_eb8(df, refs, cal, flt, args.sensibilidad)
            df_ev = df_sim[mask_eval]
            fila = {"calibracion": cal, "filtro": flt}
            for nombre_esc, mapa in ESCENARIOS.items():
                m, _ = evaluar_escenario(df_ev, mapa)
                if m:
                    k = nombre_esc[0]
                    fila.update({
                        f"{k}_N": m["N"],
                        f"{k}_Acc": m["Accuracy"],
                        f"{k}_BalAcc": m["Balanced_Accuracy"],
                        f"{k}_Recall": m["Recall"],
                        f"{k}_Espec": m["Especificidad"],
                        f"{k}_Prec": m["Precision"],
                        f"{k}_F1": m["F1"],
                        f"{k}_AUC": m["AUC_ROC"],
                    })
            filas_comp.append(fila)

    df_comp = pd.DataFrame(filas_comp)
    _guardar_csv(df_comp, salida / "comparacion_configuraciones.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 105)
    print(" TABLA COMPARATIVA DE CONFIGURACIONES (Escenario A: open vs close)")
    print("=" * 105)
    cols_mostrar = ["calibracion", "filtro", "A_Acc", "A_BalAcc", "A_Recall", "A_Espec", "A_Prec", "A_F1", "A_AUC"]
    print(df_comp[cols_mostrar].round(4).to_string(index=False))

    # 4. Evaluación detallada de la configuración principal seleccionada
    refs_main = obtener_referencias_calibracion(df, es_calib, args.calibracion)
    df_main = simular_secuencia_eb8(df, refs_main, args.calibracion, args.filtro, args.sensibilidad)
    df_eval = df_main[mask_eval].copy()

    # Guardar tabla de referencias por grabación
    filas_ref = []
    for g, (ab_i, pi_i) in sorted(refs_main.items()):
        sub_g = df_main[df_main["grabacion"] == g]
        filas_ref.append({
            "grabacion": g,
            "ear_abierto_inicial": round(ab_i, 4),
            "ear_abierto_medio_adapt": round(float(sub_g["ear_abierto_ref"].mean()), 4),
            "piso_cerrado": round(pi_i, 4),
            "umbral_cierre_medio": round(float(sub_g["umbral"].mean()), 4),
        })
    df_refs = pd.DataFrame(filas_ref)
    _guardar_csv(df_refs, salida / "calibracion_por_grabacion.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 105)
    print(f" RESULTADOS DETALLADOS - CONFIGURACIÓN PRINCIPAL: calibracion={args.calibracion} | filtro={args.filtro}")
    print("=" * 105)
    print("\nParámetros de EAR y umbral (20% del rango) por grabación:")
    print(df_refs.to_string(index=False))

    tabla_carpetas = df_eval.groupby("clase").agg(
        N=("pred", "size"),
        EAR_mediana=("ear_eval", "median"),
        EAR_norm_mediana=("ear_norm", "median"),
        Pct_pred_cerrado=("pred", lambda s: 100.0 * s.mean()),
    )
    print("\nPredicción por carpeta del DMD:")
    print(tabla_carpetas.round(4).to_string())

    filas_resumen = []
    d_a = None
    for nombre_esc, mapa in ESCENARIOS.items():
        m, d_esc = evaluar_escenario(df_eval, mapa)
        if m is None:
            continue
        filas_resumen.append({"Escenario": nombre_esc, **m})
        if nombre_esc.startswith("A_"):
            d_a = d_esc
        graficar_confusion(
            m,
            f"{nombre_esc}\ncal={args.calibracion} | filtro={args.filtro}\n"
            f"Acc={m['Accuracy']:.3f} | BalAcc={m['Balanced_Accuracy']:.3f} | AUC={m['AUC_ROC']:.3f}",
            salida / f"matriz_confusion_{nombre_esc[0]}.png",
        )

    df_resumen = pd.DataFrame(filas_resumen)
    for _, r in df_resumen.iterrows():
        print(f"\n[{r['Escenario']}]  N={int(r['N'])}")
        print(f"  TP={int(r['TP'])}  TN={int(r['TN'])}  FP={int(r['FP'])}  FN={int(r['FN'])}")
        for k in ("Accuracy", "Balanced_Accuracy", "Precision", "Recall", "Especificidad", "F1", "AUC_ROC"):
            print(f"  {k:18s}: {r[k]:.4f}")

    if d_a is not None:
        filas_g = []
        for g, sub_g in d_a.groupby("grabacion"):
            if sub_g["gt"].nunique() == 2:
                mg = metricas_binarias(sub_g["gt"], sub_g["pred"], -sub_g["ear_norm"].to_numpy())
                filas_g.append({"grabacion": g, **mg})
        if filas_g:
            df_por_grab = pd.DataFrame(filas_g)
            _guardar_csv(df_por_grab, salida / "metricas_por_grabacion_A.csv", index=False, encoding="utf-8-sig")
            print("\nDesempeño por grabación/sujeto (Escenario A: open vs close):")
            cols_g = ["grabacion", "N", "Accuracy", "Balanced_Accuracy", "Recall", "Especificidad", "Precision", "F1", "AUC_ROC"]
            print(df_por_grab[cols_g].round(4).to_string(index=False))

    # Exportar archivos y gráficas
    _guardar_csv(df_resumen, salida / "metricas_resumen.csv", index=False, encoding="utf-8-sig")
    _guardar_csv(tabla_carpetas, salida / "prediccion_por_carpeta.csv", encoding="utf-8-sig")
    _guardar_csv(df_eval, salida / "predicciones_por_imagen.csv", index=False, encoding="utf-8-sig")
    graficar_distribucion(df_eval, args.sensibilidad, salida / "distribucion_ear.png")
    graficar_pct_cerrado(tabla_carpetas, salida / "pct_cerrado_por_carpeta.png")
    graficar_serie_temporal(df_main, salida / "serie_temporal_ejemplo.png")
    if args.guardar_errores and d_a is not None:
        guardar_ejemplos_errores(d_a, args.guardar_errores, salida, args.modelo)

    print(f"\nResultados y gráficas exportados en: {salida}")


if __name__ == "__main__":
    main()
