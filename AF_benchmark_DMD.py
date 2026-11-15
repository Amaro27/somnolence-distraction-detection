"""
AF_benchmark_DMD.py
===================
Evaluación por IMAGEN del detector de ojo cerrado (dlib 68 landmarks + EAR) sobre la
carpeta `eyes_state` del Driver Monitoring Dataset (DMD).

Estructura esperada:
    <eyes_state>/close/*.jpg      -> ojo cerrado
    <eyes_state>/closing/*.jpg    -> ojo cerrándose   (transición)
    <eyes_state>/open/*.jpg       -> ojo abierto
    <eyes_state>/opening/*.jpg    -> ojo abriéndose   (transición)

Pipeline (2 etapas):
  1) EXTRACCIÓN: para cada imagen se localiza el rostro, se calcula el EAR con las landmarks de
     dlib y se guarda en un CSV (caché). Es la etapa lenta; se hace en paralelo y solo una vez.
     NOTA: las imágenes de esta carpeta son frames completos de cabina a 224x224 (rostro de
     ~50 px), no recortes de cara. El HOG de dlib solo detecta ~5 %, por eso la caja de rostro
     sale por defecto de una cascada Haar (~95 %); las landmarks siguen siendo de dlib.
  2) EVALUACIÓN: el umbral es individual por GRABACIÓN (= sujeto; hay 16), con la misma
     fórmula que AF_benchmark_EB8: piso + (1 - SENSIBILIDAD) * (EAR_abierto - piso), donde
     EAR_abierto y piso son las medianas de EAR de una muestra de open / close de ese sujeto
     (excluida de la evaluación). Se predice "cerrado" si EAR <= umbral y se compara contra la
     etiqueta de la carpeta.

Como no hay secuencia temporal en la evaluación por imagen, NO se usan los filtros
(mediana / media móvil) ni el umbral adaptativo en línea: el umbral es estático por sujeto.

Ground truth (escenarios reportados):
  A) Binario estricto : open (0) vs close (1). closing/opening se excluyen. (Principal)
  B) Con transiciones : open+opening (0) vs close+closing (1).
  Además se reporta, para las 4 carpetas, el % de imágenes predichas como cerradas.

Uso rápido:
    python AF_benchmark_DMD.py                       # todo el dataset
    python AF_benchmark_DMD.py --limite 300          # prueba rápida (300 imgs por carpeta)
    python AF_benchmark_DMD.py --calibracion dataset # un solo umbral global (peor: hay 16 sujetos)
    python AF_benchmark_DMD.py --recalcular          # fuerza re-extraer EAR
"""
import argparse
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
# Configuración por defecto
# ---------------------------------------------------------
DIR_EYES_STATE = BASE_DIR / "output" / "dmd_rgb" / "s5" / "eyes_state"
DIR_SALIDA = BASE_DIR / "resultados_dmd"
MODELO_DLIB = BASE_DIR / "shape_predictor_68_face_landmarks.dat"

CLASES = ["close", "closing", "open", "opening"]

# Valores calibrados en AF_benchmark_EB8 (modo --calibracion fija)
EAR_ABIERTO_EB8 = 0.2981
PISO_FISIOLOGICO_DLIB = 0.1525

# Mismo criterio que EB8: SENSIBILIDAD 0.80 -> el ojo se considera cerrado cuando
# su EAR está en el 20 % inferior del rango [piso, EAR_abierto] del individuo.
SENSIBILIDAD_CIERRE = 0.80

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
# Etapa 1: extracción de EAR por imagen
# ---------------------------------------------------------
def _init_worker(modelo, detector_modo, escala):
    global _detector, _predictor, _haar, _detector_modo, _escala
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(str(modelo))
    _haar = [cv2.CascadeClassifier(cv2.data.haarcascades + n)
             for n in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")]
    _detector_modo = detector_modo
    _escala = escala


def _es_puntero_lfs(ruta):
    """Un puntero Git LFS es un texto de ~130 bytes que empieza con 'version https://git-lfs'."""
    try:
        if os.path.getsize(ruta) > 1024:
            return False
        with open(ruta, "rb") as f:
            return f.read(24).startswith(b"version https://git-")
    except OSError:
        return False


def _detectar_rostro(gray):
    """Devuelve (rect, metodo) en coordenadas de `gray`.

    OJO: las imágenes de esta carpeta del DMD NO son recortes de cara, sino frames completos
    de la cabina reducidos a 224x224 (rostro de ~50 px), por lo que el HOG de dlib casi nunca
    detecta rostro (~5 %). Modo 'cascada' (por defecto): Haar (rápido, ~95 %) -> Haar alt2 ->
    dlib con umbral relajado. Modo 'dlib': solo el detector HOG original (upsample 1).
    Las landmarks siempre las calcula el predictor de dlib de 68 puntos."""
    if _detector_modo == "dlib":
        rostros = _detector(gray, 1)
        if len(rostros) > 0:
            return max(rostros, key=lambda x: x.area()), "dlib_hog"
        return None, "sin_rostro"
    for h, nombre in zip(_haar, ("haar_default", "haar_alt2")):
        hs = h.detectMultiScale(gray, 1.05, 3, minSize=(int(20 * _escala),) * 2)
        if len(hs) > 0:
            x, y, w, hh = max(hs, key=lambda t: t[2] * t[3])
            return dlib.rectangle(int(x), int(y), int(x + w), int(y + hh)), nombre
    dets, scores, _ = _detector.run(gray, 0, -0.6)
    if len(dets) > 0:
        return dets[int(np.argmax(scores))], "dlib_hog_relajado"
    return None, "sin_rostro"


def procesar_imagen(args):
    ruta, clase = args
    reg = {"ruta": ruta, "clase": clase, "ear": np.nan, "ear_der": np.nan,
           "ear_izq": np.nan, "metodo": "error_lectura"}
    if _es_puntero_lfs(ruta):
        reg["metodo"] = "puntero_lfs"
        return reg
    # imdecode+fromfile: robusto a rutas con ';' / caracteres especiales en Windows
    try:
        data = np.fromfile(ruta, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        img = None
    if img is None:
        return reg

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if _escala != 1:  # el rostro mide ~50 px: se amplía para detectar y para las landmarks
        gray = cv2.resize(gray, None, fx=_escala, fy=_escala, interpolation=cv2.INTER_CUBIC)
    rect, metodo = _detectar_rostro(gray)
    reg["metodo"] = metodo
    if rect is None:
        return reg

    shape = _predictor(gray, rect)
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype="int")
    ear_der = calcular_ear(pts[R_START:R_END])
    ear_izq = calcular_ear(pts[L_START:L_END])
    reg["ear_der"] = ear_der
    reg["ear_izq"] = ear_izq
    reg["ear"] = (ear_der + ear_izq) / 2.0
    return reg


def listar_imagenes(dir_eyes_state, limite, semilla):
    rng = np.random.default_rng(semilla)
    tareas = []
    for clase in CLASES:
        carpeta = Path(dir_eyes_state) / clase
        if not carpeta.is_dir():
            print(f"[AVISO] No existe la carpeta: {carpeta}")
            continue
        archivos = sorted(e.path for e in os.scandir(carpeta)
                          if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg", ".png")))
        if limite and len(archivos) > limite:
            idx = np.sort(rng.choice(len(archivos), size=limite, replace=False))
            archivos = [archivos[i] for i in idx]
        print(f"  {clase:8s}: {len(archivos):6d} imágenes")
        tareas += [(a, clase) for a in archivos]
    return tareas


def extraer_ears(args, ruta_cache):
    if ruta_cache.exists() and not args.recalcular:
        print(f"Usando caché de EAR: {ruta_cache}  (usa --recalcular para rehacerla)")
        return pd.read_csv(ruta_cache)

    print(f"Listando imágenes en {args.dir}")
    tareas = listar_imagenes(args.dir, args.limite, args.semilla)
    if not tareas:
        sys.exit("No se encontraron imágenes.")

    # Verificación temprana de punteros Git LFS (evita procesar 88k archivos inútiles)
    muestra = tareas[:: max(1, len(tareas) // 200)]
    n_ptr = sum(_es_puntero_lfs(r) for r, _ in muestra)
    if n_ptr > 0.5 * len(muestra):
        sys.exit(
            "\n[ERROR] Las imágenes son punteros de Git LFS (~130 bytes), no JPG reales.\n"
            "        Descarga los archivos reales, por ejemplo:\n"
            "            git lfs install\n"
            '            git lfs pull --include="output/dmd_rgb/s5/eyes_state/**"\n'
            "        (o copia las imágenes originales del DMD a esa carpeta) y vuelve a ejecutar."
        )

    workers = args.workers or max(1, min(8, cpu_count() - 1))
    print(f"\nExtrayendo EAR de {len(tareas)} imágenes con {workers} procesos...")
    t0 = time.time()
    registros = []
    with Pool(workers, initializer=_init_worker,
              initargs=(args.modelo, args.detector, args.escala)) as pool:
        for i, reg in enumerate(pool.imap(procesar_imagen, tareas, chunksize=32), 1):
            registros.append(reg)
            if i % 2000 == 0 or i == len(tareas):
                v = time.time() - t0
                print(f"  {i}/{len(tareas)}  ({i / v:.1f} img/s, ETA {(len(tareas) - i) / (i / v):.0f}s)")

    df = pd.DataFrame(registros)
    # metadatos del nombre: face_<fecha>_<...>_<frame>.jpg
    nombre = df["ruta"].map(lambda p: Path(p).stem)
    df["frame"] = nombre.map(lambda s: int(m.group(1)) if (m := re.search(r"_(\d+)$", s)) else -1)
    df["sesion"] = nombre.map(lambda s: s.rsplit("_", 1)[0])
    # timestamp de la grabación (= un sujeto/video): face_<AAAA-MM-DD-HH;MM;SS>_<a>_<b>_<frame>
    df["grabacion"] = nombre.map(
        lambda s: m.group(1) if (m := re.match(r"face_(.+?)_\d+_\d+_\d+$", s)) else "desconocida")
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_cache, index=False, encoding="utf-8-sig")
    print(f"Caché de EAR guardada en: {ruta_cache}")
    return df


# ---------------------------------------------------------
# Etapa 2: umbral y métricas
# ---------------------------------------------------------
def _muestra_calibracion(df, frac, rng, minimo=5):
    """Índices de una muestra aleatoria de open y close (se excluyen luego de la evaluación)."""
    idx_total = []
    for clase in ("open", "close"):
        idx = df.index[df["clase"] == clase].to_numpy()
        if len(idx) == 0:
            continue
        n = min(len(idx), max(minimo, int(round(len(idx) * frac))))
        idx_total += list(rng.choice(idx, size=n, replace=False))
    return idx_total


def _referencias(calib):
    """(EAR abierto, piso) = mediana de EAR en open / close; None si falta alguna clase."""
    o = calib.loc[calib["clase"] == "open", "ear"]
    c = calib.loc[calib["clase"] == "close", "ear"]
    if len(o) == 0 or len(c) == 0:
        return None
    return float(o.median()), float(c.median())


def calcular_umbral(df, args):
    """Asigna a cada imagen su umbral individual.

    Devuelve (df_eval, tabla_calibracion, descripcion). df_eval incluye las columnas:
      ear_abierto, piso, umbral (= piso + (1-sensibilidad)*rango) y ear_norm
      (EAR normalizado por sujeto: 0 = piso/cerrado, 1 = abierto).
    Modos:
      fija      : valores de EB8 para todos.
      dataset   : un único par (abierto, piso) global.
      grabacion : un par por grabación (= sujeto), el equivalente a la calibración individual
                  del sistema en tiempo real. Es el modo recomendado: este dataset mezcla
                  varios sujetos con EAR base muy distintos.
    """
    rng = np.random.default_rng(args.semilla)
    df = df.copy()
    if args.calibracion == "fija":
        df["ear_abierto"], df["piso"] = EAR_ABIERTO_EB8, PISO_FISIOLOGICO_DLIB
        df_eval, tabla = df, pd.DataFrame()
        desc = "fija (valores calibrados en EB8)"
    else:
        calib_idx = _muestra_calibracion(df, args.frac_calib, rng)
        ref_global = _referencias(df.loc[calib_idx])
        df["ear_abierto"], df["piso"] = ref_global
        filas = [{"grabacion": "GLOBAL", "n_calib": len(calib_idx),
                  "ear_abierto": ref_global[0], "piso": ref_global[1]}]
        if args.calibracion == "grabacion":
            for g, sub in df.groupby("grabacion"):
                idx_g = _muestra_calibracion(sub, args.frac_calib, rng)
                ref = _referencias(df.loc[idx_g])
                if ref is None or len(idx_g) < 10:
                    ref, usado = ref_global, "global (pocos datos)"
                else:
                    calib_idx += [i for i in idx_g if i not in set(calib_idx)]
                    usado = "propia"
                df.loc[sub.index, "ear_abierto"] = ref[0]
                df.loc[sub.index, "piso"] = ref[1]
                filas.append({"grabacion": g, "n_calib": len(idx_g), "ear_abierto": ref[0],
                              "piso": ref[1], "referencia": usado})
        tabla = pd.DataFrame(filas)
        df_eval = df.drop(index=sorted(set(calib_idx)))
        desc = (f"{args.calibracion}: mediana EAR open/close con {args.frac_calib:.0%} de cada "
                f"clase ({len(set(calib_idx))} imágenes de calibración, excluidas de la evaluación)")
    rango = (df_eval["ear_abierto"] - df_eval["piso"]).clip(lower=0.02)
    df_eval = df_eval.copy()
    df_eval["umbral"] = df_eval["piso"] + (1.0 - args.sensibilidad) * rango
    df_eval["ear_norm"] = (df_eval["ear"] - df_eval["piso"]) / rango
    return df_eval, tabla, desc


def metricas_binarias(y_true, y_pred, scores_cerrado):
    """Clase positiva = ojo cerrado (1)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    div = lambda a, b: a / b if b > 0 else 0.0
    prec, rec, esp = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fp)
    # AUC independiente del umbral (Mann-Whitney)
    n1, n0 = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n1 and n0:
        r = rankdata(scores_cerrado)
        auc = (r[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    else:
        auc = float("nan")
    return {
        "N": len(y_true), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": div(tp + tn, len(y_true)),
        "Balanced_Accuracy": (rec + esp) / 2.0,
        "Precision": prec, "Recall": rec, "Especificidad": esp,
        "F1": div(2 * prec * rec, prec + rec), "AUC_ROC": auc,
    }


def mejor_umbral_oraculo(y_true, ear):
    """Umbral que maximiza balanced accuracy (SOLO diagnóstico: usa las etiquetas de evaluación)."""
    y_true, ear = np.asarray(y_true), np.asarray(ear)
    cand = np.unique(np.quantile(ear, np.linspace(0.01, 0.99, 197)))
    mejor, mejor_ba = cand[0], -1
    for t in cand:
        pred = (ear <= t).astype(int)
        rec = ((pred == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())
        esp = ((pred == 0) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())
        if (rec + esp) / 2 > mejor_ba:
            mejor, mejor_ba = t, (rec + esp) / 2
    return float(mejor), float(mejor_ba)


def graficar_confusion(m, titulo, ruta):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    etiquetas = ["Abierto", "Cerrado"]
    ax.set_xticks([0, 1], etiquetas)
    ax.set_yticks([0, 1], etiquetas)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Ground truth (DMD)")
    ax.set_title(titulo, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / max(1, cm[i].sum()):.1%})", ha="center",
                    va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_distribucion(df_eval, sensibilidad, ruta):
    """Distribución del EAR normalizado por sujeto (0 = piso/cerrado, 1 = abierto)."""
    colores = {"open": "tab:green", "opening": "tab:olive", "closing": "tab:orange", "close": "tab:red"}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = np.linspace(-0.5, 2.0, 100)
    for clase in ("open", "opening", "closing", "close"):
        v = df_eval.loc[df_eval["clase"] == clase, "ear_norm"].dropna()
        if len(v):
            ax.hist(v.clip(-0.5, 2.0), bins=bins, density=True, alpha=0.5, color=colores[clase],
                    label=f"{clase} (n={len(v)})")
    ax.axvline(1.0 - sensibilidad, color="k", ls="--",
               label=f"Umbral de cierre = {1.0 - sensibilidad:.2f} del rango")
    ax.set_xlabel("EAR normalizado por sujeto  (0 = ojo cerrado típico, 1 = ojo abierto típico)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución del EAR normalizado por carpeta del DMD")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_pct_cerrado(tabla, ruta):
    fig, ax = plt.subplots(figsize=(6, 4))
    orden = [c for c in ("open", "opening", "closing", "close") if c in tabla.index]
    vals = [tabla.loc[c, "Pct_pred_cerrado"] for c in orden]
    barras = ax.bar(orden, vals, color=["tab:green", "tab:olive", "tab:orange", "tab:red"][:len(orden)])
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")
    ax.set_ylim(0, 110)
    ax.set_ylabel("% predichas como OJO CERRADO")
    ax.set_title("Predicción del sistema por carpeta del DMD")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_errores(df_a, n, dir_salida):
    """Guarda ejemplos anotados de falsos positivos / negativos (equivalente a los clips de FP de EB8)."""
    rng = np.random.default_rng(0)
    for tipo, filtro in (("FP", (df_a["gt"] == 0) & (df_a["pred"] == 1)),
                         ("FN", (df_a["gt"] == 1) & (df_a["pred"] == 0))):
        sub = df_a[filtro]
        if sub.empty:
            continue
        carpeta = Path(dir_salida) / "errores" / tipo
        carpeta.mkdir(parents=True, exist_ok=True)
        for _, r in sub.iloc[rng.choice(len(sub), size=min(n, len(sub)), replace=False)].iterrows():
            img = cv2.imdecode(np.fromfile(r["ruta"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            txt = f"{tipo} GT={r['clase']} EAR={r['ear']:.3f} thr={r['umbral']:.3f}"
            cv2.putText(img, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            nombre = f"{r['ear']:.3f}_{Path(r['ruta']).stem.replace(';', '-')}.jpg"
            cv2.imencode(".jpg", img)[1].tofile(str(carpeta / nombre))


def main():
    ap = argparse.ArgumentParser(description="Benchmark EAR/dlib por imagen sobre DMD eyes_state")
    ap.add_argument("--dir", default=str(DIR_EYES_STATE), help="carpeta eyes_state")
    ap.add_argument("--salida", default=str(DIR_SALIDA), help="carpeta de resultados")
    ap.add_argument("--modelo", default=str(MODELO_DLIB), help="shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--workers", type=int, default=0, help="procesos paralelos (0 = auto)")
    ap.add_argument("--limite", type=int, default=0, help="máx. imágenes por carpeta (0 = todas)")
    ap.add_argument("--calibracion", choices=["grabacion", "dataset", "fija"], default="grabacion",
                    help="grabacion: umbral individual por sujeto (recomendado); "
                         "dataset: un solo umbral global; fija: valores de EB8")
    ap.add_argument("--frac-calib", type=float, default=0.10,
                    help="fracción de open/close usada para calibrar (se excluye de la evaluación)")
    ap.add_argument("--sensibilidad", type=float, default=SENSIBILIDAD_CIERRE,
                    help="0.80 -> umbral al 20%% inferior del rango del individuo")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--detector", choices=["cascada", "dlib"], default="cascada",
                    help="cascada: Haar -> dlib relajado (recomendado para este dataset); "
                         "dlib: solo el HOG de dlib")
    ap.add_argument("--escala", type=float, default=3.0,
                    help="factor de ampliación de la imagen antes de detectar/predecir landmarks")
    ap.add_argument("--recalcular", action="store_true", help="ignorar caché de EAR")
    ap.add_argument("--guardar-errores", type=int, default=20,
                    help="nº de ejemplos de FP/FN a guardar (0 = ninguno)")
    args = ap.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    nombre_cache = "ear_por_imagen.csv" if not args.limite else f"ear_por_imagen_limite{args.limite}.csv"
    df = extraer_ears(args, salida / nombre_cache)

    # --- cobertura de detección ---
    print("\nMétodo de localización de rostro:")
    print(df["metodo"].value_counts().to_string())
    df_valido = df[df["ear"].notna()].reset_index(drop=True)
    n_sin = len(df) - len(df_valido)
    if n_sin:
        print(f"[AVISO] {n_sin} imágenes sin EAR (excluidas de las métricas).")
    if df_valido.empty:
        sys.exit("Ninguna imagen produjo un EAR válido.")

    # --- umbral individual ---
    df_eval, tabla_calib, desc = calcular_umbral(df_valido, args)
    df_eval["pred"] = (df_eval["ear"] <= df_eval["umbral"]).astype(int)   # 1 = cerrado
    print(f"\nCalibración {desc}")
    if not tabla_calib.empty:
        print(tabla_calib.round(4).to_string(index=False))
        tabla_calib.to_csv(salida / "calibracion_por_grabacion.csv", index=False, encoding="utf-8-sig")
    print(f"Umbral medio de cierre = {df_eval['umbral'].mean():.4f} "
          f"(rango entre sujetos: {df_eval['umbral'].min():.3f} - {df_eval['umbral'].max():.3f})")

    # --- tabla por carpeta ---
    tabla = df_eval.groupby("clase").agg(
        N=("pred", "size"), EAR_mediana=("ear", "median"), EAR_norm_mediana=("ear_norm", "median"),
        Pct_pred_cerrado=("pred", lambda s: 100.0 * s.mean()))
    print("\nPredicción por carpeta del DMD:")
    print(tabla.round(4).to_string())

    # --- escenarios ---
    escenarios = {
        "A_open_vs_close (estricto)": {"open": 0, "close": 1},
        "B_open+opening_vs_close+closing": {"open": 0, "opening": 0, "close": 1, "closing": 1},
    }
    filas = []
    df_a = None
    for nombre, mapa in escenarios.items():
        d = df_eval[df_eval["clase"].isin(mapa)].copy()
        if d.empty or d["clase"].map(mapa).nunique() < 2:
            continue
        d["gt"] = d["clase"].map(mapa)
        m = metricas_binarias(d["gt"], d["pred"], -d["ear_norm"].to_numpy())
        thr_or, ba_or = mejor_umbral_oraculo(d["gt"], d["ear_norm"])
        m["Umbral_oraculo_norm(diagnostico)"] = thr_or
        m["BalAcc_oraculo(diagnostico)"] = ba_or
        filas.append({"Escenario": nombre, **m})
        if nombre.startswith("A_"):
            df_a = d
        graficar_confusion(m, f"{nombre}\nBalAcc={m['Balanced_Accuracy']:.3f}",
                           salida / f"matriz_confusion_{nombre.split('_')[0]}.png")

    resumen = pd.DataFrame(filas)
    print("\n" + "=" * 60)
    print(" RESULTADOS BENCHMARK DMD (evaluación por imagen)")
    print("=" * 60)
    for _, r in resumen.iterrows():
        print(f"\n[{r['Escenario']}]  N={int(r['N'])}")
        print(f"  TP={int(r['TP'])}  TN={int(r['TN'])}  FP={int(r['FP'])}  FN={int(r['FN'])}")
        for k in ("Accuracy", "Balanced_Accuracy", "Precision", "Recall", "Especificidad", "F1", "AUC_ROC"):
            print(f"  {k:18s}: {r[k]:.4f}")
        print(f"  (diagnóstico) umbral óptimo (EAR normalizado) = {r['Umbral_oraculo_norm(diagnostico)']:.3f} "
              f"[el usado es {1.0 - args.sensibilidad:.2f}] -> BalAcc {r['BalAcc_oraculo(diagnostico)']:.4f}")
    print("=" * 60)

    # --- métricas por grabación (escenario A) ---
    if df_a is not None:
        filas_g = []
        for g, sub in df_a.groupby("grabacion"):
            if sub["gt"].nunique() == 2:
                filas_g.append({"grabacion": g, **metricas_binarias(sub["gt"], sub["pred"],
                                                                    -sub["ear_norm"].to_numpy())})
        if filas_g:
            por_grab = pd.DataFrame(filas_g)
            por_grab.to_csv(salida / "metricas_por_grabacion_A.csv", index=False, encoding="utf-8-sig")
            print("\nEscenario A por grabación (sujeto):")
            print(por_grab[["grabacion", "N", "Balanced_Accuracy", "Recall", "Especificidad", "AUC_ROC"]]
                  .round(3).to_string(index=False))

    # --- exportes ---
    resumen.to_csv(salida / "metricas_resumen.csv", index=False, encoding="utf-8-sig")
    tabla.to_csv(salida / "prediccion_por_carpeta.csv", encoding="utf-8-sig")
    df_eval.to_csv(salida / "predicciones_por_imagen.csv", index=False, encoding="utf-8-sig")
    graficar_distribucion(df_eval, args.sensibilidad, salida / "distribucion_ear.png")
    graficar_pct_cerrado(tabla, salida / "pct_cerrado_por_carpeta.png")
    if args.guardar_errores and df_a is not None:
        guardar_errores(df_a, args.guardar_errores, salida)
    print(f"\nResultados exportados en: {salida}")


if __name__ == "__main__":
    main()
