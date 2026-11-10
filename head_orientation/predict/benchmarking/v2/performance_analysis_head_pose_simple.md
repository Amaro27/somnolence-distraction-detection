# Análisis de Rendimiento y Benchmarking v2: Arquitectura Desacoplada

Este documento describe la arquitectura de **Benchmarking Desacoplado (v2)** para [`head_pose_simple.py`](../../head_pose_simple.py).

---

## 1. Motivación y Filosofía del Enfoque v2

En la versión **v1**, la instrumentación de medición se inyectó directamente dentro de `head_pose_simple.py`. Si bien es directo, presenta inconvenientes en entornos de producción:
* Modifica el archivo objetivo y añade código ajeno al algoritmo central (importaciones de `csv`, `math`, `BenchmarkCollector`).
* Riesgo de introducir regresiones o sobrecarga permanente en despliegues reales.

La versión **v2** implementa un **arnés de pruebas desacoplado (*Harness Runner*)**:
1. **Archivo de producción 100% intacto**: [`head_orientation/predict/head_pose_simple.py`](../../head_pose_simple.py) se mantiene limpio y sin ninguna alteración.
2. **Importación modular**: El ejecutor de benchmark [`benchmark_head_pose.py`](benchmark_head_pose.py) importa las clases y funciones del script original (`Calibrator`, `FatigueClassifier`, `estimate_head_pose`, `draw_landmarks`, etc.).
3. **Reproducibilidad y automatización**: Permite alimentar tanto la cámara web en vivo como archivos de video pregrabados (`.avi`, `.mp4`) y ejecutar pruebas acotadas a un número exacto de frames (`--max-frames`).

---

## 2. Comparativa: v1 (In-Place) vs. v2 (Harness Desacoplado)

| Característica | v1 (In-Place) | v2 (Harness Desacoplado) |
| :--- | :--- | :--- |
| **Integridad del código base** | Modifica `head_pose_simple.py` | Deja `head_pose_simple.py` intacto |
| **Separación de responsabilidades** | Código de producción mezclado con telemetría | Telemetría totalmente aislada en `benchmarking/v2/` |
| **Fuentes de entrada** | Solo Webcam `cv2.VideoCapture(0)` | Webcam o videos grabados (`--source video.avi`) |
| **Control de ejecución** | Manual interactivo con `'q'` | Automático a 1000 frames por defecto (o manual con `'q'`, configurable vía `--max-frames`) |
| **Modo Headless (sin ventana GUI)** | No soportado | Soportado con flag `--no-gui` (mide puro cómputo CV) |
| **Formato de datos de salida** | Event-Log + Resumen CSV | Event-Log + Resumen CSV idéntico |

---

## 3. Jerarquía de Llamadas e Instrumentación en v2

El script desacoplado [`benchmark_head_pose.py`](benchmark_head_pose.py) importa los bloques lógicos de `head_pose_simple` y los envuelve en scopes de medición mediante context managers:

```text
Depth 0: run_benchmark() [Configuración de modelos, fuentes y bucle principal]
│
├── Depth 1: Ciclo por frame [1 ejecución por fotograma]
│   ├── [MEDIDO] camera_read              ── cap.read() [I/O Cámara o Video]
│   ├── [MEDIDO] frame_preprocess         ── cv2.flip() + cv2.cvtColor()
│   ├── [MEDIDO] face_detector            ── detector(gray, 0) [Dlib HOG]
│   │
│   ├── Bucle por rostro detectado (0, 1 o N veces según detección):
│   │   │
│   │   ├── Depth 2: Operaciones por rostro [Importadas de head_pose_simple]
│   │   │   ├── [MEDIDO] shape_predictor          ── predictor(gray, rostro)
│   │   │   ├── [MEDIDO] landmarks_array_init     ── np.zeros + 68 coordenadas
│   │   │   ├── [MEDIDO] draw_landmarks           ── head_pose_simple.draw_landmarks()
│   │   │   │
│   │   │   ├── [Fase Calibración] (si not calibrator.is_calibrated):
│   │   │   │   ├── [MEDIDO] estimate_head_pose   ── head_pose_simple.estimate_head_pose(..., baseline=None)
│   │   │   │   │   └── Depth 3: [Interna] build_face_frame(puntos)
│   │   │   │   ├── [MEDIDO] calibrator_accumulate── calibrator.accumulate(metrics)
│   │   │   │   ├── [MEDIDO] draw_face_axes       ── head_pose_simple.draw_face_axes(...)
│   │   │   │   └── [MEDIDO] draw_calibration_bar ── head_pose_simple.draw_calibration_overlay(...)
│   │   │   │
│   │   │   └── [Fase Inferencia] (si calibrator.is_calibrated):
│   │   │       ├── [MEDIDO] estimate_head_pose   ── head_pose_simple.estimate_head_pose(..., baseline)
│   │   │       │   └── Depth 3: [Interna] build_face_frame(puntos)
│   │   │       ├── [MEDIDO] fatigue_classify     ── classifier.classify(...)
│   │   │       ├── [MEDIDO] draw_face_axes       ── head_pose_simple.draw_face_axes(...)
│   │   │       └── [MEDIDO] draw_hud             ── head_pose_simple.draw_hud(...)
│   │
│   ├── [MEDIDO] gui_imshow               ── cv2.imshow(...) [Opcional si not --no-gui]
│   ├── [MEDIDO] gui_waitkey              ── cv2.waitKey(1) [Opcional si not --no-gui]
│   └── [MEDIDO] total_frame_cycle        ── Tiempo acumulado completo del frame
```

---

## 4. Gestión de Longitudes Dispares (*Ragged Data*)

El recolector almacena las métricas en formato **Tidy / Event-Log**:
Cada invocación representa un renglón en memoria con su metadato de contexto (`frame_id`, `face_id`, `scope_name`, `depth`, `elapsed_ms`, `context`).

### Ventajas:
1. **Invariante ante número de rostros**: Si en el frame 5 no hay rostros, no se registran filas de `shape_predictor`. Si en el frame 12 hay 3 rostros, se registran 3 filas diferenciadas por `face_id`.
2. **Invariante ante cambios de fase**: Los primeros 60 fotogramas registran eventos de `calibrator_accumulate`, y los subsecuentes registran `fatigue_classify`.
3. **Cero pérdida de sincronización**: A diferencia de tener listas paralelas (que se desfasarían si una función no se llama), cada fila es completamente autosuficiente.

---

## 5. Salidas Generadas

Al alcanzar el límite de 1000 frames (por defecto), salir anticipadamente con `'q'`, alcanzar un `--max-frames` personalizado o finalizar el video de prueba, se generan automáticamente en `benchmarking/v2/`:

1. `benchmark_events_YYYYMMDD_HHMMSS.csv`: Registro detallado de cada una de las invocaciones.
2. `benchmark_summary_YYYYMMDD_HHMMSS.csv`: Resumen consolidado con `count`, `mean_ms`, `std_ms`, `median_ms`, `p95_ms`, `min_ms`, `max_ms`, `total_ms` y `%_of_total_frame`.
3. **Tabla resumen en consola**: Imprime el desglose jerárquico de cuellos de botella directamente en la terminal.

---

## 6. Modo de Uso

### Caso 1: Benchmark estándar con Webcam (corre 1000 frames y se cierra automáticamente)
```bash
python head_orientation/predict/benchmarking/v2/benchmark_head_pose.py
```
*(Aprox. 33 segundos de captura a 30 FPS. Al completar el fotograma 1000, el arnés guarda los CSVs, imprime la tabla y termina solo).*

### Caso 2: Benchmark con archivo de video pregrabado (hasta 1000 frames)
```bash
python head_orientation/predict/benchmarking/v2/benchmark_head_pose.py --source path/to/video.avi
```

### Caso 3: Modificar el número de frames objetivo
```bash
# Ejecutar una prueba más corta de 200 frames:
python head_orientation/predict/benchmarking/v2/benchmark_head_pose.py --max-frames 200

# Ejecutar sin límite de frames (solo se detiene pulsando 'q'):
python head_orientation/predict/benchmarking/v2/benchmark_head_pose.py --max-frames 0
```

### Caso 4: Medición de rendimiento puro de cómputo (Headless sin GUI)
```bash
python head_orientation/predict/benchmarking/v2/benchmark_head_pose.py --source path/to/video.avi --max-frames 1000 --no-gui
```
