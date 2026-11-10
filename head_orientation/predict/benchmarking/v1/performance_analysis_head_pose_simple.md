# Análisis de Rendimiento y Benchmarking: `head_pose_simple.py`

Este documento define el plan metodológico, la jerarquía de llamadas, los niveles de profundidad (*call depth*), los alcances de comparación (*comparison scopes*) y el diseño de recolección de métricas para el script de estimación de pose cefálica [`head_pose_simple.py`](head_pose_simple.py).

---

## 1. Contexto y Objetivos

El script [`head_pose_simple.py`](head_pose_simple.py) ejecuta un ciclo en tiempo real de captura de video (webcam), detección facial con Dlib (HOG + SVM), extracción de 68 puntos faciales, cálculo geométrico de ángulos *Pitch* (inclinación vertical / somnolencia) y *Roll* (inclinación lateral / distracción), y renderizado de overlays gráficos con OpenCV.

### Objetivos del Benchmarking
1. **Identificar cuellos de botella reales**: Medir el tiempo de ejecución de las operaciones invocadas directamente por `main()` usando deltas de alta resolución con `time.perf_counter()`.
2. **Resolver el problema de datos de longitud dispar (*ragged data*)**: Diseñar un esquema de almacenamiento que soporte variaciones dinámicas en el número de llamadas (0, 1 o múltiples rostros detectados, fase de calibración vs. inferencia continua, salidas tempranas).
3. **Persistencia dual en CSV**: Exportar un registro detallado de eventos (*Event Log*) y un resumen estadístico agregado (*Summary Table*) al finalizar la sesión (al pulsar `'q'`).

---

## 2. Jerarquía de Llamadas y Niveles de Profundidad

La siguiente jerarquía refleja la estructura de ejecución del script. 
* Las funciones e instrucciones marcadas con **[MEDIDO EN MAIN]** son el alcance directo del benchmark.
* Las sub-llamadas internas marcadas con *[Interna]* se documentan para comprender el flujo, pero **no** se instrumentan individualmente para evitar overhead innecesario.

```text
Depth 0: main() [Inicialización de modelos y loop principal]
│
├── Depth 1: Ciclo por frame [Ejecutado 1 vez por cada iteración del bucle while True]
│   ├── [MEDIDO] camera_read              ── cap.read()
│   ├── [MEDIDO] frame_preprocess         ── cv2.flip() + cv2.cvtColor()
│   ├── [MEDIDO] face_detector            ── detector(gray, 0) [Dlib HOG]
│   │
│   ├── Bucle por rostro detectado (0, 1 o N veces según el frame):
│   │   │
│   │   ├── Depth 2: Operaciones por rostro [Invocadas directamente en main()]
│   │   │   ├── [MEDIDO] shape_predictor          ── predictor(gray, rostro) [Dlib 68-points]
│   │   │   ├── [MEDIDO] landmarks_array_init     ── np.zeros + bucle iterativo de 68 coordenadas
│   │   │   ├── [MEDIDO] draw_landmarks           ── draw_landmarks(frame, puntos)
│   │   │   │
│   │   │   ├── [RAMA A] Fase de Calibración (si not calibrator.is_calibrated):
│   │   │   │   ├── [MEDIDO] estimate_head_pose   ── estimate_head_pose(puntos, baseline=None)
│   │   │   │   │   └── Depth 3: [Interna] build_face_frame(puntos)
│   │   │   │   ├── [MEDIDO] calibrator_accumulate── calibrator.accumulate(metrics)
│   │   │   │   ├── [MEDIDO] draw_face_axes       ── draw_face_axes(frame, metrics, "NORMAL")
│   │   │   │   └── [MEDIDO] draw_calibration_bar ── draw_calibration_overlay(frame, progress)
│   │   │   │
│   │   │   └── [RAMA B] Fase de Inferencia / Fatiga (si calibrator.is_calibrated):
│   │   │       ├── [MEDIDO] estimate_head_pose   ── estimate_head_pose(puntos, baseline)
│   │   │       │   └── Depth 3: [Interna] build_face_frame(puntos)
│   │   │       ├── [MEDIDO] fatigue_classify     ── classifier.classify(...)
│   │   │       ├── [MEDIDO] draw_face_axes       ── draw_face_axes(frame, metrics, pitch_lvl)
│   │   │       └── [MEDIDO] draw_hud             ── draw_hud(frame, metrics, pitch_lvl, roll_lvl)
│   │
│   ├── [MEDIDO] gui_imshow               ── cv2.imshow(...)
│   ├── [MEDIDO] gui_waitkey              ── cv2.waitKey(1)
│   └── [MEDIDO] total_frame_cycle        ── Tiempo total transcurrido en el frame completo
```

---

## 3. Pseudocódigo Estructurado con Scopes de Comparación

El siguiente pseudocódigo ilustra el flujo de control y las fronteras de medición (`bench.measure`):

```python
# Depth 0: Configuración global
predictor, detector, cap, calibrator, classifier = init_pipeline()
bench = BenchmarkCollector()
frame_id = 0

while True:
    with bench.measure("total_frame_cycle", depth=1, frame_id=frame_id):
        # 1. Captura de Cámara (I/O)
        with bench.measure("camera_read", depth=1, frame_id=frame_id):
            ret, frame = cap.read()
        if not ret:
            break

        # 2. Preprocesamiento de Imagen
        with bench.measure("frame_preprocess", depth=1, frame_id=frame_id):
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. Detección Facial (CPU Heavy)
        with bench.measure("face_detector", depth=1, frame_id=frame_id):
            rostros = detector(gray, 0)

        # 4. Procesamiento por rostro (Frecuencia variable: 0..N)
        for face_id, rostro in enumerate(rostros):
            with bench.measure("shape_predictor", depth=2, frame_id=frame_id, face_id=face_id):
                shape = predictor(gray, rostro)

            with bench.measure("landmarks_array_init", depth=2, frame_id=frame_id, face_id=face_id):
                puntos = np.zeros((68, 2), dtype="int")
                for i in range(68):
                    puntos[i] = (shape.part(i).x, shape.part(i).y)

            with bench.measure("draw_landmarks", depth=2, frame_id=frame_id, face_id=face_id):
                draw_landmarks(frame, puntos)

            if not calibrator.is_calibrated:
                # Fase de calibración (context="uncalibrated")
                with bench.measure("estimate_head_pose", depth=2, frame_id=frame_id, face_id=face_id, context="uncalibrated"):
                    metrics = estimate_head_pose(puntos, baseline=None)

                with bench.measure("calibrator_accumulate", depth=2, frame_id=frame_id, face_id=face_id):
                    calibrator.accumulate(metrics)

                with bench.measure("draw_face_axes", depth=2, frame_id=frame_id, face_id=face_id, context="calibrating"):
                    draw_face_axes(frame, metrics, "NORMAL")

                with bench.measure("draw_calibration_bar", depth=2, frame_id=frame_id, face_id=face_id):
                    draw_calibration_overlay(frame, calibrator.progress)
            else:
                # Fase de inferencia (context="calibrated")
                with bench.measure("estimate_head_pose", depth=2, frame_id=frame_id, face_id=face_id, context="calibrated"):
                    metrics = estimate_head_pose(puntos, baseline=calibrator.baseline)

                with bench.measure("fatigue_classify", depth=2, frame_id=frame_id, face_id=face_id):
                    pitch_lvl, roll_lvl = classifier.classify(metrics["pitch_deg"], metrics["roll_deg"], calibrator.baseline)

                with bench.measure("draw_face_axes", depth=2, frame_id=frame_id, face_id=face_id, context="tracking"):
                    draw_face_axes(frame, metrics, pitch_lvl)

                with bench.measure("draw_hud", depth=2, frame_id=frame_id, face_id=face_id):
                    draw_hud(frame, metrics, pitch_lvl, roll_lvl)

        # 5. Renderizado y Event Loop (GUI)
        with bench.measure("gui_imshow", depth=1, frame_id=frame_id):
            cv2.imshow("Head Pose Estimation", frame)

        with bench.measure("gui_waitkey", depth=1, frame_id=frame_id):
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    frame_id += 1

# Exportación al salir
bench.export_csvs(output_dir=".")
bench.print_summary()
```

---

## 4. Solución al Problema de Longitudes Dispares (*Ragged Data*)

### ¿Por qué los datos tienen longitudes distintas?
1. **Detecciones dinámicas de rostros**: Si en el frame 1 no hay nadie frente a la cámara, `shape_predictor` se llama 0 veces. Si en el frame 20 entran 2 personas, se llama 2 veces.
2. **Fase de calibración vs. Inferencia**: `calibrator.accumulate` solo se llama durante los primeros 60 frames (`CALIBRATION_FRAMES`). A partir de ese momento, se llama `classifier.classify` y `draw_hud`.
3. **Interrupciones interactivas**: Si el usuario pulsa `'q'` en el frame 42, las listas tendrían solo 42 elementos de lectura de cámara, pero tal vez 30 de detección y 0 de clasificación de fatiga.

### Enfoque de Almacenamiento: Event-Log / Tidy Data
En lugar de forzar una matriz rectangular (donde las columnas tengan que medir lo mismo), cada medición se registra como una fila independiente en un **Registro de Eventos (*Event Log*)**.

#### 1. Archivo Detallado: `benchmark_events_<timestamp>.csv`
Cada llamada genera una fila con metadatos contextuales:

| Columna | Tipo | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- |
| `timestamp` | ISO String | Momento exacto de la medición | `2026-11-03T19:45:12.102` |
| `frame_id` | Entero | Índice correlativo del fotograma | `42` |
| `face_id` | Entero / Nulo | Índice del rostro dentro del frame (-1 si es a nivel de frame) | `0` |
| `scope_name` | String | Nombre de la función o bloque medido | `shape_predictor` |
| `depth` | Entero | Nivel en el árbol de llamadas (1 o 2) | `2` |
| `elapsed_ms` | Float | Duración en milisegundos (`Δt * 1000`) | `3.421` |
| `context` | String | Metadatos de la llamada (calibrating, calibrated, uncalibrated) | `calibrated` |

#### 2. Archivo Resumen: `benchmark_summary_<timestamp>.csv`
Al pulsar `'q'`, el collector calcula automáticamente estadísticas agregadas por `scope_name` y `context`:

| Columna | Descripción |
| :--- | :--- |
| `scope_name` | Identificador de la función / bloque |
| `depth` | Profundidad jerárquica |
| `context` | Contexto de ejecución |
| `count` | Número total de invocaciones registradas |
| `mean_ms` | Tiempo medio de ejecución (milisegundos) |
| `std_ms` | Desviación estándar (estabilidad temporal) |
| `median_ms` | Mediana (p50, resistente a outliers por I/O o GC) |
| `p95_ms` | Percentil 95 (peor caso representativo / latencia de pico) |
| `min_ms` | Tiempo mínimo observado |
| `max_ms` | Tiempo máximo observado |
| `total_ms` | Tiempo acumulado total consumido por la operación |
| `pct_of_frame` | Porcentaje representativo respecto al tiempo total del frame |

---

## 5. Implementación del Collector Ligero

El collector se implementa mediante un Context Manager en Python:
* Usa `time.perf_counter()` (el reloj monotónico de mayor resolución del sistema operativo).
* No crea objetos pesados durante el loop crítico (almacena tuplas simples en memoria).
* Escribe a disco únicamente al presionar `'q'` o terminar el video, evitando bloqueos de I/O de disco durante la captura en vivo.
