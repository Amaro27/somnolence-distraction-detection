# Análisis de Rendimiento y Benchmarking: `benchmark_RTdetection.py`

Este documento define la arquitectura de benchmarking, la jerarquía de llamadas, los alcances de medición (*measurement scopes*), el manejo de datos y las instrucciones de uso para el clasificador en tiempo real basado en CNN ligera [`benchmark_RTdetection.py`](benchmark_RTdetection.py).

---

## 1. Contexto y Objetivos

El script [`RTdetection.py`](../../RTdetection.py) ejecuta inferencia en tiempo real sobre un feed de video (o cámara web) clasificando si el conductor está mirando al frente hacia el camino (`looking_road`) o distraído (`not_looking_road`) mediante la arquitectura convolucional ligera `LightGazeCNN`.

### Objetivos del Benchmarking
1. **Medición de alta resolución de etapas críticas**: Monitorear con `time.perf_counter()` las operaciones fundamentales en `main()`:
   - Lectura de cámara / video (`camera_read`).
   - Preprocesamiento de cuadro (`frame_preprocess`): conversión BGR a RGB con PIL y normalización/redimensionamiento tensorial con torchvision.
   - Inferencia convolucional (`model_inference`) con sincronización CUDA cuando aplique.
   - Postprocesamiento (`postprocessing`): softmax y cálculo de clase/confianza.
   - Renderizado gráfico y eventos de interfaz (`draw_overlay`, `gui_imshow`, `gui_waitkey`).
2. **Exportación Dual en formato Tidy / Event-Log**:
   - Registro de eventos individuales (*Event Log CSV*).
   - Tabla estadística de resumen (*Summary CSV*) con percentiles (p50, p95), medias, mínimos, máximos y porcentaje del ciclo total del fotograma (`% Frame`).
3. **Automatización y Reproducibilidad**:
   - Soporte para cámaras web (`--source 0`) o archivos de video pregrabados (`--source video.mp4`).
   - Límite de fotogramas configurable con auto-finalización (`--max-frames 1000`).
   - Modo headless sin interfaz gráfica (`--no-gui`) para aislar el cómputo puro de CV y Deep Learning sin penalizaciones del gestor de ventanas del sistema operativo.

---

## 2. Jerarquía de Llamadas e Instrumentación de Scopes

La jerarquía de medición se organiza en dos niveles de profundidad (*Depth*):

```text
Depth 0: run_benchmark() [Carga de modelo, inicialización de tensores y loop principal]
│
├── Depth 1: Ciclo por frame [total_frame_cycle]
│   ├── [MEDIDO] camera_read              ── cap.read() [I/O Cámara o Archivo de Video]
│   │
│   ├── [MEDIDO] frame_preprocess         ── Preprocesamiento completo
│   │   ├── Depth 2: color_and_pil_convert   ── cv2.cvtColor(BGR2RGB) + Image.fromarray()
│   │   └── Depth 2: transform_and_tensor    ── transforms.Resize + ToTensor + Normalize + unsqueeze(0).to(device)
│   │
│   ├── [MEDIDO] model_inference          ── model(input_tensor) [+ torch.cuda.synchronize() si CUDA]
│   │
│   ├── [MEDIDO] postprocessing           ── torch.softmax + torch.max + label extraction
│   │
│   ├── [MEDIDO] draw_overlay             ── cv2.putText (FPS y clasificación en pantalla) [si not --no-gui]
│   ├── [MEDIDO] gui_imshow               ── cv2.imshow('Real-Time Gaze Classification', frame) [si not --no-gui]
│   └── [MEDIDO] gui_waitkey              ── cv2.waitKey(1) [si not --no-gui]
```

---

## 3. Esquema de Métricas Exportadas

Al pulsar `'q'` o completar el límite especificado en `--max-frames`, se generan automáticamente dos archivos CSV en el directorio del script:

### 1. `benchmark_events_<timestamp>.csv`
| Columna | Tipo | Descripción |
| :--- | :--- | :--- |
| `timestamp` | ISO String | Marca temporal en milisegundos |
| `frame_id` | Entero | Índice correlativo del fotograma |
| `face_id` | Entero | ID del objeto (-1 para métricas a nivel de frame) |
| `scope_name` | String | Nombre del bloque medido |
| `depth` | Entero | Profundidad jerárquica (1 o 2) |
| `elapsed_ms` | Float | Duración en milisegundos |
| `context` | String | Metadatos contextuales adicionales |

### 2. `benchmark_summary_<timestamp>.csv`
Consolida métricas estadísticas por scope:
- `count`: Cantidad total de llamadas registradas.
- `mean_ms`: Latencia promedio en ms.
- `std_ms`: Desviación estándar.
- `median_ms`: Mediana (percentil 50, robusto frente a outliers de I/O o GC).
- `p95_ms`: Percentil 95 (representativo de latencia máxima sostenida).
- `min_ms` y `max_ms`: Valores extremos observados.
- `total_ms`: Tiempo total acumulado por la operación.
- `pct_of_total_frame`: Contribución porcentual al ciclo total del fotograma (`total_frame_cycle`).

---

## 4. Instrucciones de Uso

### Caso 1: Benchmark interactivo con Webcam (1000 frames por defecto)
```bash
python CNN_Gaze/Benchmark/cnn/benchmark_RTdetection.py
```
*(Al completar los 1000 frames o presionar `'q'`, se exportan los CSVs y se imprime la tabla resumen).*

### Caso 2: Benchmark con archivo de video específico
```bash
python CNN_Gaze/Benchmark/cnn/benchmark_RTdetection.py --source ruta/al/video.mp4 --max-frames 500
```

### Caso 3: Medición de rendimiento puro (Modo Headless / sin GUI)
```bash
python CNN_Gaze/Benchmark/cnn/benchmark_RTdetection.py --source ruta/al/video.mp4 --max-frames 1000 --no-gui
```

### Caso 4: Ruta de modelo personalizada
```bash
python CNN_Gaze/Benchmark/cnn/benchmark_RTdetection.py --model-path CNN_Gaze/Modelos/light_cnn_session_full_model.pth
```
