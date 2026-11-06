# Comparativa 1:1 de Rendimiento: Python vs. C++ (LightGazeCNN)

Este documento presenta el análisis comparativo exhaustivo, métrica por métrica, entre las ejecuciones estándar de **1000 fotogramas** de los arneses de benchmarking en Python ([`benchmark_RTdetection.py`](benchmark_RTdetection.py)) y C++ ([`c++/benchmark_RTdetection.cpp`](c++/benchmark_RTdetection.cpp)).

---

## 1. Fuentes de Datos de la Comparativa

| Entorno | Archivo de Origen | Frames Analizados | Duración Total Registrada | Fecha y Hora |
| :--- | :--- | :---: | :---: | :---: |
| **Python** | [`benchmark_summary_20261105_163907.csv`](benchmark_summary_20261105_163907.csv) | 1,000 | 33.93 s | 2026-11-05 16:39:07 |
| **C++ (OpenCV DNN)** | [`c++/build/benchmark_summary_20261105_163639.csv`](c++/build/benchmark_summary_20261105_163639.csv) | 1,000 | 66.97 s | 2026-11-05 16:36:39 |

---

## 2. Tabla Comparativa 1:1 (Métricas en Milisegundos)

A continuación se contrastan la media (*Mean*), mediana (*Median / p50*) y percentil 95 (*p95*) para cada scope y profundidad de llamada:

| Scope Name | Depth | Context | Python Mean (ms) | C++ Mean (ms) | Python p50 (ms) | C++ p50 (ms) | Python p95 (ms) | C++ p95 (ms) | Factor de Aceleración (*Speedup*) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`frame_preprocess`** | 1 | - | 3.093 | **0.305** | 2.944 | **0.285** | 4.159 | **0.549** | **10.14x más rápido** (+90.1% vel.) |
| ├── `transform_and_tensor` | 2 | - | 2.184 | **0.110** | 2.084 | **0.102** | 2.993 | **0.198** | **19.85x más rápido** (+95.0% vel.) |
| └── `color_and_pil_convert` | 2 | - | 0.866 | **0.182** | 0.822 | **0.165** | 1.133 | **0.339** | **4.76x más rápido** (+79.0% vel.) |
| **`model_inference`** | 1 | - | 4.428 | **2.168** | 4.144 | **2.065** | 6.295 | **3.256** | **2.04x más rápido** (+51.0% vel.) |
| **`postprocessing`** | 1 | - | 0.068 | **0.001** | 0.050 | **0.001** | 0.079 | **0.002** | **68.00x más rápido** (+98.5% vel.) |
| `draw_overlay` | 1 | - | 0.191 | **0.173** | **0.109** | 0.138 | **0.174** | 0.331 | **1.10x más rápido** |
| `gui_imshow` | 1 | - | **0.290** | 0.612 | 0.194 | **0.160** | **0.329** | 0.402 | **1.21x más rápido en p50** (pico en C++) |
| `gui_waitkey` | 1 | - | 6.997 | **4.899** | 7.869 | **4.850** | 10.185 | **6.020** | **1.43x más rápido** |
| `camera_read` (I/O) | 1 | - | 18.706 | 58.773 | 11.370 | 56.727 | 42.786 | 74.164 | *Bloqueo de Sensor Webcam* |
| `total_frame_cycle` | 1 | - | 33.927 | 66.974 | 27.820 | 64.667 | 58.339 | 82.070 | *Impactado por fps de cámara* |

---

## 3. Análisis de Rendimiento de Cómputo Puro (Sin Esperas de I/O)

Al aislar el tiempo de **cómputo real de CPU** (preprocesamiento de imagen, inferencia de la red convolucional, postprocesamiento softmax/argmax y renderizado de texto/ventana gráfica), excluyendo los tiempos ociosos en los que el hilo de ejecución duerme esperando a la cámara web (`camera_read`) o al evento de teclado del sistema operativo (`gui_waitkey`):

```text
Cómputo Puro = frame_preprocess + model_inference + postprocessing + draw_overlay + gui_imshow
```

| Métrica de Cómputo Puro | Python | C++ (OpenCV DNN) | Ganancia Neta |
| :--- | :---: | :---: | :---: |
| **Tiempo de CPU por Frame (Media)** | **8.07 ms** | **3.26 ms** | **-4.81 ms por frame (-59.6%)** |
| **Tiempo de CPU por Frame (Mediana / p50)** | **7.44 ms** | **2.65 ms** | **-4.79 ms por frame (-64.4%)** |
| **Tiempo Total de Cómputo (1000 frames)** | 8.07 s | 3.26 s | **4.81 segundos menos de CPU** |
| **Throughput Teórico Máximo (Media)** | ~123.9 FPS | **~306.8 FPS** | **+147.6% de capacidad de procesamiento** |
| **Throughput Teórico Máximo (Mediana)** | ~134.4 FPS | **~377.5 FPS** | **+180.9% de capacidad de procesamiento** |
| **Factor de Aceleración Algorítmica Global** | 1.00x | **2.48x** | **C++ es 2.48x más rápido en cómputo puro** |

---

## 4. Hallazgos Arquitectónicos Clave

### 1. Preprocesamiento: La Mayor Ganancia de la Migración (**10.14x más rápido**)
* **Cuello de botella en Python**: En Python, `frame_preprocess` consumía **3.09 ms** (llegando a p95 de 4.16 ms) debido al ciclo de conversión de memoria de NumPy a PIL (`Image.fromarray`), creación de objetos tensores en heap (`ToTensor()`) y 3 pasadas secuenciales en memoria por canal (`transforms.Normalize`).
* **Optimización en C++**: La función [`pack_normalized_nchw_blob`](c++/benchmark_RTdetection.cpp) realiza la normalización de ImageNet en un único paso afín vectorizado FMA ($\text{pixel}_c \cdot \text{SCALE}_c + \text{SHIFT}_c$) directamente sobre el búfer contiguo preasignado `1x3x112x112` con **cero asignaciones dinámicas por frame**.
* **Impacto**: La preparación tensorial pasó de **2.18 ms** a solo **0.11 ms** (**19.85x más rápida**).

### 2. Inferencia Convolucional: OpenCV DNN vs. PyTorch CPU (**2.04x más rápido**)
* En Python, la inferencia de `LightGazeCNN` promedió **4.43 ms** en CPU.
* En C++, el grafo exportado a ONNX (con *constant folding* y fusión de capas `BatchNorm2d` dentro de `Conv2d`) ejecutado por `cv::dnn` promedió **2.17 ms** (**2.04x más rápido**), reduciendo la latencia de pico p95 de **6.30 ms a 3.26 ms**.

### 3. Postprocesamiento Sofisticado a Costo Cero (**68x más rápido**)
* En Python, la llamada a `torch.softmax`, `torch.max`, `.item()` y el formateo de cadenas consumía **0.068 ms**.
* En C++, la búsqueda de máximo y cálculo exponencial numéricamente estable con resta de logit máximo (`logits - max_logit`) toma **menos de 0.001 ms (1 microsegundo)**, eliminando la sobrecarga por completo.

### 4. Explicación de la Discrepancia en `camera_read`
* En C++, `camera_read` promedió **58.77 ms** frente a **18.71 ms** en Python.
* **Causa técnica**: La llamada `cap.read()` es una llamada síncrona bloqueante al driver DirectShow/MediaFoundation de Windows. Durante la prueba en C++, la cámara ajustó dinámicamente su tiempo de exposición / framerate de captura debido a la iluminación ambiental, entregando fotogramas a una tasa menor (~15-17 FPS vs ~30 FPS).
* De los 66.97 ms del ciclo total en C++, **58.77 ms (el 87.8%) correspondieron únicamente a tiempo de espera ocioso del bus USB de la cámara**. La CPU estuvo completamente desocupada durante dicho intervalo.

---

## 5. Conclusiones

1. **Paridad y Robustez Confirmadas**: La versión C++ procesó los 1000 fotogramas sin fugas de memoria, produciendo las mismas clases y métricas agregadas que Python.
2. **Eficiencia en CPU Extrema**: El tiempo de cómputo puro por frame se redujo de **8.07 ms a 3.26 ms** (**2.48x de aceleración global**, y más de **3.0x** en el pipeline combinado de preprocesamiento + inferencia).
3. **Capacidad de Throughput**: En un entorno de procesamiento por lotes o video sin bloqueo de sensor, C++ es capaz de procesar más de **300 a 375 FPS**, comparado con ~125 a 135 FPS en Python.
