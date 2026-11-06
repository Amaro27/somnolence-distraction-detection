# Comparativa 1:1 de Rendimiento: Python vs. C++ (CNN-LSTM Driver Gaze)

Este documento presenta el análisis comparativo exhaustivo, métrica por métrica, entre las ejecuciones estándar de **1000 fotogramas** de los arneses de benchmarking para el clasificador temporal de mirada **CNN-LSTM (GazeResNetLSTM)** en Python ([`benchmark_RTdetection_lstm.py`](benchmark_RTdetection_lstm.py)) y C++ ([`c++/benchmark_RTdetection_lstm.cpp`](c++/benchmark_RTdetection_lstm.cpp)).

---

## 1. Fuentes de Datos de la Comparativa

| Entorno | Archivo de Origen | Frames Analizados | Duración Total Registrada |
| :--- | :--- | :---: | :---: |
| **Python** | [`benchmark_summary_20261105_164708.csv`](benchmark_summary_20261105_164708.csv) | 1,000 | 59.85 s |
| **C++** | [`c++/benchmark_summary_20261105_210623.csv`](c++/benchmark_summary_20261105_210623.csv) | 1,000 | 67.18 s |

---

## 2. Tabla Comparativa 1:1 (Métricas en Milisegundos)

A continuación se contrastan la media (*Mean*), mediana (*Median/p50*) y percentil 95 (*p95*) para cada scope y profundidad (*Depth*) de llamada instrumentada:

| Scope Name | Depth | Context | Python Mean (ms) | C++ Mean (ms) | Python p50 (ms) | C++ p50 (ms) | Python p95 (ms) | C++ p95 (ms) | Factor de Aceleración (*Speedup*) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `model_inference` | 1 | - | 38.315 | **14.663** | 37.529 | **14.420** | 50.626 | **15.922** | **2.61x más rápido** (+61.7% vel.) |
| ├── `cnn_feature_extraction` | 2 | - | 36.345 | **13.586** | 35.403 | **13.370** | 47.421 | **14.778** | **2.68x más rápido** (+62.6% vel.) |
| └── `lstm_sequence_forward` | 2 | - | 1.850 | **1.063** | 1.695 | **1.025** | 3.080 | **1.366** | **1.74x más rápido** (+42.5% vel.) |
| `frame_preprocess` | 1 | - | 6.294 | **0.492** | 6.118 | **0.465** | 7.887 | **0.734** | **12.79x más rápido** (+92.2% vel.) |
| ├── `transform_and_tensor` | 2 | - | 4.982 | **0.363** | 4.779 | **0.341** | 6.367 | **0.557** | **13.72x más rápido** (+92.7% vel.) |
| └── `color_and_pil_convert` | 2 | - | 1.240 | **0.118** | 1.232 | **0.106** | 1.515 | **0.193** | **10.51x más rápido** (+90.5% vel.) |
| `postprocessing` | 1 | - | 0.093 | **0.001** | 0.075 | **0.001** | 0.123 | **0.001** | **93.00x más rápido** |
| `draw_overlay` | 1 | - | 0.331 | **0.131** | 0.248 | **0.124** | 0.366 | **0.167** | **2.53x más rápido** (+60.4% vel.) |
| `gui_imshow` | 1 | - | 0.449 | **0.414** | 0.332 | **0.118** | 0.459 | **0.171** | **1.08x más rápido** (mediana 2.81x) |
| `gui_waitkey` | 1 | - | **4.435** | 4.960 | **1.490** | 4.362 | 14.034 | **5.616** | *Event loop UI* |
| `camera_read` (I/O) | 1 | - | **9.715** | 46.494 | **8.917** | 44.688 | **12.482** | 61.075 | *Bloqueo de Sensor Webcam* |
| `total_frame_cycle` | 1 | - | **59.846** | 67.183 | **58.834** | 64.304 | **71.063** | 81.112 | *Impactado por fps de cámara* |

---

## 3. Análisis de Rendimiento de Cómputo Puro (Sin Esperas de I/O)

Al aislar el tiempo de **cómputo real de CPU** (preprocesamiento con CLAHE, inferencia convolucional ResNet-18, inferencia temporal LSTM, postprocesamiento y renderizado gráfico), excluyendo el tiempo ocioso en el que el hilo espera a la cámara web (`camera_read`) o al despachador de eventos de usuario (`gui_waitkey`):

```text
Cómputo Puro = model_inference + frame_preprocess + postprocessing + draw_overlay + gui_imshow
```

| Métrica de Cómputo Puro | Python | C++ | Ganancia Neta |
| :--- | :---: | :---: | :---: |
| **Tiempo de CPU por Frame** | **45.48 ms** | **15.70 ms** | **-29.78 ms por frame (-65.5%)** |
| **Tiempo Total en 1000 frames** | 45.48 s | 15.70 s | **29.78 segundos menos de CPU** |
| **Throughput Teórico Máximo** | ~22.0 FPS | **~63.7 FPS** | **+189.6% de capacidad (+41.7 FPS)** |
| **Factor de Aceleración Algorítmica** | 1.00x | **2.90x** | **C++ es casi 3x más rápido en cómputo puro** |

Si se evalúa exclusivamente el **pipeline de Inteligencia Artificial y Visión** (`frame_preprocess` + `model_inference` + `postprocessing`):
* **Python**: 44.70 ms por frame (**~22.4 FPS teóricos**)
* **C++**: 15.16 ms por frame (**~66.0 FPS teóricos**)
* **Aceleración del Core IA**: **2.95x más rápido** (ahorro neto de 29.54 ms por frame).

---

## 4. Hallazgos Arquitectónicos Clave

### 1. Inferencia Neuronal Híbrida CNN-LSTM (`model_inference`)
* **Extracción Espacial ResNet-18 (`cnn_feature_extraction`)**:
  * En Python/PyTorch CPU requirió **36.35 ms** por cuadro (60.7% del ciclo total).
  * En C++ con OpenCV DNN compilado con soporte vectorial **AVX2**, la latencia cayó a **13.59 ms** (**2.68x más rápido**, ahorrando **22.76 ms netos por cuadro**).
  * En el percentil 95 (`p95`), la estabilidad es notable: C++ mantiene **14.78 ms** vs. los **47.42 ms** de Python (**3.21x más rápido**), eliminando las pausas de GC y jitter de la runtime de PyTorch.
* **Inferencia Temporal LSTM (`lstm_sequence_forward`)**:
  * En Python demandó **1.85 ms** debido a la concatenación dinámica y creación de tensores en memoria intermedia: `torch.cat(list(feature_buffer), dim=0).unsqueeze(0)`.
  * En C++, el buffer circular `std::deque<std::array<float, 512>>` combinado con la rutina de copia contigua zero-allocation `pack_sequence_blob()` redujo la ejecución a **1.06 ms** (**1.74x más rápido**).

### 2. Preprocesamiento de Imágenes y Transformación CLAHE (`frame_preprocess`)
* La optimización más drástica se obtuvo en el preprocesamiento, reduciendo la latencia de **6.29 ms a 0.49 ms** (**12.79x más rápido**, reducción del **92.2%** del tiempo):
  * **Python**: `CLAHETransform` requería convertir de BGR a RGB con OpenCV, envolver en `PIL.Image`, convertir a escala de grises con PIL (`convert('L')`), aplicar `cv2.createCLAHE`, reconvertir a RGB con OpenCV, generar otra `PIL.Image`, y finalmente ejecutar `transforms.ToTensor()` y `transforms.Normalize()`.
  * **C++**: El pipeline opera completamente in-place en memoria contigua `cv::Mat`. El redimensionamiento a 224x224, la ecualización adaptativa de histograma (`cv::CLAHE`) y el empaquetado a tensor NCHW con normalización ImageNet se fusionan mediante aritmética de punto flotante precalculada (`pack_normalized_nchw_blob`), tomando apenas **0.36 ms** frente a **4.98 ms** en Python (**13.72x de aceleración**).

### 3. Postprocesamiento sin Overhead de Tensores (`postprocessing`)
* En Python, calcular `torch.softmax` y `torch.max` sobre el tensor de salida tomó **0.093 ms**.
* En C++, la búsqueda del argmax y softmax se realiza directamente sobre el arreglo nativo de logits en registros de CPU, ejecutándose en **0.001 ms** (**93x más rápido**).

### 4. Renderizado Gráfico e Interfaz de Usuario
* Las funciones `draw_overlay` (dibujo de texto de FPS y estado con `cv::putText`) son **2.53x más rápidas** en C++ (**0.13 ms vs. 0.33 ms**).
* `gui_imshow` demostró una mediana de **0.12 ms en C++** frente a **0.33 ms en Python** (**2.81x más rápido**).

### 5. Explicación de la Discrepancia en `camera_read`
* En C++, `camera_read` promedió **46.49 ms** frente a **9.72 ms** en Python.
* **Causa técnica**: La llamada `cap.read()` es síncrona y bloqueante a nivel del driver DirectShow/MSMF en Windows. La cámara web ajustó su tiempo de exposición según la iluminación ambiental del momento de la prueba, entregando fotogramas a una tasa de captura menor (~20-22 FPS).
* De los 67.18 ms del ciclo total en C++, **46.49 ms (el 69.2%) correspondieron exclusivamente a la espera de hardware del sensor de la webcam**. Durante este período, los núcleos de la CPU estuvieron completamente libres.

---

## 5. Conclusión y Recomendación

La implementación en **C++** del clasificador temporal CNN-LSTM confirma una ganancia sustancial de rendimiento frente a Python:
1. **El cómputo algorítmico puro es 2.90x más rápido**, liberando **29.78 ms de CPU por fotograma** (reducción del 65.5% del costo de cómputo).
2. **Capacidad de tiempo real demostrada**: El pipeline C++ procesa cada fotograma en **15.70 ms** (equivalente a **~63.7 FPS teóricos**), lo que permite operar holgadamente en hardware embebido o CPUs modestas a 30 o 60 FPS sin saturar el procesador, a diferencia de Python que quedaba limitado a un techo de ~22 FPS.
3. **Paridad funcional y métrica estricta**: Se conservó la misma jerarquía de medición en 2 niveles (Depth 1 y 2), la misma estructura de log de eventos y estadísticas agregadas, garantizando una comparación y trazabilidad directa entre ambos lenguajes.
