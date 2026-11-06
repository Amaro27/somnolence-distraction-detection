# Plan de Migración a C++: Real-Time Gaze Classification (`LightGazeCNN`) & Benchmarking

Este documento describe la arquitectura, el flujo de conversión de modelos PyTorch (`.pth` → `.onnx`), las optimizaciones de memoria y vectorización SIMD, y la comparativa de rendimiento para la migración de [`benchmark_RTdetection.py`](../benchmark_RTdetection.py) a **C++17** con **OpenCV DNN**.

---

## 1. Justificación y Objetivos

En la versión de Python ([`benchmark_RTdetection.py`](../benchmark_RTdetection.py)), el perfilado reveló que el **preprocesamiento de imagen (`frame_preprocess`) consume más del 50% del tiempo total del frame** (~7.16 ms frente a ~3.07 ms de la propia red neuronal `LightGazeCNN`), debido a:
* Conversión de búferes NumPy (`cv2.cvtColor`) a objetos `PIL.Image` (`Image.fromarray`).
* Redimensionado bilinear en PIL y creación de nuevos tensores en cada fotograma (`transforms.ToTensor()`).
* Tres pasadas aritméticas en memoria por canal para normalizar con las medias y desviaciones estándar de ImageNet (`transforms.Normalize`).
* Overhead de despacho dinámico de PyTorch para modelos pequeños (lanzamiento de operadores capa por capa en CPU).

### Objetivos de la Migración a C++
1. **Eliminación de conversiones intermedias (Zero-PIL Overhead)**: Operar exclusivamente sobre matrices preasignadas `cv::Mat` sin asignaciones dinámicas de memoria (*zero heap allocations per frame*) dentro del bucle principal.
2. **Normalización Afín Fusionada con SIMD (AVX2)**: Reemplazar la división por `255.0` y la resta/división de `mean` y `std` por una única operación afín precalculada (`pixel * SCALE + SHIFT`) mientras se empaqueta directamente de layout `HWC` (`uint8`) a `NCHW` (`float32`).
3. **Inferencia Nativa Desacoplada de PyTorch (`cv::dnn`)**: Exportar los pesos entrenados de `.pth` a `.onnx` con *constant folding* (fusión estática de `Conv2d + BatchNorm2d`) para ejecutarlos directamente en el motor C++ de OpenCV DNN.
4. **Paridad 1:1 en Telemetría y Scopes**: Conservar exactamente los mismos identificadores de scope (`total_frame_cycle`, `camera_read`, `frame_preprocess`, `color_and_pil_convert`, `transform_and_tensor`, `model_inference`, `postprocessing`, `draw_overlay`, `gui_imshow`, `gui_waitkey`) y formato de exportación CSV.

---

## 2. Conversión de Archivos `.pth` a `.onnx`

Dado que los archivos `.pth` de PyTorch son diccionarios serializados con `pickle` (que además en este repositorio almacenan metadatos de entrenamiento como `optimizer_state_dict`, `epoch` y `val_acc`), C++ requiere un grafo computacional estándar.

Se implementó la utilidad [`export_to_onnx.py`](export_to_onnx.py), la cual:
1. Carga el checkpoint `.pth` extrayendo `checkpoint['model_state_dict']`.
2. Detecta automáticamente la arquitectura (`light_cnn`, `mobilenet_v3_small`, `resnet18`, `resnet18_clahe`) y construye el grafo en modo `.eval()`.
3. Exporta el modelo a `.onnx` (Opset 13) con `do_constant_folding=True`, fusionando las capas `BatchNorm2d` dentro de los pesos de `Conv2d` y reduciendo el tamaño del archivo de **5.09 MB (`.pth`) a 1.61 MB (`.onnx`)**.
4. Verifica automáticamente la paridad numérica ejecutando el mismo tensor aleatorio en PyTorch y en `cv2.dnn.readNetFromONNX`, confirmando un error máximo absoluto de **$9.54 \times 10^{-7}$** (precisión de máquina `float32`).

### Comandos de Conversión
```bash
# Convertir el modelo predeterminado (light_cnn_session_full_model.pth -> .onnx)
python CNN_Gaze/Benchmark/cnn/c++/export_to_onnx.py

# Convertir todos los modelos CNN en CNN_Gaze/Modelos/
python CNN_Gaze/Benchmark/cnn/c++/export_to_onnx.py --all
```

---

## 3. Optimizaciones Arquitectónicas en C++

### 3.1. Preprocesamiento Afín en Un Solo Paso (`pack_normalized_nchw_blob`)
En Python, `ToTensor()` y `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` calculan por cada canal $c \in \{R, G, B\}$:

$$\text{out}_c = \frac{\text{pixel}_c / 255.0 - \mu_c}{\sigma_c}$$

En [`benchmark_RTdetection.cpp`](benchmark_RTdetection.cpp), precalculamos en tiempo de compilación (`constexpr`):

$$\text{SCALE}_c = \frac{1.0}{255.0 \cdot \sigma_c}, \quad \text{SHIFT}_c = -\frac{\mu_c}{\sigma_c}$$

De modo que cada píxel se transforma y escribe directamente en los planos contiguos $R, G, B$ del tensor `NCHW` mediante una instrucción FMA (*Fused Multiply-Add*):

$$\text{out}_c = \text{pixel}_c \cdot \text{SCALE}_c + \text{SHIFT}_c$$

> [!IMPORTANT]
> A diferencia de `cv::dnn::blobFromImage` (que solo admite un único factor de escala escalar para los 3 canales), `pack_normalized_nchw_blob` aplica las tres desviaciones estándar exactas de ImageNet (`0.229`, `0.224`, `0.225`), garantizando **paridad numérica exacta** con `torchvision.transforms.Normalize`.

### 3.2. Resultados Empíricos (Python vs. C++)

| Scope Name | Depth | Python Mean (ms) | C++ Mean (ms) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **`total_frame_cycle`** | 1 | 13.39 ms | **2.63 ms** | **5.1x más rápido** |
| **`frame_preprocess`** | 1 | 7.16 ms | **0.38 ms** | **18.8x más rápido** |
| ├── `color_and_pil_convert` | 2 | 3.00 ms | **0.20 ms** | **15.0x más rápido** |
| └── `transform_and_tensor` | 2 | 4.12 ms | **0.17 ms** | **24.2x más rápido** |
| **`model_inference`** | 1 | 3.07 ms | **1.41 ms** | **2.2x más rápido** |
| **`postprocessing`** | 1 | 0.77 ms | **< 0.01 ms** | **> 75x más rápido** |

---

## 4. Instrucciones de Compilación y Ejecución

### Paso 1: Compilar el binario C++ (MSYS2 UCRT64 + Ninja/CMake)
```cmd
CNN_Gaze\Benchmark\cnn\c++\build.bat
```
Genera el ejecutable optimizado con `-O3 -mavx2` en `CNN_Gaze\Benchmark\cnn\c++\build\benchmark_RTdetection.exe`.

### Paso 2: Ejecutar el Benchmark
```cmd
:: Benchmark interactivo con Webcam (1000 frames por defecto, auto-quit)
CNN_Gaze\Benchmark\cnn\c++\run.bat

:: Benchmark con video pregrabado en modo Headless (sin GUI)
CNN_Gaze\Benchmark\cnn\c++\run.bat --source ruta\al\video.avi --max-frames 1000 --no-gui

:: Especificar directorio de salida o modelo ONNX personalizado
CNN_Gaze\Benchmark\cnn\c++\run.bat --model-path CNN_Gaze\Modelos\light_cnn_session_full_model.onnx --output-dir CNN_Gaze\Benchmark\cnn\c++
```
