# Como utilizar codigos de CNN (Entrenamiento y TR)

Se crearon 2 codigos de python para el entrenamiento de CNNs que buscan detectar distraccion en el conductor. Igualmente, se crearon 2 archivos para correr en tiempo real los modelos entrenados. Para simplicidad y personificacion de entrenamiento, los codigos son mejor utilizados en terminal donde se inidican especificaciones del modelo. A continuacion se explica como utilizar cada uno de estos codigos en la terminal. Cuando corras la indicacion en terminal asegurate de estar en la misma ubicacion que tu codigo .py (recuerda utilizar cd)

- train_light_cnn.py
Todos los codigos a excepcion de RTdetection.py fueron programados con la libreria de argparse para poder dar inidcaciones desde la terminal al codigo. La manera en la que esta libreria funciona es que dentro del codigo se programa una indicacion, como puede ser por ejemplo "--epochs" para que si se agrega esta indicacion en la terminal, el usuario pueda indicar cuantas epocas entrenar su modelo. A continuacion se muestran todas las indicaciones que se pueden agregar en terminal a este codigo 

    * --datasetpath: con esta indicacion se puede ingresar la ubicacion del dataset, y el codigo buscara automaticamente la direccion de looking_on_road y not_looking_road siempre y cuando estas se encunetren en una carpeta dmd_rgb y gaze_on_road, como esta acomodado en el dataset y GitHub. Esto gracias a la funcion scan_dataset_by_session (puedes revisarla en el .py para entender como escanea la informacion si lo deseas). 
    * --batch_size: establece el tamaño de batch que el modelo agarrara (cantidad de datos en los que se parten los grupos a utilizar en el entrenamiento). Actualmente se utiliza un batch size de 128 pues es comunmente utilizado, pero puese modificarse con intencion de mejorar rendimieto.
    * --epochs: como se menciono anteriormente, con esta indicacion se puede definir la cantidad de epocas que se entrena el modelo.
    * --lr: con lr se puede establecer el learning rate con el cual se entrena el modelo. (Debe de funcionar con 0.0001 o 1e-4, pero no he confirmado)
    * --img_size: se puede cambiar el tamaño con el cual se entrenan las imagenes. El tamaño original es de 224x224, pero se estuvo entrenando con 114x114 por ahora. 
    * --max_samples: puedes elegir cuantas imagenes del total seleccionar. Si no se usa simplemente se usaran todas las imagenes. 
    * --experiment_name: este configura el nombre especifico con el cual se guarda el modelo y graficas de resultados. Todos los nombres siguen una estructura especifica, y simplemente se agrega la palabra que pongas aqui en ese nombre

Ejemplo de indicacion ejecutada en terminal para entrenal un modelo: **python -u train_session_split_cnn.py --epochs 5 --batch_size 128 --max_samples 50000 --experiment_name session_subsampleD1**

**No es necesario poder todas las indicaciones siempre que quieras utilizar el codigo, pues todos cuentan con valores default en caso de que no los uses**
- train_transfer_cnn.py
Este codigo cuenta con todos los modificadores que tiene el codigo anterior, pero se agregan unos pocos relacionados al transfer learning realizado. 
    * --model_name: este indica cual de los dos modelos quieres utilizar, MobileNet o ResNet. Para MobileNet escribe mobilenet_v3_small y para ResNet resnet18.
    * --unfreeze_mode: con esta indicacion se define cuanto de la red neuronal se descongela para reentrenar con tus imagenes. Hay tres opciones por el momento, siendo head, partial, y full, siendo solo la ultima capa, las ultimas, o completa respectivamente. 

Ejemplo: **-u train_transfer_cnn.py --model_name resnet18 --unfreeze_mode partial --epochs 10 --batch_size 128 --max_samples 50000 --experiment_name subsample10**

- RTdetection.py
Este codigo no cuenta con configuracion argparse, por lo que solo necesitas correrlo en VScode y asegurarte que la ubicacion del modelo sea la correcta.

- RTdetection_transfer.py
Dentro de este codigo se pueden correr en tiempo real cualquiera de los dos modelos establecidos en el entrenamiento de transfer learning ( MobileNet y ResNet) usando las indicaciones necesarias, de manera parecida a la ya explicada. Este codigo cuenta con algunas indicaciones ya mencionadas siendo: --model_name e img_size (recuerda que debe ser el mismo tamaño que entenamiento). A continuacion se muetran las indicaciones propias de este codigo:
    * --window_size: Cantidad de frames que se utilizan para establecer la moda y resultado de prediccion mostrado en pantalla
    * --model_path: ubicacion del modelo, que es basicamente el nombre + .pth
    * --camera_idx: numero de la camara en caso de tener varias, default es 0
 
Ejemplo: **python RTdetection_transfer.py --model_name mobilenet_v3_small --model_path transfer_mobilenet_v3_small_full_model.pth --window_size 15**
