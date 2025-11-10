# Face (vs No-Face) Classification – Training, Detection and Model Zoo

Este proyecto entrena clasificadores binarios (cara vs no-cara), permite detectar múltiples caras en imágenes completas y compararlas usando diferentes arquitecturas, incluyendo modelos preentrenados con fine-tuning.

## Estructura del proyecto

```
.
├── artifacts/
│   ├── <modelo>/
│   │   ├── best_model.pt        # checkpoint del mejor modelo para ese nombre
│   │   └── history.json         # historia de loss/acc
│   ├── detections_*.png         # imágenes anotadas de detección
│   └── compare_*.txt            # resultados de comparaciones
├── load_data.py                 # dataloaders, preprocesado y data augmentation
├── models.py                    # todas las arquitecturas y registro de modelos
├── net.py                       # CNN original (baseline antigua)
├── train.py                     # script de entrenamiento/evaluación
├── detect_and_classify.py       # detector+clasificador por imagen completa
├── compare_models.py            # compara todos los modelos sobre una imagen
├── train_images/                # dataset de entrenamiento (ImageFolder)
│   ├── 0/ ... (noface)
│   └── 1/ ... (face)
└── test_images/                 # dataset de test (ImageFolder)
    ├── 0/ ... (noface)
    └── 1/ ... (face)
```

Formato de datos (ImageFolder):
- `train_images/0` = no-cara, `train_images/1` = cara
- `test_images/0` = no-cara, `test_images/1` = cara

## Requisitos

- Python 3.9+
- PyTorch, TorchVision
- OpenCV
- scikit-learn (para métricas únicamente)

Instalación (pip):
```bash
pip install torch torchvision opencv-python scikit-learn
```

## Modelos disponibles (MODEL_REGISTRY)

Los modelos se referencian por nombre con `--model <nombre>`.

Modelos “pequeños” (entrada 36x36, escala de grises):
- tiny: CNN mínima (8→16 canales, FC pequeña). Muy rápida, menos capacidad.
- small: CNN pequeña (12→24 canales). Compromiso entre velocidad y capacidad.
- baseline: CNN base original (16→32, 2 FC). Punto de referencia.
- bn: CNN con BatchNorm. Suele estabilizar y acelerar el entrenamiento.
- threeconv: CNN con 3 bloques conv y dos poolings. Más capacidad.
- residual: CNN con bloques residuales simples. Mejora el flujo de gradientes.
- improved: CNN mejorada (BatchNorm en todas las capas, 3 bloques, Dropout2D/1D). Generalmente más robusta.
- attention: CNN con atención por canal (tipo SE). Mejora el foco en características relevantes.

Modelos preentrenados (entrada 224x224, RGB, normalización ImageNet):
- resnet18: ResNet18 preentrenada (ImageNet) + fine-tuning. Buena precisión con coste moderado.
- mobilenetv2: MobileNetV2 preentrenada. Muy eficiente en CPU/edge.
- efficientnet: EfficientNet-B0 preentrenada. Excelente relación precisión/eficiencia.

Diferencias clave entre familias:
- Tamaño de entrada: 36x36 (CNN pequeñas) vs 224x224 (preentrenados).
- Preprocesado: Escala de grises + normalización (pequeños) vs RGB + normalización de ImageNet (preentrenados).
- Capacidad/regularización: `improved`/`attention` usan BatchNorm y Dropout; `residual` usa atajos; preentrenados reutilizan características generales de ImageNet con fine-tuning.

## Preprocesado y Data Augmentation

`load_data.py` adapta automáticamente el preprocesado según el modelo:
- CNN pequeñas: 36x36, Grayscale, Normalización mean=0.5, std=0.5.
- Preentrenados: 224x224, Grayscale→RGB (3 canales), Normalización ImageNet.

Data augmentation (entrenamiento):
- Preentrenados: Resize+RandomCrop(224), HorizontalFlip, Rotación, ColorJitter.
- Pequeños: HorizontalFlip, Rotación ligera.

## Entrenamiento

Entrena un modelo y guarda el mejor checkpoint y la historia en `artifacts/<modelo>/`:
```bash
# CNN pequeñas
python3 train.py --model baseline --epochs 10
python3 train.py --model improved --epochs 10
python3 train.py --model attention --epochs 10

# Preentrenados (suelen requerir batch-size menor)
python3 train.py --model resnet18 --epochs 10 --batch-size 32
python3 train.py --model mobilenetv2 --epochs 10 --batch-size 32
python3 train.py --model efficientnet --epochs 10 --batch-size 32
```
Parámetros útiles:
- `--epochs`, `--batch-size`, `--lr`, `--num-workers`

Durante el entrenamiento se imprime en cada época: loss/acc train y valid, y al final se evalúa el test con el mejor checkpoint.

## Detección y clasificación en imágenes completas

Detecta caras (Haar cascade), recorta, preprocesa y clasifica cada cara con el modelo seleccionado. Genera una imagen anotada de salida.
```bash
python3 detect_and_classify.py /ruta/a/imagen.jpg --model improved --show
# Por defecto carga artifacts/<model>/best_model.pt
# Ajustes del detector Haar
python3 detect_and_classify.py /ruta/imagen.jpg --model resnet18 \
  --scale-factor 1.1 --min-neighbors 5 --threshold 0.5
```
Salida:
- Imagen anotada en `artifacts/detections_<nombre>.png` (o la ruta de `--save`).
- Log con bbox, etiqueta y probabilidad por detección.

### Ejecutar detección con todos los modelos

Para procesar una imagen con cada modelo entrenado y generar una imagen anotada por modelo:

```bash
python3 detect_all_models.py /ruta/imagen.jpg --skip-missing
```

- Salva las imágenes en `artifacts/detections/<nombre_imagen>/<modelo>.png`.
- Usa `--models baseline,improved,resnet18` para limitar la lista.
- Ajusta `--threshold`, `--scale-factor`, `--min-neighbors` igual que en `detect_and_classify.py`.
- Con `--skip-missing` omitirá modelos sin checkpoint, en lugar de fallar.

## Comparar modelos rápidamente sobre una imagen

Calcula la probabilidad de “face” de TODOS los modelos registrados sobre una misma imagen (sin detección, a imagen completa). Útil para un sanity check y ranking rápido.
```bash
python3 compare_models.py /ruta/a/imagen.jpg --save
```
- Imprime una tabla en consola.
- Con `--save` genera `artifacts/compare_<nombre>.txt`.

## Evaluación comparativa de modelos (métricas + gráficos)

Para evaluar todos (o un subconjunto) de modelos sobre el set de test y obtener una tabla con accuracy, precision, recall y F1, además de gráficos comparativos:

```bash
python3 evaluate_models.py
# o limitar a algunos
python3 evaluate_models.py --models baseline,improved,resnet18
```

Salida en `artifacts/evaluation/`:
- `summary.csv`: tabla con métricas por modelo.
- `accuracy.png`, `f1.png`, `precision.png`, `recall.png`: gráficos de barras comparativos.
- `cm_<modelo>.png`: matrices de confusión de los top-3 modelos por accuracy.

### Intervalos de confianza con Bootstrapping

Puedes estimar la incertidumbre de las métricas con bootstrapping (remuestreo con reemplazo sobre el set de test):

```bash
# 1000 remuestreos y CI del 95%
python3 evaluate_models.py --bootstrap 1000 --ci 95
```

Archivos adicionales:
- `bootstrap_summary.csv`: media, desvío estándar y CI [low, high] por modelo y métrica.
- `bootstrap_<modelo>.csv`: distribución bootstrap por métrica para cada modelo.
- `<metric>_violin.png`: violines comparando la distribución bootstrap entre modelos.

## Guía rápida para principiantes

1) Instalar dependencias
```bash
# En entornos sin GUI (Jupyter/servidores) usa headless:
pip install torch torchvision opencv-python-headless scikit-learn
# En entornos con GUI local:
pip install torch torchvision opencv-python scikit-learn
```

2) Preparar datos
- Crea carpetas `train_images/0`, `train_images/1`, `test_images/0`, `test_images/1`.
- Coloca imágenes de “no-cara” en `0/` y “cara” en `1/`.

3) Entrenar
```bash
# Todos los modelos (recomendado si quieres compararlos)
python3 train_all.py

# Un modelo específico (más rápido)
python3 train.py --model improved --epochs 10
```
Salida: `artifacts/<modelo>/best_model.pt` y `history.json`.

4) Probar detección en una imagen
```bash
python3 detect_and_classify.py /ruta/imagen.jpg --model improved --show
# Si estás en servidor/Jupyter, omite --show y revisa artifacts/detections_<nombre>.png
```

5) Probar con todos los modelos a la vez
```bash
python3 detect_all_models.py /ruta/imagen.jpg --skip-missing
```
Salida: una imagen anotada por modelo en `artifacts/detections/<nombre_imagen>/`.

6) Comparar “probabilidad de cara” sin detección (imagen completa)
```bash
python3 compare_models.py /ruta/imagen.jpg --save
```

## ¿Qué significa el “porcentaje” que ves en las imágenes?
- Es la probabilidad que el modelo asigna a la clase “face” (cara) para cada recorte detectado.
- Si el valor es mayor o igual al umbral (`--threshold`, por defecto 0.5), la etiqueta será “face”; si no, “noface”.
- Puedes elevar el umbral (por ejemplo 0.6–0.7) si prefieres menos falsos positivos, a costa de perder algunas caras verdaderas (recall menor).

## Diferencias entre modelos (alto nivel)

- Modelos pequeños (entrada 36x36, escala de grises):
  - `tiny` y `small`: muy rápidos y ligeros; menor capacidad.
  - `baseline`: referencia original, desempeño decente.
  - `bn`: añade BatchNorm; entreno más estable.
  - `threeconv`: un poco más profundo; suele mejorar.
  - `residual`: atajos residuales; mejor flujo del gradiente.
  - `improved`: mejor práctica (BatchNorm+Dropout2D/1D); robusto.
  - `attention`: añade atención por canal (tipo SE); puede enfocarse mejor en rasgos útiles.

- Preentrenados (entrada 224x224, RGB, normalización ImageNet):
  - `resnet18`: buen equilibrio precisión/tiempo.
  - `mobilenetv2`: muy eficiente en CPU/edge.
  - `efficientnet`: gran precisión/eficiencia, algo más pesado.

Reglas generales:
- Si tu dataset es pequeño o variado, los preentrenados suelen generalizar mejor.
- Si necesitas velocidad y poco consumo, usa `mobilenetv2` o `tiny/small`.
- Si quieres un “caballo de batalla” sin preentrenamiento, prueba `improved` o `residual`.

## ¿Cómo comparar modelos de forma justa?

- Mismo conjunto de test para todos.
- Mismo umbral (`--threshold`) si comparas “porcentaje de cara”.
- Métricas recomendadas: accuracy, precision, recall, F1.
- Opciones:
  - Usar `test.py` para una evaluación y matriz de confusión (un modelo a la vez).
  - Extender `compare_models.py` para iterar sobre `test_images/` y calcular métricas globales por modelo (sugerido para informes).

Idea rápida (no implementada aún):
- Script que recorra `test_images/`, ejecute cada modelo y compute accuracy/F1 por modelo, generando una tabla CSV.

## Preguntas frecuentes / Troubleshooting

- “No encuentro cv2 (OpenCV)”
  - Instala: `pip install opencv-python-headless` (servidor/Jupyter) o `pip install opencv-python` (local con GUI).

- “Los modelos preentrenados fallan con errores de ‘shared memory (shm)’”
  - Ejecuta entrenos con `num_workers=0` para preentrenados. `train_all.py` ya lo hace por defecto.
  - También puedes bajar `--batch-size` si hay poco RAM/GPU.

- “No veo ventana con la imagen”
  - En servidores/Jupyter no hay GUI. Omite `--show` y revisa la imagen guardada en `artifacts/`.

- “No tengo checkpoint”
  - Entrena primero: `python3 train.py --model baseline` (u otro).

- “¿Cómo ajusto sensibilidad?”
  - Sube el umbral (`--threshold 0.6` o `0.7`) para menos falsos positivos.
  - Ajusta detector Haar con `--scale-factor` y `--min-neighbors`.

## Glosario breve
- “Detección”: encontrar dónde hay una cara (bounding box) en una imagen.
- “Clasificación”: decidir si un recorte es “cara” o “no-cara”.
- “Preentrenado”: el modelo ya aprendió rasgos generales de millones de imágenes (ImageNet) y luego lo ajustamos a tu tarea (fine-tuning).
- “Data augmentation”: transformaciones de imagen (rotar, recortar, voltear) para hacer el modelo más robusto.

## Comandos clave (chuleta)
```bash
# Entrenar todo
python3 train_all.py

# Entrenar uno
python3 train.py --model improved --epochs 10

# Detección con un modelo
python3 detect_and_classify.py /ruta/imagen.jpg --model resnet18 --threshold 0.6

# Detección con todos los modelos
python3 detect_all_models.py /ruta/imagen.jpg --skip-missing

# Comparar probabilidades sin detección
python3 compare_models.py /ruta/imagen.jpg --save
```

## Evaluación más completa (sugerida)

- Usa `test.py` para métricas y matriz de confusión sobre el conjunto de test (para un modelo a la vez).
- Extiende `compare_models.py` para iterar sobre `test_images/` y producir una tabla agregada con accuracy/precision/recall/F1 por modelo.

## Recomendaciones prácticas

- Balanceo de clases: Si el dataset está desbalanceado, puedes:
  - Usar `WeightedRandomSampler` (integrarlo en `load_data.py`).
  - Ajustar `CrossEntropyLoss` con `weight` por clase.
- Umbral de decisión: Ajusta `--threshold` para priorizar recall o precision según tu objetivo.
- Early stopping y schedulers: Puedes agregar un scheduler (Cosine/OneCycle) y early stopping para los preentrenados.
- Recursos: Preentrenados consumen más memoria; baja `--batch-size` si ves OOM.

## Dependencias y notas

- Instalar dependencias principales:
```bash
pip install torch torchvision opencv-python scikit-learn
```
- scikit-learn: El paquete a instalar es `scikit-learn`; el import en código es `from sklearn ...`.
- OpenCV Haar cascade está incluido en `cv2.data.haarcascades`.

## Preguntas frecuentes

- No abre mi imagen con `detect_and_classify.py`:
  - Verifica la ruta. Si usas ruta absoluta, no olvides comillas si tiene espacios.
  - Ejemplo correcto: `python3 detect_and_classify.py "~/Pictures/foto.jpg" --model baseline`
- No hay checkpoint:
  - Debes entrenar primero: `python3 train.py --model baseline`
- ¿Puedo usar video/webcam?
  - El flujo está listo para imagen. Se puede extender con captura de frames y aplicar el mismo pipeline (puedo añadirlo si lo necesitas).

---

Si quieres, puedo añadir un script de evaluación sobre `test_images/` que produzca una tabla comparando todas las métricas por modelo y un gráfico de barras con accuracy/F1.


