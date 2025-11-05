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

## Comparar modelos rápidamente sobre una imagen

Calcula la probabilidad de “face” de TODOS los modelos registrados sobre una misma imagen (sin detección, a imagen completa). Útil para un sanity check y ranking rápido.
```bash
python3 compare_models.py /ruta/a/imagen.jpg --save
```
- Imprime una tabla en consola.
- Con `--save` genera `artifacts/compare_<nombre>.txt`.

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


