# Clasificador de Géneros Musicales con Deep Learning

Este repositorio contiene una implementación de clasificación de géneros musicales usando PyTorch y el dataset GTZAN. El proyecto utiliza diferentes arquitecturas de redes neuronales para clasificar fragmentos de audio en 10 géneros musicales diferentes.

## Descripción

El proyecto implementa un sistema de clasificación de géneros musicales que:
- Utiliza el dataset GTZAN con 10 géneros musicales
- Procesa archivos de audio de 5 segundos
- Convierte las formas de onda en espectrogramas Mel para el análisis
- Implementa diferentes arquitecturas de redes neuronales (MLP, MLP2, CEL, CEL2, CEL3)
- Utiliza Weights & Biases (wandb) para el seguimiento de experimentos

## Características

- Preprocesamiento de audio:
  - Normalización
  - Transformación a espectrogramas Mel
  - Aumento de datos (time stretch, pitch shift)
  
- Arquitecturas implementadas:
  - MLP (Perceptrón Multicapa simple)
  - MLP2 (Perceptrón Multicapa con capas configurables)
  - CEL (Modelo Convolucional con ReLU)
  - CEL2 (Modelo Convolucional con LeakyReLU)
  - CEL3 (Modelo Convolucional con ELU)

- Seguimiento de experimentos:
  - Integración con Weights & Biases
  - Métricas de entrenamiento y validación
  - Comparación de diferentes configuraciones

## Funcionamiento del Repositorio

### Transformaciones de Audio
El sistema procesa el audio de dos formas diferentes:

Principales dependencias:
- PyTorch
- torchaudio
- wandb
- numpy
- matplotlib

2. **Espectrograma**
   - Transformación a espectrograma Mel
   - Conversión a decibeles
   - Normalización
   - Aumento de datos específico para espectrogramas

### Modelos Disponibles

1. **MLP (Perceptrón Multicapa Simple)**
   - 3 capas lineales
   - Función de activación ReLU
   - Capa de salida con softmax

2. **MLP2 (Perceptrón Multicapa Configurable)**
   - Número configurable de capas ocultas
   - Tamaño de capas ajustable
   - Dropout para regularización

3. **CEL (Modelo Convolucional)**
   - Capas convolucionales con batch normalization
   - Pooling y dropout
   - Capas densas finales
   - Función de activación ReLU

4. **CEL2 (Modelo Convolucional con LeakyReLU)**
   - Similar a CEL pero con:
   - LeakyReLU como función de activación (negative_slope=0.1)
   - Dropout ajustado
   - Mejor manejo del problema de neuronas muertas

5. **CEL3 (Modelo Convolucional con ELU)**
   - Similar a CEL pero con:
   - ELU como función de activación
   - Mejor convergencia y rendimiento
   - Manejo suave de valores negativos

## Entrenamiento de Modelos

### Configuración Inicial

1. **Configuración de Weights & Biases**
   - Crear una cuenta en [Weights & Biases](https://wandb.ai/)
   - Crear un archivo `.env` en la raíz del proyecto
   - Agregar tu API key:
   ```
   WANDB_API_KEY=tu_api_key_aqui
   ```

2. **Configuración del Entorno**
   ```bash
   conda create -n TDVI python=3.9
   conda activate TDVI
   pip install -r requirements.txt
   ```

## Entrenamiento de Modelos

### Proceso de Entrenamiento

El entrenamiento se realiza en dos modalidades diferentes:

1. **Usando Waveform (forma de onda)**
   ```python
   by = "waveform"
   train_model(model_name, num_epochs, device, hyperparameter_configs, 
              dataset, train_dataset, val_dataset, n_input, by)
   ```

2. **Usando Espectrograma**
   ```python
   by = "spectogram"
   train_model(model_name, num_epochs, device, hyperparameter_configs, 
              dataset, train_dataset, val_dataset, n_input, by)
   ```

### Configuraciones de Hiperparámetros

1. **Para MLP**
```python
hyperparameter_configs_MLP = [
    {
        "learning_rate": 0.001,
        "batch_size": 32,
        "nodes": 64
    }
]
```

2. **Para MLP2**
```python
hyperparameter_configs_MLP2 = [
    {
        "learning_rate": 0.001,
        "batch_size": 32,
        "nodes": 64,
        "hidden_layers": 3
    }
]
```

3. **Para modelos CEL, CEL2 y CEL3**
```python
hyperparameter_configs_CEL = [
    {
        "learning_rate": 0.001,
        "batch_size": 32,
        "nodes": 64,
        "hidden_layers": 3,
        "dropout_prob": 0.3,
        "dense_layers": 2
    }
]
```

### Parámetros de Entrenamiento

- **num_epochs**: 100 (por defecto)
- **batch_size**: Configurable en hyperparameter_configs
- **learning_rate**: Configurable en hyperparameter_configs
- **device**: Automáticamente detecta si hay GPU disponible

### Seguimiento del Entrenamiento

El proceso registra automáticamente en W&B:
- Pérdida de entrenamiento
- Pérdida de validación
- Precisión
- Matrices de confusión
- Curvas de aprendizaje

### Guardado de Modelos

- Los mejores modelos se guardan automáticamente en `best_models/`
- Se guarda el modelo con la menor pérdida de validación
- Formato de nombre: `{model_name}_{by}_best_model.pth`

## Estructura del Proyecto

- `Cooking.ipynb`: Notebook principal con implementación y experimentos
- `best_models/`: Directorio con los mejores modelos guardados
- `wandb/`: Logs y datos de experimentos de Weights & Biases
- `.env`: Archivo de configuración con la API key de W&B

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de crear un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.