# Clasificador de Géneros Musicales con Deep Learning

Este repositorio contiene una implementación de clasificación de géneros musicales usando PyTorch y el dataset GTZAN. El proyecto utiliza diferentes arquitecturas de redes neuronales para clasificar fragmentos de audio en 10 géneros musicales diferentes.

## Descripción

El proyecto implementa un sistema de clasificación de géneros musicales que:
- Utiliza el dataset GTZAN con 10 géneros musicales
- Procesa archivos de audio de 5 segundos
- Convierte las formas de onda en espectrogramas Mel para el análisis
- Implementa diferentes arquitecturas de redes neuronales (MLP, MLP2, CEL)
- Utiliza Weights & Biases (wandb) para el seguimiento de experimentos

## Características

- Preprocesamiento de audio:
  - Normalización
  - Transformación a espectrogramas Mel
  - Aumento de datos (time stretch, pitch shift)
  
- Arquitecturas implementadas:
  - MLP (Perceptrón Multicapa simple)
  - MLP2 (Perceptrón Multicapa con capas configurables)
  - CEL (Modelo Convolucional con capas densas)

- Seguimiento de experimentos:
  - Integración con Weights & Biases
  - Métricas de entrenamiento y validación
  - Comparación de diferentes configuraciones

## Requisitos