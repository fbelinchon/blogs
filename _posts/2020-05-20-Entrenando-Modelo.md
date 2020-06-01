---
layout: post
title: "Entrenamiento de un modelo de red neuronal"
subtitle: "Backpropagation para entrenar modelos de redes neuronales."
date: 2020-05-19 10:45:13 -0400
background: '/img/posts/06.jpg'
categories: AI
---

# ¿Como entrenar una red neuronal?

## Objetivo

El objetivo de este artículo es explicar de forma sencilla el proceso de entrenamiento de una red neuronal como la descrita en el artículo anterior. Intentaremos evitar las explicaciones matemáticas detalladas, pero si daremos una visión intuitiva del proceso que nos ayude a entenderlo.

## Intuición sobre el proceso de entrenamiento.

El concepto del entrenamiento de un modelo de redes neuronales es sencillo y lo hemos mencionado en artículos anteriores. El modelo realiza una predicción sobre los datos de ejemplo que le suministramos y compara la predicción con el valor real que debe proporcionar. El objetivo es que ese error disminuya poco a poco hasta conseguir un modelo con unas predicciones muy cercanas a los valores reales.

En los primeros pasos del entrenamiento las predicciones tendrán un error muy grande porque los parámetros del modelo se han seleccionado de modo aleatorio. Por ejemplo, si el modelo intenta predecir si la imagen es de un perro o de un gato la predicción sería como tirar una moneda al aire. Según el modelo ajusta los parámetros la tasa de acierto empieza a mejorar. La parte más complicada de este proceso es decidir cuando dejamos de entrenar. Llegado un momento el modelo deja de aprender de los datos de ejemplo y empieza a memorizarlos. Si el modelo empieza a memorizar los datos de ejemplo realizará predicciones muy exactas sobre estos datos pero las predicciones sobre datos nuevos serán muy inexactas. Ya vemos como detectamos este comportamiento.

La clave del aprendizaje está en ir ajustando los parámetros de forma progresiva para ir disminuyendo el error cometido. El modelo analiza cada parámetro y decide si aumentando o disminuyendo cada uno de los parámetros se reduce el error cometido.

