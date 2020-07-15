---
layout: post
title: "Entrenamiento de un modelo de red neuronal"
subtitle: "Backpropagation para entrenar modelos de redes neuronales."
date: 2020-05-19 10:45:13 -0400
background: '/img/posts/06.jpg'
categories: AI
---

# Objetivo

El objetivo de este artículo es explicar de forma sencilla el proceso de entrenamiento de una red neuronal como la descrita en el artículo anterior. Intentaremos evitar las explicaciones matemáticas detalladas, pero si daremos una visión intuitiva del proceso que nos ayude a entenderlo.

# Intuición sobre el proceso de entrenamiento.

El concepto del entrenamiento de un modelo de redes neuronales es sencillo y lo hemos mencionado en artículos anteriores. El modelo realiza una predicción sobre los datos de ejemplo que le suministramos y compara la predicción con el valor real que debe proporcionar. El objetivo es que ese error disminuya poco a poco hasta conseguir un modelo con unas predicciones muy cercanas a los valores reales. Para dismuir el error modificamos ligeramente los parámetros de nuestro modelo de forma iterativa.

En los primeros pasos del entrenamiento las predicciones tendrán un error muy grande porque los parámetros del modelo se han seleccionado de modo aleatorio. Por ejemplo, si el modelo intenta predecir si la imagen es de un perro o de un gato la predicción sería como tirar una moneda al aire. Un conjunto aleatorio de parámetros significa que en esos momentos el modelo no sabe nada sobre los datos que tiene que predecir. Según el modelo ajusta los parámetros la tasa de acierto empieza a mejorar. La parte más complicada de este proceso es decidir cuando dejamos de entrenar. Llegado un momento el modelo deja de aprender de los datos de ejemplo y empieza a memorizarlos. Si el modelo empieza a memorizar, realizará predicciones muy exactas sobre los ejemplos de entrenamiento pero las predicciones sobre datos nuevos serán muy inexactas. Posteriormente veremos como detectar ese comportamiento.

La clave del aprendizaje está en ir ajustando los parámetros de forma progresiva para ir disminuyendo el error cometido. El modelo analiza cada parámetro y decide si aumentando o disminuyendo su valor para conseguir el menor error posible.

# Gradient Descent como método de aprendizaje (optimización).

Para decidir si tenemos que aumentar o disminuir los parámetros del modelo recurrimos al concepto de deriva de una función.

> Derivada de una función en un punto: se puede interpretarse geométricamente como la pendiente de la recta tangente a la gráfica de la función en dicho punto.

En definitiva la derivada es la pendiente de una curva en un punto. Si lo que buscamos es el mínimo de una función tiene sentido apoyarnos en la pendiente de esa curva para saber si tenemos que aumentar o disminuir ese parámetro. La pendiente nos informa de con que rapidez varia la función y en que sentido crece o decrece.

En el primer ártículo de la serie comparábamos nuestro objetivo de encontrar el mínimo con la búsqueda del camino más rápido para bajar desde el pico de la montaña al valle. En cada paso seleccionábamos la dirección de mayor pendiente para asegurar que bajábamos lo más rápidamente posible.

Nuestro modelo puede tener miles o millones de parámetros con lo que utilizamos lo que se llama derivada parcial de una función con respecto a un parámetro. Esta operación nos devuelve la pendiente de la función en un punto para un parámrtro concreto dejando el resto de parámetros fijos. El cálculo de las derivadas parciales de miles de pármetros es una operación muy costosa y en muchos casos inviable. En la práctica utilizamos una aproximación que se denomina *Gradient Descent*.



## Stochastic Gradiente descent SGD.
Para calcular el gradiente en un punto estudiamos como se comporta la función a su alrededor. Aumentamos ligeramente el parámetro y vemos si la función crece o decrece. De esta forma obtenemos valores similares a la deriva parcial de forma menos costosa. Si tenemos el valor de la función en un punto y el valor cuando un poco el parámetro podemos estimar el valor de pendiente en ese punto. No es el valor exacto pero es una buena aproximación para saber si tenemos que aumentar o disminuir el valor de nuestro parámetro para aproximarnos al mínimo de la función de coste.

El método más popular que se utiliza en el entrenamiento de redes neuronales se denomina Stochastic Gradient Descent o simplemente SGD. Es un proceso iterativo que calcula el gradiente de un grupo de ejemplos y con esa información actualiza los parámetros el modelo. No cecesitamos conocer en detalle como calcular los gradientes de la función de coste porque todas las librería de Deep Learning lo hacen por nosotros. Simplemente necesitamos una ligera intuición para entender el mecanismo de optimización. Vamos a detallar el proceso utilizando la nomneclatura propia asociada a redes neuronales.

### Fases del proceso de optimización utilizando SGD
Nuestro conjunto de entrenamiento se divide en pequeños subconjuntos denominados batch o mini-batch. La selección de qué ejemplos entran en cada batch se suele hacer de forma aleatoria. El núemro de elementos de cada batch se define inicialmente y es el mismmo para todos excepto para el último que tendrá el número de ejemplo que queden. Este valor se denomina *batch size* y es una de los principales valores a definir al entrenar un modelo de red neuronal.

Pongamos un ejemplo. Si tenemos un conjunto con 1000 elementos y utilizamos un tamaño de batch (batch Size) igual a 64 tendríamos.
* 15 batches con 64 elementos.
* 1 batch final con 40 elementos que son los que quedan.

Para entrenar el modelo vamos pasando uno por uno nuestros 16 grupos o batches. Por cada batch el modelo calcula la salida, la compara con el resultado que debería producirse y actualiza los parámetros para que la función de error disminuya.
Es importante entender que para un batch solamente hay una actualización de los parámetros. No se produce una modificación de los parámetros por cada uno de los 64 ejemplo sino que se realiza una estimación para el conjunto total de los 64 ejemplos. En estudios realizados se ha demostrado que el cálculo del gradiente de esta forma nos permite aproximarnos al mínimo de una forma más rápida y más consistente que el hecho de calcular el gradiente individual de cada ejemplo.

Cuando hemos completado un ciclo entero, en este caso los 16 batches definidos, el modelo ha recibido todos los ejemplos de nuestro conjunto de entrenamiento. Esto no es suficiente y lo normal es volver a repetir el proceso completo varias veces para ir ajustando el modelo. Cada vez que utilizamos todos los ejemplos de entrenamiento en nuestro modelo se denomina *epoch*. Si decimos que vamos a entrenar un modelo durante 20 epochs significa que vamos a realizar el proceso de dividir en batches de forma aleatorio todos los datos y mostrarlos al modelo 20 veces seguidas.

### Learning rate (ratio de aprendizaje)


Proceso bacth

## Underfiting vs Overfitting


## Fases de proceso de aprendizaje.
- Toma de contacto.
  - Inicialización de parámetros.
- Aprendizaje
- Memorización
  
# Preparación de los datos
## Conjunto de entrenamiento, validación y test.
- Conjunto de entrenamiento vs conjunto de validación.
- Conjunto de test
  
# Flujo del proceso de entrenamiento.
- blucle por epoch
  - Bucle por batch
    - Training
    - Validación

Análisis de resultados entrenamiento vs validación.

# Conclusiones

Método de optimización. 
Proceso iterativo.
Prueba con diferentes hiperparámetros.
Ajustes.

Objetivo -> modelo entrenado
