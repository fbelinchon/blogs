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

El concepto del entrenamiento de un modelo de redes neuronales es sencillo y lo hemos mencionado en artículos anteriores. El modelo realiza una predicción sobre los datos de ejemplo que le suministramos y compara la predicción con el valor real que debe proporcionar. El objetivo es que ese error disminuya poco a poco hasta conseguir un modelo con unas predicciones muy cercanas a los valores reales. Para disminuir el error modificamos ligeramente los parámetros de nuestro modelo de forma iterativa.

En los primeros pasos del entrenamiento las predicciones tendrán un error muy grande porque los parámetros del modelo se han seleccionado de modo aleatorio. Por ejemplo, si el modelo intenta predecir si la imagen es de un perro o de un gato la predicción sería como tirar una moneda al aire. Un conjunto aleatorio de parámetros significa que en esos momentos el modelo no sabe nada sobre los datos que tiene que predecir. Según el modelo ajusta los parámetros la tasa de acierto empieza a mejorar. La parte más complicada de este proceso es decidir cuando dejamos de entrenar. Llegado un momento el modelo deja de aprender de los datos de ejemplo y empieza a memorizarlos. Si el modelo empieza a memorizar, realizará predicciones muy exactas sobre los ejemplos de entrenamiento pero las predicciones sobre datos nuevos serán muy inexactas. Posteriormente veremos como detectar ese comportamiento.

La clave del aprendizaje está en ir ajustando los parámetros de forma progresiva para ir disminuyendo el error cometido. El modelo analiza cada parámetro y decide si aumentando o disminuyendo su valor para conseguir el menor error posible.

# Gradient Descent como método de aprendizaje (optimización).

Para decidir si tenemos que aumentar o disminuir los parámetros del modelo recurrimos al concepto de deriva de una función.

> Derivada de una función en un punto: se puede interpretarse geométricamente como la pendiente de la recta tangente a la gráfica de la función en dicho punto.

En definitiva la derivada es la pendiente de una curva en un punto. Si lo que buscamos es el mínimo de una función tiene sentido apoyarnos en la pendiente de esa curva para saber si tenemos que aumentar o disminuir ese parámetro. La pendiente nos informa de con que rapidez varia la función y en que sentido crece o decrece.

En el primer artículo de la serie comparábamos nuestro objetivo de encontrar el mínimo con la búsqueda del camino más rápido para bajar desde el pico de la montaña al valle. En cada paso seleccionábamos la dirección de mayor pendiente para asegurar que bajábamos lo más rápidamente posible.

Nuestro modelo puede tener miles o millones de parámetros con lo que utilizamos lo que se llama derivada parcial de una función con respecto a un parámetro. Esta operación nos devuelve la pendiente de la función en un punto para un parámetro concreto dejando el resto de parámetros fijos. El cálculo de las derivadas parciales de miles de parámetros es una operación muy costosa y en muchos casos inviable. En la práctica utilizamos una aproximación que se denomina *Gradient Descent*.



## Stochastic Gradiente descent SGD.
Para calcular el gradiente en un punto estudiamos como se comporta la función a su alrededor. Aumentamos ligeramente el parámetro y vemos si la función crece o decrece. De esta forma obtenemos valores similares a la deriva parcial de forma menos costosa. Si tenemos el valor de la función en un punto y el valor cuando un poco el parámetro podemos estimar el valor de pendiente en ese punto. No es el valor exacto pero es una buena aproximación para saber si tenemos que aumentar o disminuir el valor de nuestro parámetro para aproximarnos al mínimo de la función de coste.

El método más popular que se utiliza en el entrenamiento de redes neuronales se denomina Stochastic Gradient Descent o simplemente SGD. Es un proceso iterativo que calcula el gradiente de un grupo de ejemplos y con esa información actualiza los parámetros el modelo. No necesitamos conocer en detalle como calcular los gradientes de la función de coste porque todas las librerías de Deep Learning lo hacen por nosotros. Simplemente necesitamos una ligera intuición para entender el mecanismo de optimización. Vamos a detallar el proceso utilizando la nomenclatura propia asociada a redes neuronales.

### Fases del proceso de optimización utilizando SGD
Nuestro conjunto de entrenamiento se divide en pequeños subconjuntos denominados batch o mini-batch. La selección de qué ejemplos entran en cada batch se suele hacer de forma aleatoria. El número de elementos de cada batch se define inicialmente y es el mismo para todos excepto para el último que tendrá el número de ejemplo que queden. Este valor se denomina *batch size* y es una de los principales valores a definir al entrenar un modelo de red neuronal.

Pongamos un ejemplo. Si tenemos un conjunto con 1000 elementos y utilizamos un tamaño de batch (batch Size) igual a 64 tendríamos.
* 15 batches con 64 elementos.
* 1 batch final con 40 elementos que son los que quedan.

Para entrenar el modelo vamos pasando uno por uno nuestros 16 grupos o batches. Por cada batch el modelo calcula la salida, la compara con el resultado que debería producirse y actualiza los parámetros para que la función de error disminuya.
Es importante entender que para un batch solamente hay una actualización de los parámetros. No se produce una modificación de los parámetros por cada uno de los 64 ejemplo sino que se realiza una estimación para el conjunto total de los 64 ejemplos. En estudios realizados se ha demostrado que el cálculo del gradiente de esta forma nos permite aproximarnos al mínimo de una forma más rápida y más consistente que el hecho de calcular el gradiente individual de cada ejemplo.

Cuando hemos completado un ciclo entero, en este caso los 16 batches definidos, el modelo ha recibido todos los ejemplos de nuestro conjunto de entrenamiento. Esto no es suficiente y lo normal es volver a repetir el proceso completo varias veces para ir ajustando el modelo. Cada vez que utilizamos todos los ejemplos de entrenamiento en nuestro modelo se denomina *epoch*. Si decimos que vamos a entrenar un modelo durante 20 epochs significa que vamos a realizar el proceso de dividir en batches de forma aleatorio todos los datos y mostrarlos al modelo 20 veces seguidas.

### Learning rate (ratio de aprendizaje)

El ratio de aprendizaje o learning rate nos permite controlar la forma en que modificamos los parámetros del modelo. Modificamos los parámetros restando el gradiente en ese valor multiplicado por el ratio de aprendizaje. La razón por la que restamos es porque si el gradiente es positivo el mínimo de la función está a la izquierda del valor actual por lo que tenemos que disminuir el parámetro. La razón por la que multiplicamos el gradiente por el learning rate es para controlar cuanto modificamos el valor del parámetro.

Valores típicos de learning rate pueden variar entre 0,01 a 0,00001. Suelen ser valores menores de uno para evitar estar saltando de izquierda a derecha cuando nos acercamos al mínimo. También nos sirve para salvar algunos problemas que nos podemos encontrar en nuestra función de coste.

* Mínimos locales: son zonas de la función donde se produce un mínimo local que no es el mínimo general. Si el learning rate es muy pequeño el entrenamiento de la función se quedara en ese punto pensando que es el mínimo de la función. No conseguiremos un aprendizaje optimo.
* Zonas planas: si encontramos una zona muy plana el gradiente en esos puntos sera muy pequeño y no podremos avanzar en la búsqueda del mínimo global. Necesitamos un learning rate alta para salir de estas zonas.
* Mínimo con una curva muy estrecha: en este caso necesitamos tener un learning rate muy pequeño porque en caso contrario estaríamos saltando de izquierda a derecha indefinidamente sin acercarnos al mínimo.

Para empezar a entrenar un modelo se pueden ir probando diferentes learning rate y ver cual nos proporciona mejores resultados. A la hora de optimizar más y buscar mejores resultados se utilizan learning rate variable con el objetivo de saltar mínimos locales y zonas planas y acercarse al mínimo general lo más posible. Esto es lo que se llama learning rate annealing y existen varios métodos que dan muy buenos resultados. En general empiezan con learning rate más bajos que aumentan rápidamente su valor hasta llegar a un máximo. Luego descienden de forma más moderada hasta terminar con un descenso mucho más ligero.

El ratio de aprendizaje nos permite controlar en que proporción modificamos los parámetros del modelo según su gradiente. Es otro de los valores que tenemos que configurar y probar al entrenar un modelo de redes neuronales.

# Underfitting vs Overfitting

Ya hemos hablado brevemente de estos dos problemas que nos podemos encontrar al entrenar un modelo pero es ahora cuando podemos profundizar y ver como podemos detectarlos y minimizarlo.

## Undefitting

Este problema aparece cuando nuestro modelo es demasiado simple o nuestro conjunto de datos es demasiado reducido para la tarea que queremos resolver. En definitiva o no tenemos suficientes datos o no tenemos un modelo suficientemente potente.

¿Como lo detectamos?. No es complicado. Llegado un punto veremos que el modelo no mejora sus resultados y siguen estando muy lejos de los deseados. Veremos que los resultados para el conjunto de datos de entrenamiento y para el conjunto de validación son igual de pobres. Si estamos trabajando con una red neuronal tenemos dos opciones.

* Añadir más complejidad al modelo: esto lo conseguimos añadiendo más neuronas a las capas del modelo o incorporando más capas intermedias (hidden layers).
* Buscamos más datos de ejemplo: cuantos más datos de ejemplo tengamos más posibilidades hay de que el modelo aprenda nuevas características o relaciones entre los datos que estamos pasando. Los modelos de redes neuronales necesitan una gran cantidad de datos para poder aprender y dar resultados óptimos. Si no tenemos demasiados datos existen otras técnicas como transfer learning que nos permiten utilizar modelos ya entrenados que podemos utilizar como base de nuestro propio modelo.

## Overfitting

Este problema es más difícil de detectar. Nos encontramos con el caso contrario. Nuestro modelo es demasiado complejo y en lugar de aprender de los datos de ejemplo empieza a memorizarlos. Esto es un problema grave porque veremos que nuestro modelo proporciona unos resultados muy buenos en nuestros datos de ejemplo pero no generaliza bien. Los resultados en el conjunto de validación (datos que no se utilizan para entrenar el modelo) van empeorando según vamos entrenando el modelo. Imaginate que el modelo aprende de memoria que una de las imágenes es un gato pero no es capaz de deducirlo de la forma del animal, de las orejas, de las proporciones del cuerpo etc. Si le mostramos una imagen de otro gato distinto que no se ha utilizado en su entrenamiento no sera capaz de reconocerlo.

¿Que podemos hacer?
* Obtener más datos de ejemplo: Si tenemos más datos sera más difícil que el modelo los memorice todos y sera más fácil que detecte las características que diferencias unas imágenes de otras.
* Simplificar el modelo: si nuestro modelo es más sencillo significa que tiene menos parámetros. Los parámetros son como células de memoria. Si no tenemos los suficientes no podemos memorizar todas las imágenes y, como en el caso anterior forzamos al modelo a aprender. Aunque parezca mentira en estos casos lo que tenemos que hacer es dificultar el aprendizaje de nuestro modelo para forzarlo a deducir características y relaciones. Este proceso se denomina regularización y consiste exactamente en eso, en ponerle las cosas difíciles a nuestro modelo. Por ejemplo, una de las técnicas más usadas se denomina Dropout y consiste en poner a cero un porcentaje aleatorio de los parámetros de nuestro modelo en cada iteración de aprendizaje. De esa forma forzamos a que el modelo no preste siempre todas la atención en los mismo parámetros y encuentre nuevas relaciones y patrones.

## Fases de proceso de aprendizaje.

El proceso de aprendizaje de un modelo de redes neuronales pasa por tres fases diferentes.

1. Desconocimiento del problema: cuando empezamos el proceso de entrenamiento los parámetros se inicializan de forma aleatorio. El modelo no tiene conocimiento alguno de que es lo que tiene que predecir y en las primeras iteraciones los resultados son poco menos que tirar una moneda al aire. El valor de los parámetros puede estar muy lejos de su valor óptimo.

2. Proceso de aprendizaje: después de algunas iteraciones el modelo comienza a obtener resultados aceptables. En esta fase el modelo empieza a descubrir patrones y relaciones entre parámetros que le ayudan predecir con mayor fiabilidad. El objetivo es llegar a esta fase en el menor tiempo posible. Si nuestro modelo sufre de underfitting solamente es posible que no lleguemos a esta fase de aprendizaje y que los resultados no sean los deseados.

3. Proceso de memorización: en esta fase final el modelo deja de aprender patrones y relaciones para empezar a memorizar los ejemplos que les vamos pasando. Por así decirlo se acomoda. Si es modelo es demasiado complejo o tenemos pocos datos de aprendizaje, es más fácil memorizar los datos en lugar de deducir el resultado de las características de los mismo. Este fenómeno es lo que hemos llamado overfitting. Tenemos que evitar llegar a esta fase porque nuestro modelo sería inservible para predecir nuevos datos.
  
# Preparación de los datos
## Conjunto de entrenamiento, validación y test.
Ya hemos comentado que necesitamos un conjunto de datos para entrenar el modelo. Necesitamos además de cantidad una buena calidad. Al hablar de calidad nos referimos a que sean una representación lo más realista posible de los datos que vamos a encontrar en la realizada. La calidad del modelo entrenado depende directamente de la calidad de los datos. Por ejemplo, si intentamos categorizar de forma automática una serie de imágenes y de una categoría tenemos significativamente menos datos que del resto, ser'a muy difícil detectar correctamente esta categoría.
## Conjunto de validación
También hemos hablado brevemente de la necesidad de tener un conjunto de datos para ir validando el modelo durante el entrenamiento. Este conjunto de validación se excluye del proceso de aprendizaje y solamente se utiliza para verificar que el modelo se comportará de forma adecuada con datos nuevos, que no ha utilizado en la fase de entrenamiento.

Normalmente se suele reservar entre un 10% y un 20% de los datos de entrenamiento para crear el conjunto de validación. Al igual que en los datos de entrenamiento necesitamos una distribución representativa para validar el modelo con todo el espectro de datos reales. En estos datos de validación es incluso más importante. Tenemos que intentar simular de la forma más exacta posibles el tipo de datos que se va a encontrar el modelo una vez que ya está entrenado. Por ejemplo, si los datos tienen una fuerte componente temporal el conjunto de validación se suele tomar como las ultimas fechas de los datos que tenemos de entrenamiento. De esa forma simulamos los datos que va a tener que predecir el modelo cuando esté que ---serán siempre posteriores a los utilizados para el entrenamiento.

## Conjunto de test

Todavía tenemos un conjunto más de datos del que no hemos hablado. En proyectos reales se suele separar otro conjunto de datos que se denomina de test para realizar las validaciones finales del modelo. ¿En que se diferencia del conjunto de validación?

El objetivo es el mismo pero se utilizan de forma distinta. El entrenamiento de modelos de redes neuronales se basa en la experimentación. Se realizan múltiples pruebas con diferentes opciones y diferentes modelos. Al realizar tantas pruebas se puede dar el caso que favorezcamos aquellos modelos o aquellas opciones que nos dan mejores resultados con respecto a nuestro conjunto de validación. La forma de trabajar es la siguiente:
- Primero realizamos las pruebas con diferentes modelos y diferentes parametrizaciones. Para evaluar sus resultados utilizamos el conjunto de validación.
- Finalmente seleccionamos solamente aquellos modelos que pensamos que son los mejores y los testeamos con los datos de test. Si los datos son consistentes podemos seleccionar el modelo que mejor resultado nos dé. Si por el contrario, los resultados no son similares a los obtenidos con el conjunto de validación significa que el conjunto de validación no está bien diseñado. Tendremos que analizar los resultados y volver a definir un conjunto de validación que represente mejor los datos reales.

La principal diferencia es que los datos de validación los utilizamos constantemente para experimentar y probar diferentes modelos y parámetros. Los datos de test solamente los utilizamos en la fase final de decisión como validación final.

  
# Flujo del proceso de entrenamiento.

Antes de empezar a entrenar el modelo tenemos que definir lo que se llaman hiper-parámetros que van a influir de forma decisiva en el proceso. Varios de estos parámetros ya los hemos mencionado anteriormente.
- batchsize: el tamaño de los batchs en que agrupamos los datos de entrenamiento
- Numero de epoch: numero de veces que vamos a mostrar todo el conjunto de datos a nuestro modelo para el entrenamiento.
- Learning rate: valor por el que vamos a multiplicar el gradiente antes de restar el valor al parámetro correspondiente para controlar el proceso de ajuste.
- Arquitectura del modelo: numero de hidden layer y cantidad de neuronas en cada capa.

Además hay que decidir la función de coste y el algoritmo de optimización. Hemos comentado que se utiliza SGD pero existen varias variantes que optimizan el proceso de ajuste.

Estos hiperparámetros son los que nos permitirán experimentar hasta encontrar la combinación optima para que nuestro modelo puede predecir la información con la mayor exactitud posible.
Una vez seleccionados estos parámetros el proceso de entrenamiento es siempre el mismo. Explicado en seudocódigo sería de la siguiente manera.

`Blucle epoch in 1.. número epochs
   
  Para datos de entrenamiento
  Bucle datos in 1..número de grupos (total de ejemplos de entrenamiento%batchsize + 1)
    resultado= modelo(datos)
    error_train= función de coste(resultado, valor real)
    gradiente= gradiente de la función de coste
    parámetros=parámetros - learning rate * gradiente
    
  Para datos de validación
  Bucle datos in 1.. número de grupos (total de ejemplos de validación%batchsize +1)
    resultado= modelo(datos)
    error_validacion = función coste (resultado, valor real)
  
  print('Ciclo: ' + epoch)
  print('error de entrenamiento: '+ media(error_train))
  print('error de validacion: '+ media(error_validacion))`
  
Vamos a explicarlo con un ejemplo sencillo. Supongamos que queremos construir un modelo para diferenciar entre fotografías de perros y gatos (muy típico). Tenemos un total de 1200 imágenes perfectamente etiquetadas. La mitad de las fotografías son de perros y la otra mitad de datos.

Primero definimos el conjunto de datos de entrenamiento y de validación. Tomaremos de forma aleatorio 1000 imágenes de entrenamiento y 200 de validación. Procuraremos tener aproximadamente el 50% de perros y gatos en cada conjunto.

A continuación seleccionamos los hiperparámetros que vamos a utilizar en nuestra primera prueba.
* Numero de epoch: 10
* Batchsize: 16
* Learning rate: 0,003
* Modelo con tres capas ocultas cada una con 300 neuronas.
* Las imágenes las redimensionamos a 300x300 para facilitar el aprendizaje y optimizar los recursos.
* Función de coste: para este caso concreto probablemente utilizaríamos cross entropy loss. Esta función de coste nos permite cuantificar el error producido en la identificación de una imagen dentro de una categoría. El modelo proporciona un porcentaje de exactitud, es decir, 70% de que la imagen sea un gato. Si la imagen es realmente un gato el error cometido ser'a menor que si la imagen es de un perro.
* Optimizador: SGD básico.

Con los datos que tenemos entendemos que vamos a repetir un ciclo completo de aprendizaje 10 veces. Por cada epoch vamos a realizar una fase de entrenamiento que actualizar'a los parámetros más una fase de validación posterior para testear el modelo con datos no utilizados en el entrenamiento. Una vez terminado un ciclo tendremos información del error medio cometido por el modelo tanto para el conjunto de entrenamiento como para el conjunto de validación.

Por cada ciclo tenemos
1. Fase de entrenamiento. Pasamos al modelo grupos de 16 imágenes. Como el numero de ejemplos es 1000 tendremos un total de 62 grupos de 16 elemento más un grupo final de 8 imágenes. En total 63 grupos. Por lo tanto definiremos un bucle de 1 a 63 para pasar al modelo todos los datos que tenemos en cada ciclo. En cada nuevo ciclo los grupos se vuelven a generar de forma aleatorio de forma que los componentes de cada grupo cambian.

Por cada grupo obtenemos el resultado de nuestro modelo y calculamos el error cometido. La función de coste es la que hemos definido al inicio como un hiperparámetro.
A continuación obtenemos el gradiente de cada parámetro. Estos cálculos los realiza cualquier librería de redes neuronales que utilicemos por que no tenemos que preocuparnos por calcularlo nosotros. Con saber que es el gradiente y cual es su propósito es suficiente para empezar.

Finalmente a cada parámetro le restamos su gradiente multiplicado por el ratio de aprendizaje (learning rate). Esta actualización de los parámetros representa el autentico aprendizaje del modelo.

2. Fase de validación. Realizamos los mismo pasos que en la fase anterior excepto la actualización del modelo. En esta fase utilizamos el conjunto de validación por lo que el modelo no puede utilizar estos datos para optimizar los parámetros. Esta fase se utiliza para validar como se comporta el modelo con datos que no ha visto en la fase de aprendizaje.

En este caso tenemos 200 imágenes que utilizando el mismo batchsize serían 12 grupos de 16 elementos más uno grupo final con 8. En total 13 grupos.

Por lo tanto tendremos un bucle de 13 grupos que pasaremos al modelo y obtendremos posterior su error medio.

Finalmente por cada ciclo mostraremos en consola el numero de ciclo y el error tanto del conjunto de entrenamiento como del conjunto de validación. De esta forma podemos ir comprobando como se comporta el modelo y detectar cuando se produce el fenómeno de overfitting (el error de entrenamiento sigue disminuyendo pero el error de validación comienza a aumentar).

En proyectos reales se añade también lo que llamamos una medida para facilitar el análisis del modelo. En este caso podríamos mostrar el porcentaje de acierto del modelo al predecir si la imagen es un perro o un gato. La medida se calcula en cada iteración igual que el error y se muestra al final de cada ciclo igual que el error.

Ahora solamente nos resta analizar los datos que nos ha devuelto el modelo y seguir experimentando con diferentes hiperparámetros para encontrar la combinción que nos de el mejor resultado.


`NOTA: En un caso real para este problema utilizaríamos una red convolucional y posible transfer learning para agilizar y mejorar los resultados. Pero el ejemplo es válido para explicar un proceso general de entrenamiento de una red neuronal.`


# Conclusiones

Método de optimización. 
Proceso iterativo.
Prueba con diferentes hiperparámetros.
Ajustes.

Objetivo -> modelo entrenado
