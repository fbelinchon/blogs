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

El ratio de aprendizaje o learning rate nos permite controlar la forma en que modificamos los parametos del modelo. Modificamos los parametros restando el gradiente en ese valor multiplicado por el ratio de aprendizaje. La razon por la que restamos es porque si el gradiente es positivo el minimo de la funci'on est'a a la izquierda del valor actual por lo que tenemos que disminuir el parametro. La raz'on por la que multiplicamos el gradiente por el learning rate es para controlar cuanto modificamos el valor del parametro.

Valores tipicos de learning rate pueden variar entre 0,01 a 0,00001. Suelen ser valores menores de uno para evitar estar saltando de izquierda a derecha cuando nos acercamos al m'inimo. Tambi'en nos sirve para salvar algunas problemas que nos podemos encontrar en nuestra funci'on de coste.

* M'inimos locales: son zonas de la funcion donde se produce un minimo local que no es el m'inimo general. Si el learning rate es muy pequeno el entrenamiento de la la funci'on se quedara en ese punto pensando que es el minimo de la funcion. No consigueremos un aprendizaje optimo.
* Zonas planas: si encontramos una zona muy plana el gradiente en esos puntos sera muy pequeno y no podremos avanzar en la busqueda del minimo global. Necesitamos un learning rate alta para salir de estas zonas.
* M'inimo con una curva muy estrecha: en este caso necesitamos tener un learning rate muy pequeno porque en caso contrario estariamos saltanto de izquierda a derecha indefinidmente sin acercarnos al minimo.

Para empezar a entrenar un modelo se pueden ir probando diferentes learning rate y ver cual nos proporciona mejores resultados. A la hora de optimizar mas y buscar mejores resultados se utilizan learning rate variable con el objetivo de saltar minimos locales y zonas planas y acercarse al minimo general lo mas posible. Esto es lo que se llama learning rate annealing y existen varios metodos que dan muy buenos resultados. En general empiezan con learning rate m'as bajos que aumentan rapidamente su valor hasta llegar a un maximo. Luego descienden de formaa mas moderada hasta terminar con un descenso mucho mas ligero.

El ratio de aprendizaje nos permite controlar en que proporcion modificamos los parametros del modelo seg'un su gradiente. Es otro de los valores que tenemos que configurar y probar al entrenar un modelo de redes neuronales.

# Underfiting vs Overfitting

Ya hemos hablado brevemente de estos dos problemas que nos podemos encontrar al entrenar un modelo pero es ahora cuando podemos profundizar y ver como podemos detectarlos y minimizarlo.

## Undefiting

Este problem aparece cuando nuestro modelo es demasiado simple o nuestro conjunto de datos es demasiado reducido para la tarea que queremos resolver. En definitiva o no tenemos suficientes datos o no tenemos un modelo suficientemente potente.

?Como lo detectamos?. No es complicado. LLegado un punto veremos que el modelo no mejora sus resultados y siguen estando muy lejos de los deseados. Veremos que los resultados para el conjunto de datos de entrenamiento y para el conjunto de validaci'on son igual de pobres. Si estamos trabajando con una red neuronal tenemos dos opciones.

* Anadir mas complejidad al modelo: esto lo conseguimos anadiendo mas neuronas a las capas del modelo o incorporando mas capas intermedias (hidden layers).
* Buscamos mas datos de ejemplo: cuantos mas datos de ejemplo tengamos mas posibilidades hay de que el modelo aprenda mas caracticas o relaciones entre los datos que estamos pasando. Los modelos de redes neuronales necesitan una gran cantidad de datos para poder aprender y dar resultados optimos. Si no tenemos demasiados datos existen otras tecnicas como transfer learning que nos permiten utilizar modelos ya entrenados que podemos utilizar como base de nuestro propio modelo.

## Overfiting

Este problema es mas dificil de detectar. Nos encontramos con el caso contrario. Nuestro modelo es demasiado complejo y en lugar de aprender de los datos de ejemplo empieza a memorizarlos. Esto es un problema grave porque veremos que nuestro modelo proporciona unos resultados muy buenos en nuestros datos de ejemplo pero no generaliza bien. Los resultados en el conjunto de validaci'on (datos que no se utilizan para entrenar el modelo) van empeorando seg'un vamos entrenando el modelo. Imaginate que el modelo aprende de memoria que una de las imagenes es un gato pero no es capaz de deducirlo de la forma del animal, de las orejas, de las proporciones del cuerpo etc. Si le mostramos una imagen de otro gato distinto que no se ha utilizado en su entrenamiento no sera capaz de reconocerlo.

?Que podemos hacer?
* Obtener mas datos de ejemplo: Si tenemos mas datos sera mas dificil que el modelo los memorice todos y sera mas facil que detecte las caracteristicas que diferencias unas imagenes de otras.
* Simplificar el modelo: si nuestro modelo es mas sencillo significa que tiene menos parametros. Los parametros son como celulas de memoria. Si no tenemos los suficientes no podemos memorizar todas las imagenes y, como en el caso anterior forzamos al modelo a aprender. Aunque parezca mentira en estos casos lo que tenemos que hacer es dificultar el aprendizaje de nuestro modelo para forzarlo a deducir caracteristicas y relaciones. Este proceso se denomina regularizacion y consiste exactamente en eso, en ponerle las cosas dificiles a nuestro modelo. Por ejemplo, una de las tecnicas mas usadas se denomina Dropout y consiste en poner a cero un porcentaje aleatorios de los parametros de nuestro modelo en cada iteracion de aprendizaje. De esa forma forzamos a que el modelo no preste siempre todas la atenci'on en los mismo parametros y encuentre nuevas relaciones y patrones.

## Fases de proceso de aprendizaje.

El proceso de aprendizaje de un modelo de redes neuronales pasa por tres fases diferentes.

1. Desconocimiento del problema: cuando empezamos el proceso de entrenamiento los par'ametros se inicializan de forma aleatorio. El modelo no tiene conocimiento alguno de que es lo que tiene que predecir y en las primeras iteraciones los resultados son poco menos que tirar una moneda al aire. El valor de los par'ametros puede estar muy lejos de su valor 'optimo.

2. Proceso de aprendizaje: despu'es de algunas iteraciones el modelo comienza a obtener resultados aceptables. En esta fase el modelo empieza a descubrir patrones y relaciones entre par'ametros que le ayudan predecir con mayor fiabilidad. El objetivo es llegar a esta fase en el menor tiempo posible. Si nuestro modelo sufre de underfiting solamente es posible que no lleguemos a esta fase de aprendizaje  y que los resultados no sean los deseados.

3. Proceso de memorizaci'on: en esta fase final el modelo deja de aprender patrones y relaciones para empezar a memorizar los ejemplos que les vamos pasando. Por as'i decirlo se acomoda. Si es modelo es demasiado complejo o tenemos pocos datos de aprendizaje, es m'as f'acil memorizar los datos en lugar de deducir el resultado de las caracter'isticas de los mismo. Este fen'omeno es lo que hemos llamado overfiting. tenemos que evitar llegar a esta fase porque nuestromodelo ser'ia inservible para predecir nuevos datos.
  
# Preparación de los datos
## Conjunto de entrenamiento, validación y test.
Ya hemos comentado que necesitamos un conjunto de datos para entrenar el modelo. Necesitamos adem'as de cantidad una buena calidad. Al hablar de calidad nos referimos a que sean una representaci'on lo mas realista posible de los datos que vamos a encontrar en la realizada. La calidad del modelo entrenado depende directamente de las calidad de los datos. Por ejemplo, si intentamos categorizar de forma autom'atica una serie de imagenes y de una categoria tenemos significativamente menos datos que del resto, ser'a muy dificil detectar correctamente esta categoria.
## Conjunto de validaci'on
Tambi'en hemos hablado brevemente de la necesidad de tener un conjunto de dtos para ir validando el modelo durante el entrenamiento. Este conjunto de validaci'on se excluye del proceso de aprendizaje y solamente se utiliza para verificar que el modelo se comportar'a de forma adecuada con datos nuevos, que no ha utilizado en la fase de entrenamiento.

Normalmente se suele reservar entre un 10% y un 20% de los datos de entrenamiento para crear el conjunto de validaci'on. Al igual que en los datos de entrenamiento necesitamos una distribuci'on representativa para validar el modelo con todo el espectro de datos reales. En estos datos de validaci'on es incluso m'as importante. Tenemos que intentar simular de la forma m'as exacta posibles el tipo de datos que se va a encontrar el modelo una vez que ya esta entrenado. Por ejemplo, si los datos tienen una fuerte componente temporal el conjunto de validaci'on se suele tomar como las ultimas fechas de los datos que tenemos de entrenamiento. De esa forma simulamos los datos que va a tener que predecir el modelo cuando est'e que ser'an siempre posteriores a los utilizados para el entrenamiento.

## Conjunto de test

Todav'ia tenemos un conjunto m'as de datos del que no hemos hablado. En proyectos reales se suele separar otro conjunto de datos que se denomina de test para realizar las validaciones finales del modelo. ?En que se diferencia del conjunto de validacion?

El objetivo es el mismo pero se utilizan de forma distinta. El entrenamiento de modelos de redes neuronales se basa en la experimentaci'on. Se realizan multiples pruebas con diferentes opciones y diferentes modelos. Al realizar tantas pruebas se puede dar el caso que favorezcamos aquellos modelos o aquellas opciones que nos dan mejores resultados conr especto a nuestro conjunto de validaci'on. La forma de trabajar es la siguiente:
- Primero realizamos las pruebas con diferentes modelos y diferentes parametrizaciones. Para evaluar sus resultados utilizamops el conjuto de validacion.
- Finalmente seleccionamos solamente aquellos modelos que pensamos que son los mejores y los testeamos con los datos de test. SI lod datos son consistentes podemos seleccionar el modelo que mejor resultado nos de. Si por el contrario, los resultados no son similares a los obtenidos con el conjunto de validacion significa que el conjuntod e validacion no esta bien disenado. Tendremos que alnalizar los resultados y volver a definir un conjunto de validaci'on que represente mejor los datos reales.

La principal diferencia es que los datos de validaci'on los utilizamos constatemente para experimentar y probar diferentes modelos y par'ametros. Los datos de test solamente los utilizamos en la fase final de decisi'on como validaci'on final.

  
# Flujo del proceso de entrenamiento.

Antes de empezar a entrenar el modelo tenemos que definir lo que se llaman hiper-parametros que van a influir de form decisiva en el proceso. Varios de estos par'ametros ya los hemos mencionado anteriormente.
- batchsize: el tamano de los batch en que agrupamos los datos de entrenamiento
- Numero de epoch: numero de veces que vamos a mostrar todo el conjunto de datos a nuestro modelo para el entrenamiento.
- Learning rate: valor por el que vamos a multiplicar el gradiente antes de restar el valor al parametro correspondiente para controlar el proceso de ajuste.
- Arquitectura del modelo: numero de hidden layer y cantidad de neuronas en cada capa.

Adem'as hay que decidir la funci'on de coste y el algoritmo de optimizaci'on. Hemos comentado que se utiliza SGD pero existen varias varientes que optimizan el proceso de ajuste.

Estos hiper-parametros son los que nos permitiran experimentar hasta encontrar la combinaci'on optima para que nuestro modelo puede predecir la informaci'on con la mayor exactitud posible.
Una vez selkeccionados estos parametros el proceso de entrenamiento es siempre el mismo. Explicado en seudoc'odigo ser'ia de la siguiente manera.

`Blucle epoch in 1.. numero epochs
   
  Para datos de entrenamiento
  Bucle datos in 1..numero de grupos (total de ejemplos de entrenamiento%batchsize + 1)
    resultado= modelo(datos)
    error_train= funci'on de coste(resultado, valor real)
    gradiente= gradiente de la funcion de coste
    parametros=parametros - learning rate * gradiente
    
  Para datos de validacion
  Bucle datos in 1.. numero de grupos (total de ejemplos de validacion%batchsize +1)
    resultado= modelo(datos)
    error_validacion = funcion coste (resultado, valor real)
  
  print('Ciclo: ' + epoch)
  print('error de entrenamiento: '+ media(error_train))
  print('error de validcion: '+ media(error_validacion))`
  
Vamos a desarrollarlo un poco.

1. Definimos el numero de epochs que es el numero de ciclos completos que vamos a realizar. Un ciclo completo significa que utilizamos todos los datos.
1.1. Primero tomamos los datos de entrenamiento. De forma aleatoria agrupamos los datos en grupos de un tamano igual al batchsize. El 'ultimo grupo normalmente tiene un tamano menor.
1.2. definimos un nuevo bucle para recorrer todos los grupos definidos en el paso anterior.

Análisis de resultados entrenamiento vs validación.

# Conclusiones

Método de optimización. 
Proceso iterativo.
Prueba con diferentes hiperparámetros.
Ajustes.

Objetivo -> modelo entrenado
