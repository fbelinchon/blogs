---
layout: post
title: "Introducción a Redes Neuronales"
subtitle: "Entrenamiento y validación de modelos de Deep Learning"
date: 2020-05-19 10:45:13 -0400
background: '/img/posts/06.jpg'
categories: AI
---
# Redes Neuronales, introducción 

## Objetivo del artículo
El contenido de este artículo se centra en dar una visión general de lo que son las redes neuronales. Está destinado a aquellas personas que no están familiarizadas con estas tecnologías o que quieren consolidar los unos conocimientos básicos. Intentaremos evitar, en la medida de lo posible, las demostraciones matemáticas complejas pero si trataremos de explicar cada uno de los mecanismo que conforman una red neuronal básica.

Comencemos.

## la Inteligencia artificial


La Inteligencia Artificial se suele relacionar con robots que se comportan como humanos. Una definición más formal es la siguiente.

>Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving

Por lo tanto estamos hablando de máquinas que intentan simular ciertos comportamientos humanos. En realidad la inteligencia artificial es un tema mucho menos espectacular y exótico de lo que parece. La parte positiva es que, al mismo tiempo, también es mucho más asequible y real de lo que podríamos pensar.

**Machine Learning** es un campo dentro de la inteligencia artificial que se centra en modelos de aprendizaje. Estos modelos aprenden de los ejemplo que se les proporciona para detectar patrones y obtener predicciones. También son capaces de extraer información simplemente analizando un conjunto de datos.
  
>Machine Learning is the field of study that gives computers the ability to learn without
being explicitly programmed.
Arthur Samuel, 1959`


Las **redes neuronales** pertenecen al ámbito de Machine Learning. Hablamos de modelos que son capaces de aprender sin ser explicitamente programados. Se diferencian de otros modelos de machine learning porque tienen una estructura de capas más compleja que permite un aprendizaje más profundo. Por este motivo, al trabajo con redes neuronales también se le conoce como _Deep Learning_. Gracias a esta capacidad las redes neuronales han conseguido resolver problemas que hace unos años parecían inalcanzable, como por ejemplo.

- Reconocimiento de imágenes: en este campo se han realizado avances sorprendentes a nivel de clasificación automática de imágenes, detección de objetos, segmentación, generación automáticas de imágenes (style transfer), super resolución, descripción automática de imágenes etc.
- Tratamiento de lenguaje natural (NLP): clasificación automática de documentos, análisis de sentimiento, traducciones de textos, generación automática de subtítulos en videos …
- Sistemas de recomendaciones: como los que utilizan las grandes compañías para ofrecerte productos basados en tu historial relacionado con el comportamiento de otros usuarios.

La buena noticia es que estás tecnologías no sólo son una realidad, sino que están al alcance de cualquiera.



## Aprendiendo mediante ejemplos
### Otra forma de abordar los problemas

La forma tradicional y más extendida de abordar soluciones desde un punto de vista informático se basa en el desarrollo de algoritmos. 

>ALGORITMO:
"Conjunto ordenado de operaciones sistemáticas que permite hacer un cálculo y hallar la solución de un tipo de problemas"`


Para desarrollar un algoritmos tenemos que conocer todas las variables, dependencias, condiciones y situaciones que se pueden dar.
¿Qué sucede cuando nuestro problema es tan complejo que no somos capaces de conocer toda su casuística?

### Modelos de aprendizaje

En estos casos tenemos que plantear la solución de otra manera.

¿Cómo aprendemos cuando somos pequeños a distinguir los números?

Entre los profesores y nuestros padres nos bombardean a imágenes de números. Al final conseguimos distinguir unos de otros después de cientos y cientos de ejemplos.
No es un problema tan trivial porque cada uno tiene su propia forma de escribir los números.



Si lo analizamos un poco, no es fácil desarrollar un algoritmo que nos permita distinguir números manuscritos pero para nosotros, una vez aprendidos, es la tarea más simple del mundo.

A continuación os pongo algunos ejemplos de extraidos de MNIST (base de datos de números manuscritos).

<img src="/blogs/img/posts/mnist2.jpg" alt="mnist"
	title="base de datos de números manuscritos" width="100%" />

[mnist:](http://yann.lecun.com/exdb/mnist/)
`base de datos de imágenes con 60.000 ejemplo de números manuscritos de tamaño 28x28 pixels. Se utiliza como base de entranamiento en explicaciones sencilla de tratamiento de imagenes.`

*¿Como simulamos esta forma de aprender basada en ejemplos?.*

Empezamos creando un modelo de aprendizaje al que vamos pasando ejemplos de lo que queremos predecir. Tomando el caso anterior le vamos mostrando imágenes de números y le decimos que número es el correcto. Para empezar, nuestro modelo no sabe absolutamente sobre números. Lo que tenemos que conseguir es que nuestro modelo se adapte (aprenda) de forma que cada vez la tasa acierto sea mayor. Esto es lo que llamamos entrenamiento de nuestro modelo.



Es similar a la forma que los humanos tenemos de aprender. Primero recopilamos un buen número de ejemplo (imágenes de números) con su correspondiente valor real (a que número corresponde). Luego le mostramos esa información a nuestro modelo hasta que consigamos que aprenda a distinguirlos. Esta es la forma de trabajar con redes neuronales y con la mayoría de los modelos de machine learning. Parece sencillo, ¿verdad?. Es un proceso iterativo de aprendizaje que finalizará cuando nuestro modelo nos proporcione un índice de aciertos satisfactorio.

Antes de pasar al siguiente apartado vamos a consolidar los conocimientos adquiridos y a relacionarlos con la terminología que se usa más frecuentemente en redes neuronales. Esto nos ayudará a entender mejor cualquier artículo o publicación que nos encontremos sobre el tema.

### ¿Qué puedo hacer con modelos de aprendizaje (machine learning)?

Conceptos como data science, machine learning, deep learning, inteligencia artifial, redes neuronales etc están en boca de todos pero normalmente se relacionan con grandes grandes compañías, fuertes inversiones tecnológicas o perfiles muy especializados.

Esto ha cambiado de forma radical en los últimos años. La capacidad de computación se ha multiplicado y han aparecido herramientas que nos permiten diseñar modelos de aprendizaje de forma sencilla con uno conocimientos básicos. Actualmente podemos entrenar modelos que nos permiten categorizar imagenes en un ordenador de casa en pocos minutos.

*¿Que tipos de problemas podemos resolver con modelos de aprendizaje?*

Básicamente diferenciamos dos tipos de modelos.

- Modelos Supervisados: estos modelos se entrenan con un conjumto de datos de ejemplo. Los datos de ejemplo constan de una serie de características y el resultado final esperado. Una vez entrenado y optimizado, este modelo nos permite predecir el resultado de nuevos datos con las mismas características que los datos de ejemplo. El modelo es capaz de inferir las relaciones y los patrones en los datos de entrenamiento para predecir resultados en datos nuevos. 
  
- Modelos no supervisados: disponemos de un conjunto de datos pero no buscamos ningún resultado. Lo que buscamos es agrupar por comportamientos similares. El ejemplo más común es la segmentación de clientes. Si tenemos datos sufientes de los hábitos de compra de nuestros clientes, podemos segmentar en grupos de usuarios con comportamientos parecidos. Esto nos permite enfocar campañas de marketing personalizadas. Se denominan no supervisados porque no hay un resultado concreto que predecir por cada ejemplo que dirija el entrenamiento como en el caso anterior.

> Otro gran grupo de modelos son los modelos de refuerzo. En este caso los modelos van aprendiendo mediante prueba-error. Cuando las decisiones son correctas se favorece ese camino y cuando no lo es se penaliza.

Para hablar de redes neuronales nos vamos a centrar en los llamamos modelos supervisados. Se dividen en dos grandes grupos.

- Modelos de regresión: devuelven un valor númerico como por ejemplo el valor de compra de una casa. Como entrada del modelo tendremos información de la localización, año de construcción, distrito, metros cuadrados, número de habitaciones etc. Con esa información tenemos que predecir el precio de mercado de esa vivienda.
  
- Modelos de clasificación: devuelven la categoría a la que pertenece la información que le suministramos. En nuestro ejemplo de los números manuscriptos, con la información de los pixeles de la imagen, el modelo nos dirá cual es la categoría más probable (en definitiva, que número hemos escrito). En este apartado entraría también la detección de correos spam (spam o no spam).

## Modelos supervisados
### Entrada al modelo

Para entrenar un modelo de este tipo tenemos que pasarle un conjunto de datos de ejemplo con su correspodiente resultado como hemos comentado anteriormente.

Si hablamos de clasificar números manuscriots, el conjunto de imágenes es la entrada al modelo (input). Cada imagen tendrá una serie de valores para identificar cada pixel (tratamiento de imágenes). Si en lugar de identificar números, queremos predecir el precio de la vivienda tendremos que proporcionarle a nuestro modelo información sobre localización, tamaño, número de habitaciones, año de construcción etc.
Esta información tiene varios nombres.


>Estos valores de entrada se denominan variables independientes **(independent variable)**. Cuando tratamos de definir matemáticamente nuestro modelo a este conjunto de valores se le suele denominar como x.
	
El tratamiento de estos datos antes de pasarlos a cualquier modelo de aprendizaje es todo un mundo.

#### Valor real

Cuando hablamos del valor real nos referimos al número correcto que le hemos asignado a la imagen que estamos mostrando a nuestro modelo. Si la imagen es de un UNO le tenemos que decir a nuestro modelo que eso es un 1. Si el modelo predice un 7 es que está equivocado.
En el caso de calcular el precio de una vivienda nuestro valor real es el precio real de ese ejemplo de vivienda que estamos mostrando a nuestro modelo. Cuando más se ajuste el resultado obtenido por el modelo a este valor real mejor se comportará nuestro modelo.

>Estos valores se denominan  variable dependiente **dependent variable** o etiqueta **label**.

Es el conjunto de datos reales de nuestros ejemplos y que nuestro modelo tiene que aprender a predecir.

#### Nuestro Modelo

El modelo es básicamente un conjunto de operaciones que realizamos sobre la entrada para obtener una salida. Visto así se parece bastante a un algoritmo verdad?. Si ya sabemos las operaciones que hay que realizar sobre nuestra entrada de datos ya tenemos el problema resuelto.

En este caso no conocemos los coeficientes concretos que tenemos que utilizar. Es decir, sabemos que tenemos que multiplicar nuestra entrada por algo y sumarle algo más pero no sabemos exactamente los coeficientes que tenemos que utilizar para que el modelo nos proporcione datos correctos.

Un ejemplo sencillo. Si tenemos la siguiente función

$$f(x) = a·x + b$$

No sabemos el valor de [a] ni [b] que mejor se ajustan a la función que buscamos.

Si nos dicen  $$f(2)=5$$ nos serviría $$a=2$$ y $$b=1$$ pero también $$a=1$$ y $$b=3$$ o $$a=0$$ y $$b=5$$
Si la información es $$f(2)=5$$ y $$f(3) = 7$$ solamente tenemos una solución  $$a=2$$ y $$b=1$$

En los casos reales las funciones son mucho más complejas y no existe una solución exacta. Tendremos que encontrar la solución que mejor se adapte a los datos de ejemplo que tenemos.

Comenzaremos con unos coeficientes aleatorios que iremos actualizando para ir mejorando las predicciones del modelo. Cuando estemos satisfechos de las predicciones de nuestro modelo podemos parar el proceso de aprendizaje y ya tendríamos nuestro modelo listo para predecir cualquier imagen de números que le pasemos.

Como veremos en el siguiente apartado realmente las operaciones básicas de nuestro modelo son precisamente multiplicaciones y sumas. No hay mucho más misterio en ese sentido. Es verdad que son multiplicaciones y suma de matrices. También es verdad que en modelos potentes estas matrices están compuestas por cientos de millones de parámetros.

Si representamos nuestro modelo con una fórmula matemática simple sería.

Si nuestro modelo es $$f(x)=W∗x+b$$     ($$x$$ es nuestro ejemplo de entrada)
La salida de nuestro modelo sería $$\hat{y}=W∗x+b$$

Los valores W y b son los parámetros de nuestro sistema y precisamente son los valores que nuestro modelo tiene que ir actualizar para mejorar sus predicciones. Más concretamente se definen como
- W: pesos (weight)  son los valores por los que multiplicamos nuestra entrada.
- b: bias (sesgo) corresponde con los valores que sumamos al final. Estos valores sirven para ajustar el modelo cuando los valores de x son cero.

El proceso de multiplicar la entrada por un valor y sumarle un segundo valor es lo que se denomina combinación linear. Si simplificamos el proceso y lo vemos como números simple en lugar de matrices vemos que podemos representar cualquier recta en un sistema de coordenadas simplemente seleccionando estos dos factores. El factor bias (sesgo), en este caso, sería el punto de corte de la recta con el eje y (x=0)

### ¿Cómo de bueno es nuestro modelo?

Hemos comentado que $$x$$ es la entrada a nuestro modelo (los ejemplos de imágenes) y que la variable $$y$$ es el valor real. Si estamos diseñando un modelo para predecir un valor numérico es fácil de deducir que el error cometido es:

$$error=(y − \hat{y})$$ la diferencia entre el valor predicho por nuestro modelo y el valor real.

Esta es precisamente la definición de la función de error *loss function* o *cost function*. Tenemos que calcular el error medio cometido en todos nuestros ejemplos.
Para evitar que los errores positivos y negativos se compensen se suele utiliza el valor absoluto o la diferencia al cuadrado de cada error.

Típicas definiciones de funciones de error o de coste serían.


- MAE (mean absolute error):$$\displaystyle \frac{\sum_{i=1}^{n}\|y − \hat{y}\|}{n}$$, sumatorio del valor absoluto de cada uno de los errores dividido por el número de pruebas (n).

- RMSE (root mean square error):$$\displaystyle \sqrt{\frac{\sum_{i=1}^{n}(y-\hat{y})^2}{n}}$$, sumatorio del cuadrado de los errores dividido por el numero de ejemplos que tenemos (n). Finalmente obtiene la raiz cuadrada del número calculado.

La función de coste nos indica el error medio cometido por nuestro modelo al intentar predecir el valor real. Veremos que está función va a desempeñar un papel primordial a la hora de entranar el modelo.

No tenemos que perder de vista cual es nuestro objetivo. Queremos entrenar un modelo que nos permita predecir valores, como el precio de una vivienda o el número manucristo que le pasamos como entrada. No sirve de nada un modelo que solamente predice bien los ejemplo que le hemos pasado para entrenarlo pero que falla al pasarle valores nuevos.


### ¿Cómo entrenar el modelo?

Para mejorar nuestro modelo tenemos que minimizar el error de predicción. Este es un proceso iterativo en el que vamos ajustando nuestros parámetros para ir mejorando el resultado de nuestro modelo.

¿Cómo sabemos en que dirección tenemos que modificar nuestros parámetros para mejorar la predicción?

El objetivo es minimizar nuestro error, es decir, nuestra función de coste. Tenemos que saber en que dirección modificar nuestros parámetros para que esto ocurra. 

Imaginemos que estamos en lo alto de una montaña y queremos llegar al valle. La forma más rápida es seguir el camino que tenga una mayor pendiente. Cuanto mayor sea la pendiente, más rápido nos acercamos al valle. Nuestro problema es similar, tenemos que buscar cual es la pendiente que nos acerca al mínimo de nuestra función. Recordando algo de cálculo.

> La derivada de una función en un punto puede interpretarse geométricamente, ya que se corresponde con la pendiente de la recta tangente a la gráfica de la función en dicho punto.

Esto quiere decir que si obtenemos la derivada parcial (derivada de un parámetros manteniendo constante el resto) sabremos si tenemos que aumentar o disminuir el valor de ese parámetros para disminuir el valor de la función. La derivada parcial es nuestro camino de mayor pendiente al valle.


En realidad se utiliza el gradiente en un punto, que es algo similar a la derivada parcial pero más sencillo de calcular. El método más popular es SGD (Stochastic Gradient Descent).

>SGD: método iterativo de optimización como una aproximación a Gradiente Descent mediante el cáculo del gradiente sobre un subconjunto de los datos seleccionados de forma aleatoria.

Cuando la cantidad de datos es muy elevada el cáculo del gradiente se sustituye por una aproximación utilizando un subconjunto de los datos de forma iterativa. Dicho de otro modo, en lugar de calcular el gradiente de todo el conjunto de los datos se desarrolla un proceso iterativo sobre un subconjunto pequeño de datos seleccionados de forma aleatoria.

El proceso iterativo es el siguiente
1. Se selecciona un subconjunto pequeño de datos
2. Se calcula el gradiente de este conjunto
3. Modificamos nuestros parámetros según el gradiente calculado en el paso anterior.
4. Seleccionamos un conjunto diferente de datos de forma aleatoria volviendo al primer paso.

Al final se utiliza todo el conjunto de datos pero calculando el gradiente de grupos pequeños hasta que utilizamos todos los datos. Lo normal es repetir todo este proceso seleccionando varias veces todo el conjunto de datos para ir optimizando nuestro modelo.


Otro elemento clave es este proceso de optimización es el coeficiente de aprendizaje o **learning rate**. El valor de nuestra pendiente puede ser elevado y corremos el riesgo de estar dando saltos entre aumentar o disminuir el valor de un parámetro. Para controlar este proceso multiplicamos el valor de nuestro gradiente por este coeficiente.

La selección del valor óptimo de **learning rate** que nos permita obtener la mejor optimización de nuestra función de coste es clave para el aprendizaje de nuestro modelo. Valores de **learning rate** muy altos provocaran que nuestro modelo no consiga llegar a un mínimo porque estará dando saltos muy grandes en torno a este punto. Si el **learning rate** es muy pequeño tardaremos mucho tiempo en llegar al mínimo.

# Conclusión

Los modelos de aprendizaje nos proporcionan una nueva herramienta para abordar soluciones que antes parecían inalcanzables. 

Estamos ante una nueva forma de solucionar el 
La Inteligencia Artificial está aquí para quedarse. Muchos de los problems que hace unos años paecían inaborbales ahora se pueden solucionar de forma sencilla y asequible. 