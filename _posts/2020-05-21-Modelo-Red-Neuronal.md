---
layout: post
title: "Modelos simples a Redes Neuronales"
subtitle: "Entendiendo modelos simples de Redes Neuronales"
date: 2020-05-20 10:45:13 -0400
background: '/img/posts/06.jpg'
categories: AI
mathjax: true
---
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
# Modelo simple

## Objetivo.
El objetivo del artículo es entender como se define un modelo de apredizaje. Nos centraremos 
en la definición de modelos de redes neuronales sencillas para entender los mecanismos básicos.
Las redes neuronales se denominan **deep learning** porque se basan en arquitecturas de varios niveles que se
conectan de forma encadenada hasta producir el resultado final.

## Configuración simple: red neuronal de una sola capa

### Perceptron
El modelo más básico de red neuronal es el perceptrón. Está definido por una neurona que acepta una entrada con varios parámetros y produce una salida. Idealmente la salida es 0 o 1. Para calcular el resultado final se multiplica cada entrada por un peso que nos indica el aporte de esa entrada al resultado final.

 Al sumatorio de todas esas operaciones se le compara con un valor constante para devolver un 1 si el sumatorio es mayor que ese valor o un 0  en caso contrario. Si nos damos cuenta tenemos los mismo elementos que describimos en el árticulo anterior.
- Una entrada de datos.
- Una matriz (en este caso un vector) de pesos.
- Un parámetro de ajuste para indicar el umbral del cero que denominamos bias y que se añade a la multiplicación de ,los valores anteriores.
- El resultado de nuestro modelo. En este caso concreto sería 0 o 1.

Este modelo lo podemos representar matemáticamente como:



$$\hat{y}=W*x + b$$

<img src="/blogs/img/perceptron.png" alt="perceptron"
	title="perceptron" width="80%" />


> Imagen perceptrón con cinco entradas tomada de Wikipedia

Para ajustar la salida de un perceptrón que sería de 0 o 1 podemos indicar que el valor será un 1 si el resultado es positivo y cero en caso contrario.

El perceptrón es un modelo muy simple donde solamente tenemos una neurona y una sola capa. Como veremos más adelante las redes neuronales constan de cientos de neuronas en cada capa y de varias capas que se encadenan unas a otras. Es decir, la salida de una capa es la entrada a la siguiente.



### Multiplicación de matrices

El ejemplo del perceptron se define matemáticamente como una multiplicación de vectores. Al multiplicar el vector entrada $$ \vec{x} $$ por el vector $$ \vec{w} $$ de los pesos tenemos las operaciones de multiplicación que hemos definido.


$$
\left[\begin{array}{ccc}x_1& x_2 & x_3\end{array}\right] \left[\begin{array}{ccc}w_1 \\ w_2 \\ w_3\end{array}\right] = x_1*w_1+x_2*w_2+x_3*w_3
$$

Si queremos añadir más neuronas a nuestro modelo necesitamos tratar con multiplicaciones de matrices. Imaginemos que se añade una nueva neurona. Tendrá el mismo número de entradas pero tendremos dos resultados de nuestro modelo, uno por cada neurona. Cada nueva columna de nuestra matriz de pesos representa una neurona con sus correspondietnes pesos que se aplican a cada entrada. Si lo expresamos como operaciones de matrices trandríamos lo siguiente.


$$
\left[\begin{array}{ccc}x_1& x_2 & x_3\end{array}\right] \left[\begin{array}{ccc}w_{1n1}&w_{1n2} \\ w_{2n1} & w_{2n2}\\ w_{3n1} & w_{3n2}\end{array}\right] = \begin{cases} x_1*w_{1n1}+x_{2}*w_{2n1}+x_3*w_{3n1} \\ x_1*w_{1n2}+x_{2}*w_{2n2}+x_3*w_{3n2}\end{cases}
$$

Hemos numerado los índices de los pesos de forma que el primer elemento es el orden del peso y el segundo la neurona. Por ejemplo $$ w_{3n2} $$ sería el peso que se aplica a la tercera entrada perteneciente a la segunda neurona. Normalmente en matrices se habla de filas y columnas (en ese orden). A partir de ahora utilizaremos esa nomenclatura donde el primer valor (fila) es el orden del peso y el segundo (columna) es la neurona, $$w_{32}$$ (es el tercer parámetro de la segunda neurona)

Es importante tener claro unos conceptos básicos de multiplicación de matrices.
- Para multiplicar dos matrices necesitamos que el número de columnas de la primera matriz sea igual al número de filas de la segunda matriz. Tiene sentido porque el número de parámetros de entrada tiene que ser igual al número de pesos que tiene cada neurona.
- La matriz resultado tendrá el mismo número de filas que la primera matriz y el mismo número de columnas de la segunda. También concuerda con lo que hemos explicado. Si tenemos como entrada un vector $$\vec{x} = \left[\begin{array}{ccc}x_1& x_2 & x_3\end{array}\right]$$ (1 fila y 3 columnas) y nuestro modelo tiene dos neuronas (3 filas y 2 columnas) el resultado tendría una fila y dos columnas, una columnas para el resultado de cada neurona. Simplificando, el resultado de nuestro modelo sería igual al número de registros de entrada (en nuestro ejemplo uno) y tantas columnas como neuronas tenemos $$X(1,3) * W(3,2) = R(1,2)$$
- El orden de la multiplicación afecta al resultado. No es lo mismo multiplicar $$A*B$ que $$B*A$$.
  
  > Nota: Si $$A*B = R$$ entonces $$B^T*A^T =R^T$$. Eso quiere decir que si cambiamos el orden de los factores de la multiplicación tenemos que utilizar las matrices traspuestas de cada uno de ellos y el resultado será la traspuesta de $A*B$. La traspuesta de una matriz se obtiene cambiando filas por columnas.
  
Para aclarar un poco más el tema podemos poner un ejemplo donde como entrada tenemos dos registros y nuestro modelo tiene 3 neuronas. $$X(2,3) * W(3,3) = R(2,3)$$

$$
\left[\begin{array}{ccc}x_{11}& x_{12} & x_{13} \\ x_{21}& x_{22} & x_{23}\end{array}\right] * \left[\begin{array}{ccc}w_{11}&w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23}\\ w_{31} & w_{32}& w_{33}\end{array}\right] = \left[\begin{array}{ccc} r_{11}  & r_{12} & r_{13}\\ r_{21}  & r_{22} & r_{23}\end{array}\right]
$$

$$ 
r_{11} = x_{11}*w_{11} + x_{12}*w_{21} + x_{13}*w_{31}
$$

$$
r_{12} = x_{11}*w_{12} + x_{12}*w_{22} + x_{13}*w_{32}
$$

$$
r_{13} = x_{11}*w_{13} + x_{12}*w_{23} + x_{13}*w_{33}
$$

$$
r_{21} = x_{21}*w_{11} + x_{22}*w_{22} + x_{23}*w_{31}
$$

$$
r_{22} = x_{21}*w_{12} + x_{22}*w_{22} + x_{23}*w_{32}
$$

$$
r_{23} = x_{21}*w_{13} + x_{22}*w_{23} + x_{23}*w_{33}
$$


La primera fila del resultado es el valor de cada una de las tres neuronas para el primer registro de entrada. La segunda fila corresponde a los valores de cada neurona para el segundo registro.

Conocer las dimensiones de la salida de nuestro modelo es esencial para poder incorporar más niveles a nuestro modelo.

## Red neuronal: modelo multicapa

### Añadimos nueva capa

Las redes neuronales con una sola capa tienen limitada su capacidad de predicción. El objetivo es convertir nuestro perceptron en una red neuronal con multiples capas. Esto es lo que conocemos como Deep Learning.

Desde el punto de vista de arquitectura para añadir nueva capa simplemente tomamos la salida de la primera capa como entrada de la segunda. Se trata de encadenar varias capas para generar un modelo más complejo que permita más capacidad de análisis de patrones y relaciones.


Cada capa es básicamente una matriz de pesos más el temino bias que se suma posteriormente. La dimensión de la matriz para nuestra segunda capa depende del número de neuronas de la capa anterior (ya definida) y el número de neuronas que queremos en la segunda capa. Sabemos que para multiplicar matrices las columnas de la primera capa tienen que ser iguales a las filas de la segunda. Para añadir una nueva capa tenemos que definir la matrix de pesos que tendrá como número de filas el número de neuronas de la capa anterior y el número de columnas igual al número de neuronas de la nueva capa.

Desde un punto de vista más matemático añadir una nueva capa representa una composición de funciones. Una composiciñon de funciones significa que la salida de una función es la entrada de la siguiente lo que encaja perfectamente con lo descripto en la parte de arquitectura de nuestro modelo.

Supongamos que tenemos dos funciones $$f(x) = W_1*x + b_1$$ y una función $$g(x)=W_2*x + b_2$$. El resultado final de la composición de estas dos funciones sería:
$$m(x)=g(f(x))$$. Esto quiere decir que la variable $$x$$ de la función $$g(x)$$ se sustituye por la propia función $$f(x)$$.

$$m(x)=g(f(x))=g(W_1*x + b_1)= W_2*(W_1*x + b_1) + b_2$$

Si $$m(x)$$ es nuestro modelo, los datos de entrada serían la $$x$$. Vemos que en la función desarrollada aparecen los pesos de nuestras dos capas y sus correspondientes terminos bias. Nuestro modelo tendrá que ajustar los pesos de las matrices $$W_1$$ y $$W_2$$ y los términos $$b_1$$ y $$b_2$$ para que el error cometido por nuestro modelo con respecto al valor real de losnuestros datos de entrenamiento sea el menor posible.

### Funciones de activación.

Lo primero que tenemos que decir es que el modelo descrito anteriomente no funciona. Una composición de dos funciones lineales es otra función linear. Eso significa que nuestro modelo con dos capas se comporta exactamente igual que un modelo de una sola capa. No aumentamos la capacidad de aprendizaje al añadir nuevas capas

Sin entrar en mucho detalle desarrollamos la función anterior.

$$m(x)=g(f(x))=g(W_1*x + b_1)= W_2*(W_1*x + b_1) + b_2 = W_2*W_1*x + W_2*b_1 + b_2 = W*x + b$$

> Si aplicamos las reglas del cálculo de matrices podrías comprobar $$W_2*W_1$$ sería una matriz de pesos similar la capa uno y que $$W_2*b_1$$ sería un vector de las mismas dimensiones que $$b_2$$ y al sumarlo se comportaría como un nuevo termino bias (sesgo).

Para evitar este problema se aplica una función no linear al final de cada capa. De esta forma se rompe la linearidad y podemos añadir más capas de forma efectiva. Cada nueva capa significa nuevos pesos y nuevos terminos bias (sesgos) que el modelo tiene que ajustar. De esta forma aumentamos la capacidad de aprendizaje del modelo al añadir nuevas capas.

Es importante tener en cuenta que al aplicar una función de activación no modificamos ni la dimensión de la matriz resultante ni añadimos más parámetros al modelo.

Las funciones de activación más comunes son:
- Logistic (sigmoid): $${\displaystyle S(x)={\frac {1}{1+e^{-x}}}={\frac {e^{x}}{e^{x}+1}}}$$

![logistic](/blogs/img/sigmoid.png)

- tangente hiperbolica: $${\displaystyle \tanh x={\cfrac {e^{x}-e^{-x}}{e^{x}+e^{-x}}}}$$

![tanh](/blogs/img/tanh.png)

- Relu: $$\displaystyle f(x) =\max(0,x)$$

![Relu](/blogs/img/relu.png)
  

Inicialmente la función logistic se utilizaba con mucha frecuencia como función de activación. Actualmente Relu es la función más utilizada principalmente con configuraciones de redes neuronales sencillas. En redes neuronales más complejas como redes convolucionales o recurrentes también se utiliza logistic y tanh.

### Modelo multicapa

 Ahora que entendemos como podemos añadir nuevas capas a nuestro modelo vamos a definir más en detalle una arquitectura típica. Tenemos tres tipos de capas.
 - Capa de entrada (input layer): no es una capa en si, sino la entrada al modelo. Esta capa representa la información de entrenamiento. Las columnas representan cada una de las características de nuestra información y las filas el número de ejemplos que vamos a pasar al modelo.
 - Capa de salida (output layer): capa final que devuelve el resultado del modelo. El número de neuronas depende del tipo de problema que estamos abordando. Si el modelo predice un valor único tendremos una sola neurona con una salida para cada registro de entrada. Si estamos realizando una clasificación tendremos tantas salidas como categorías de clasificación, lo que supondría una matriz de resultados.
 - Capa oculta (hidden layer): capas intermedias con la configuración que hemos descrito en el apartado anterior. Se componen de una función lineal seguido de una función de activación. La definición de la matriz de estas capas depende del número de neuronas de la capa anterior y del número de neuronas que queremos en la actual.
  
### Ejemplo Práctico

Vamos a ver un ejemplo sencillo. 
>Somos una inmoviliaría que queremos informar a nuestros clientes del precio estimado de venta de su vivienda de forma inmediata. De operaciones anteriores tenemos información de 2.000 viviendas con sus características y sus precios reales de venta. De cada vivienda tenemos 45 caracterísitcas que nos definen el inmueble (planta, metros cuadrados, número de habitaciones, localidad, barrio etc).


Tomamos la decisión de crear un modelo de red neuronal con cuatro capas. La primera capa representa la entrada y decidimos pasar la información de entrenamiento a nuestro modelo en paquetes de 32 viviendas. añadimos dos capas ocultas donde la primera capa tendrá 200 neuronas y la segunda 100. Como queremos predecir un valor único que será el precio estimado de venta la capa de salida tendrá una única neurona.

El esquema de nuestra red neuronal sería de esta manera.

<img src="/blogs/img/ejemplo_rn.jpg" alt="red neuronal"
	title="perceptron" width="100%" />


Vamos a analizar los parámetros y las dimensiones de las matrices en cada capa para conocer mejor el funcionamiento interno.

* Capa de entrada (input layer): es la más sencilla de analizar. Si cada vivienda tiene 45 características y al modelo le pasamos 32 viviendas la entrada al modelo es una matriz input(32,45). Tenemos 32 filas y 45 columnas.
* Primera capa oculta (hidden layer 1): Esta es una capa propiamente dicha de nuestro modelo por lo que tenemos una matriz de pesos y un vector de terminos bias. La matriz de entrada se multiplica por la matriz de pesos y se le suma el término bias. Tanto la matriz de pesos como el vector bias de suma son parámetros de esta capa. Tenemos 9200 parámetros
  
  - Matriz de pesos: $$W_1(45,200) = 9000$$ parámetros
  - Bias: $$b_1(200) = 200$$ parámetros

* Segunda capa oculta (hidden layer 2): Toma como entrada la salida de la capa anterior. El número de filas corresponde al número de neuronas de la capa anterior. Las columnas corresponden al número de neuronas escogido para esta capa. Tenemos 20100 parámetros
  
  - Matriz de pesos: $$W_1(200,100) = 20000$$ parámetros
  - Bias: $$b_1(100) = 100$$ parámetros

* Capa de salida (output layer): Toma como entrada la salida de la capa anterior. El número de filas corresponde al número de neuronas de la capa anterior. Las columnas corresponden al número de nueronas decidio para esta capa, en este caso una única columna. Tenemos 101 parámetros
  
  - Matriz de pesos: $$W_1(100,1) = 100$$ parámetros
  - Bias: $$b_1(1) = 1$$ parámetros

En total el modelo tiene x parámetros que tiene que ajustar en la fase de entrenamiento. Una vez entrenado nuestro modelo podemos utilizarlo para predecir el coste estimado de venta de nuestra vivienda. Un modelo ya entrenado y listo para su uso se compone de dos partes.
- Arquitectura: descripción de las diferentes capas de nuestro modelo.
- Parámetros: los valores que se han ajustado en la fase de entrenamiento.

# Conclusión

Un modelo de red neuronal se compone de varias capas que constituyen la arquitectura del modelo. Cada capa consiste en una función lineal más una función de activación. 

La función lineal es del tipo $$f(x) = W*x +b$$ y consiste en la matriz de pesos $$W$$ y el término de sesgo o bias $$b$$. La matriz de pesos y el término bias representan los parámetros de la capa que el modelo tiene que ajustar en la fase de entrenamiento.

La función de activación cumple la misión de evitar la linearidad al componer varias capas. No afecta ni a las dimensiones de la salida de la capa ni tampoco genera nuevos parámetros. Simplemente es una función que se aplica a la salida de la función lineal. Relu es una de las funciones de activación más utilizada. Simplemente devuelve un cero si el valor es negativo y el propio valor si es positivo.

El tipo de modelo que hemos explicado es el más sencillo y se denomina fully-connected neural network. Existen otros tipos de capas con configraciones más complejas pero todas comparten un patrón de capas encadenadas.

En el siguiente árticulo hablaremos del proceso de entrenamiento  y como conseguir modelos que nos pemitan predecir valores o inferir características a partir de datos nuevos (distintos a los utiilizados para entranar le modelo).