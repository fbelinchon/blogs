---
layout: post
title: "hola Enfoque"
subtitle: "Optimizar si enfocas fggg vgvgvgfgf."
date: 2020-05-19 23:45:13 -0400

background: '/img/posts/01.jpg'
---

#¿Cómo sabemos en que dirección tenemos que modificar nuestros parámetros para mejorar la predicción?

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
