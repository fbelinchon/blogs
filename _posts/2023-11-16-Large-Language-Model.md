---
layout: post
title: "1. ¿CHATGPT, Llama, Palm? Desmitificando el Poder de los Modelos de Lenguaje Grandes (LLM)"
subtitle: "Si tienes curiosidad por saber qué significa LLM (Large Languaje Model) y no tienes nada mejor que hacer en los próximos 10 minutos, sigue leyendo. Verás que nos es tan complicado"
date: 2023-11-16 10:45:13 -0400
background: '/img/posts/06.jpg'
categories: AI
---

# Article I.	Desmitificando el Poder de los Modelos de Lenguaje Grandes (LLM)

## Introducción a los Modelos de Lenguaje Grandes (LLM)

En el fascinante campo de la Inteligencia Artificial (IA), los Modelos de Lenguaje Grandes (LLM) han emergido como titanes lingüísticos, capaces de comprender y generar texto de manera sorprendentemente similar al lenguaje humano. Estos modelos, dotados con miles de millones de parámetros, han transformado la forma en que interactuamos con las máquinas y procesamos información. En esta exploración académica, desentrañaremos la evolución y las características distintivas de los LLM.

# Historia del Modelo de Lenguaje en IA

## El Surgimiento del Procesamiento del Lenguaje Natural (NLP)
Antes de adentrarnos en los LLM, es esencial contextualizar su llegada en la historia del Procesamiento del Lenguaje Natural (NLP). En sus etapas iniciales, los modelos de lenguaje eran rudimentarios, con la tarea principal de reconocer patrones gramaticales simples. Este periodo inicial reflejó la lucha por dotar a las máquinas de la capacidad de interpretar y generar texto de manera significativa.
Inicialmente se basaban en modelos estadísticos pero pronto se empezaron a utilizar modelos con redes neuronales sencillas.


## Empezamos a hablar de modelos de lengiaje.
Un punto de inflexión significativo en esta narrativa fue la introducción de la técnica de Ajuste Universal de Modelos de Lenguaje (ULMFIT). Propuesta por Howard y Ruder en 2018, ULMFIT permitió la adaptación de modelos de lenguaje preentrenados a tareas específicas, allanando el camino para una comprensión más profunda y aplicada del lenguaje.

En 2017 Jeremy Howard consiguió extrapolar el concepto de refinamiento de modelos genéricos, utilizados con éxito en tratamiento de imágenes, al campo del entendimiento del lenguage natural. El entrenamiento de un modelo genérico es muy costoso y necesita muchos recursos, pero una vez optimizado, se puede adaptar a diferentes tareas personalizadas refinando el modelo con muchos menos datos y recursos.

Se utilizaban modelos de redes neuronales recurrentes RNN para intentar capturar el contexto del texto. 

## Llegada de arquitectura Transformers.
Las redes neuronales recurrentes RNN procesan los datos de forma secuencial y que el contexto del texto según nos vamos incorporando nuevas palabras. La arquitectura de transformers soluciona estos dos problemas y es la base de los nuevos modelos de lenguaje.

rquitectura se basa en el concepto de atención que captura la relación de una palabras con las el resto de palabras del texto. El famoso artículo 
Attention is all you need https://arxiv.org/pdf/1706.03762.pdf, estableció las bases para el desarrollo de modelos basados en esta arquitectura.

La información se alimenta en bloques denominado contexto para aprovechar la paralelización de las operaciones que proporciona las GPU o TPU. De esta forma se acelera el procesamiento de la información evitando el problema de las redes neuronales recurrentes.
sta arquitectura se divide en una parte denominada Encoder y otra llamada Decoder.


# Diferencias Entre Modelos de Lengiaje y Modelos de Lenguaje Grandes.
## Escala y Complejidad.
La distinción principal entre los modelos de lenguaje convencionales y los LLM radica en su escala. Mientras que los primeros operan con una cantidad limitada de parámetros, los LLM despliegan un repertorio masivo que permite una comprensión más profunda y sensible al conetexto del lenguaje. Pasamos de modelos con cientos de millones de parámetros a los nuevos super modelos con varios miles de millones (lo que son billones americanos).


## Versatilidad y Adaptabilidad.
Otro rasgo distintivo es la versatilidad de los LLM. A través de técnicas como el ajuste fino, estos modelos pueden adaptarse a tareas específicas, desde traducción automática hasta generación de texto creativo. Esta flexibilidad es un testimonio de su capacidad para abordar una variedad de desafíos lingüísticos con éxito.

Los primeros modelos se entrenaban y se afinaban específicamente para realizar tareas concretas mientras que los modelos LLM pueden realizazar diferentes tareas según el enuncionado y el contexto que se les pasa. Es lo que se denomina *prompt engineering*, la habilidad para generar instrucciones precisan para conseguir respuestas precisas del modelo.

## Contextualización del Lenguaje
Quizás la diferencia más crucial reside en la capacidad de los LLM para entender el contexto. A diferencia de sus predecesores, estos modelos no solo procesan palabras individualmente, sino que capturan las relaciones contextuales, permitiendo una generación de texto más coherente y significativa.
El entendimiwento del contexto es mucho mayor del que tenían los modelos iniciales.

# Entrenamiento de Modelos de Lenguaje Grandes (LLM): Un Vistazo Didáctico
Los Modelos de Lenguaje Grandes (LLM) son criaturas fascinantes, pero su entrenamiento es un viaje igualmente intrigante. En términos sencillos, imagine el proceso como enseñarle a una máquina a entender y generar texto como lo haría una persona. Aquí hay un desglose académico y accesible de cómo se lleva a cabo este entrenamiento:

## Embeddings, como dar significado a las palabras.
Cuando enseñamos a las máquinas sobre el lenguaje, queremos que entiendan las palabras de una manera que puedan manejar. Aquí es donde entra el embedding. En términos sencillos, el embedding es como asignar a cada palabra del vocabulario un significativo mediante números.

Cuando entrenamos un modelo para predecir la siguiente palabra de un texto también le estamos pidiendo que nos genere un vector de números que defina el significado de cada palabra. Cuando el entrenamiento termina disponemos de una representación vectorial que aporta significado a todas las palabras del lenguage. Cada palabra o token se define por un conjunto de números que generan un vector.

Los vectores númericos calculados proporcionan información de tal forma que palabras con significados parecidos, o que pueden usarse en contextos similares, tiene vectores que están más próximos que palabras no relacionadas. Por ejemplo, "gato" y "perro" estarían más cerca que "gato" y "coche". Esto refleja la relación semántica entre las palabras y permite al modelo entender similitudes y diferencias.

El concepto de embedding es muy utilizado en inteligencia artifial para describir, en formato númerico, cualquir objeto o concepto como una palabra, una frase, una foto, un artículo, una película, las preferencias de un usuario etc.

## Pasos para entrenar un Modelo de Lenguaje.

### Paso 1: Preentrenamiento
El proceso comienza con lo que llamamos "preentrenamiento". En esta fase, el modelo se expone a vastas cantidades de texto sin etiquetas, como libros, artículos y páginas web. Aquí, el LLM aprende a predecir la siguiente palabra en una oración, desarrollando una comprensión básica del lenguaje y sus patrones. Es como enseñar al modelo a leer y entender lo que lee.

### Paso 2: Ajuste Fino (Fine-Tuning)
Después del preentrenamiento, entra en escena el "ajuste fino". En este paso, el modelo se especializa para tareas específicas, como traducción automática o análisis de sentimientos. Se le proporciona un conjunto de datos más pequeño y etiquetado para que afine sus habilidades lingüísticas según la tarea en cuestión. Piense en esto como guiar a un estudiante para que aplique sus habilidades de comprensión para aprender un tema concreto.



# Conclusión: Abrazando el Potencial de los LLM
En conclusión, los Modelos de Lenguaje Grandes representan una evolución fascinante en la inteligencia artificial. Desde sus raíces en el NLP hasta la aplicación pragmática de técnicas como ULMFIT, estos modelos han transformado nuestra comprensión y aprovechamiento del lenguaje computacional. A medida que continuamos explorando las posibilidades que ofrecen, los LLM se erigen como catalizadores clave en la convergencia exitosa entre la capacidad lingüística y la inteligencia artificial. Este viaje académico nos deja no solo con conocimiento sino con una apreciación renovada de la vastedad del potencial de los LLM en el vasto paisaje de la IA.
