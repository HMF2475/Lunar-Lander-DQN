# Resumen de lo que hay que hacer:

- Python y cualquier libreria, pero requeridas: 
	1. [Keras/tensorflow](https://keras.io) -> para generar modelos de manera sencilla
	2. [Pytorch](https://pytorch.org) -> permite controlar los pasos internos de manera más cercana que Keras (**Yo usaría esta**)
	3. [Gymnasium](https://gymnasium.farama.org/index.html) -> modelo de referencia para aprendisaje por refuerzo
- Politica de decisiones con una red neuronal Feed Forward Simple
- Modelo lo resuelve en al menos 30% de los casos (mientras más porcentaje, mejor)
- Documento en **Latex** y una presentación powerpoint:
	- Minimo 6 páginas
	- Seguir estructura general (incluso recomiendan la herramienta overleaf)
	- En PDF
	- Memoria con: Introducción, descripción algoritmos implementados (explicando dificultades y decisiones de diseño adoptadas para resolverlas), experimentación, descripcion de resultados alcanzados (con graficas de progreso de los modelos entrenados, conclusiones y bibliografia, pero **JAMÁS CODIGO EN LA MEMORIA**)
	- 1 zip con la memoria, el modelo guardado en .keras o **.h5** (.h5 si vamos a usar pytorch), modulo para la resolucion del problema (**DQN.py**) que se ejecuta con **LunarRL.ipynb** y **lunar.py** (ESTOS NO SE PUEDEN MODIFICAR)
	- Nos citan para la defensa, el powerpoint debe durar 10 minutos en el que ambos debemos participar activamente. Tiene que seguir a grandes rasgos la estructura de la memoria dando más importancia a los resultados obtenidos y el analisis de estos. En los siguientes 10 minutos nos hacen preguntas sobre la memoria y/o el codigo fuente

### Problema:

- Se genera un terreno aleatorio
- Se ponen 2 banderas de aterrizaje en la misma posicion
- Se lanza el modulo con velocidades aleatorias dentro de sus rangos (ver tabla en PDF)
- 8 datos para saber el estado y 4 acciones discretas posibles
- Recompensas ver PDF
- Pasos: ver estado -> aplicar politica (red neuronal) -> calcular accion y dar paso -> almacenar resultados del episiodio


### Descripcion del trabajo:
- **RED NEURONAL FEED FORWARD**
- Poder guardar y cargar modelos entrenados
- Recibir valores de variable para modificar los hiperparámetros de cada algoritmo -> factor de descuento, Tamaño replay buffer, frecuencia de actualizacion red objetivo, tasa de aprendisaje, tamaño minibatch en cada actualizacion (ver en PDF)
- **DQN** -> cada combinación estado-acción tiene un valor Q. En vez de tabla Q tenemos una red neuronal que habilita el aprendisaje por refuerzo y entrena estados similares
- **3 componentes**: 
	1. red neuronal para estimar Q-valor (Q-network) -> Q(s,a;0)
	2. Buffer de experiencias (recompensas, acciones, estados previos y actuales) -> (s,a,r,s')
	3. Red neuronal estatica que copia la Q-network -> Q(s',a';0-)
- La función de perdida depende de la diferencia entre los Q-valores de la Q-network y la red objetivo, observando el progreso con estos deltas que la guian hacia un pico de gradiente


### Evaluación:
- **Memoria** (1 punto): claridad de explicaciones, razonamiento de decisiones, analisis y presentacion de resultados, correcto uso del lenguaje. (No copiarse de nadie xd)
- **Busqueda solucion** (2 puntos): implementacion de la red segun el algoritmo y las dificultades presentadas. Sumaría la mejora de los algoritmos pero no es necesario
- **Código fuente** (1 punto): claridad y buen estilo de programacion, correccion y eficiencia de la implementación, calidad de los comentarios (No copiarse o descargar de internet XD XD)
- **Presentacion y defensa**: claridad y buena explicacion de los contenidos (sobretodo las respuestas a las preguntas del profe) lo que multiplica entre [0,1] tu nota total
- Citar fuentes si o si o te ponen un cero NO PLAGIOS 
- Se puede usar IA explicando los promps, para que se usaron y demostrando en la defensa que se conoce y entiente la totalidad del trabajo, incluyendo las respuestas de ChatGPT



