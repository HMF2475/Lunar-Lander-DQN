- La principal ventaja de Deep Q-Learning (La Q-Network) sobre la Q-Table es que no tenemos que contar TODOS los estados posibles y cada accion para cada uno, sino que tenemos la Red Neuronal que tiene de entrada los valores que definen los estados y de salida las acciones posibles.
- Tenemos 2 Redes neuronales, la target y la que entrenamos (principal)
    - 1º creamos una politica en la principal (random)
    - 2º la copiamos en la target (los pesos y bias)
    - 3º el agente navega el mapa 
    - 4º se carga la red principal con el estado (los primeros resultados seran random y no tienen mucho sentido)
    - 5º cargamos la target network (nos dará lo mismo que la anterior en este caso)
    - 6º calculamos el q-value para ese estado con una acción y lo seteamos en la red target (con el valor que deberia tener)
    - 7º usamos esos valores de la target network para actualizar los pesos de la principal con backpropagation
    - 8º repetimos del 3 al 7
    - 9º después de varias epocas actualizamos la target con la principal

- Los tensores de pytorch son como los arrays de numpy con la diferencia de que podemos hacer que corran en GPU o otros aceleradores hardware (hacen que optimizemos nuestro algoritmo para usar todo el poder hardware disponible)
- Tenemos que añadir el unsquezee(0) para que al array le añada un valor al principio que significa el tamaño del batch, ya que al entrenar una red siempre se espera el vector con los estados (en nuestro caso 8 números por los valores del observation space) y uno más que dice el tamaño del lote, es decir nos queda \[batch_size, state_size] en vez de solo de un elemento que la red no lo entiende

- La equacion de bellman para DQL es Qtarget(s,a) + r+ y* maxQnetwork(s',a) \[El maximo de la Qnetwork es del siguiente estado, es decir, la Q predecida]

- Un gran problema es como optimizar los hiperparametros sin tener que cambiarlos a mano uno a uno hasta obtener lo óptimo

- es deseable que la media de los Q-values actuales (Q-current) sea parecida a la de los Q-values objetivo (Q-target) durante el entrenamiento de un DQN. 
    - Objetivo del aprendizaje: La red intenta ajustar sus predicciones (Q-current) para que se acerquen a los valores objetivo (Q-target), que vienen de la ecuación de Bellman con la recompensa recibida y el valor descontado del siguiente estado. 
    - Cuando el entrenamiento progresa bien, la red minimiza la diferencia entre estos dos, por lo que los promedios deben converger y mantenerse cercanos.
    - Si están muy alejados, significa que el modelo aún no está aprendiendo o que hay un problema en la actualización.

- Se eligió Adam porque combina las ventajas de dos métodos clásicos (Momentum y RMSProp), lo que le permite:
    - Adaptar automáticamente la tasa de aprendizaje para cada parámetro.
    - Ser robusto frente a gradientes ruidosos y dispersos, como los que aparecen en el aprendizaje por refuerzo.
    - Lograr una convergencia más rápida y estable, especialmente útil cuando se entrena con redes profundas y experiencias variadas.
    - En comparación con optimizadores como SGD, Adam requiere menos ajuste fino de hiperparámetros y funciona mejor en entornos complejos como LunarLander.

- La "E" quiere decir que es el promedio del error cuadratico medio