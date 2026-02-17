Cambios a revisar y correguir:

- El boton de play no esta llevando a la seleccion de niveles en el Home, se debe reubicar el boton del modo P1vsP2.

- se debe implementar una ventana nueva para configurar el modo Player vs Player con los siguientes apartados: seleccionar el background del nivel de los 5 niveles actuales, seleccionar la skin de ambos jugadores, maximo numero de goles para ganar (1,2,3 mas) y tiempo de duracion del partido (5, 10, 15, o sin limite de tiempo).

- actualmente todos los niveles usan la misma porteria del nivel 1, se debe correguir eso usando la porteria de su respectivo nivel.

- durante cualquier partida se debe mostrar el tiempo de duracion de la partida (debe aparecer en un lugar que no tape la vizualizacion de la misma fuera de la cancha), actualmente ese cronometro aparece al finzalizar una partida y no se detiene (el cronometro sigue avanzando).

- la seccion de seleccion de skins, perfiles y similares la UI se bloquea al darle click a cualquier elemento (su rendimiento esta muy lento).

- se debe comenzar a usar un componente boton que admita (texto, animacion, colores, bordes, o background de imagen si es necesario) y usar ese boton optimo en toda la UI. 

- quiero revisar y actualizar la rubrica de rewards para agregar el siguiente reward: darle puntuacion al agente cuando este golpea el mullet en direccion de la porteria del enemigo. quiero que en la rubrica valore el comportamiento defensivo del agente evitando que este le anoten goles (crea un reward o penalizacion que haga esto).

- valida el comportamiento del enemigo de entrenamiento del agente para que este juege las partidas mejor (revisa si es mejor programarle un enemigo algoritmico o un modelo de redes adversarias), (y tenga una progesion de dificultad mejor).

- Revisa la arquitectura completa del agente para identificar factores que se puedan optimizar (numero de capas, numero de neuronas por capas, inputs, outputs, metodo de entrenamiento de parametros). consulta en internet mejoras posibles para el entrenamiento de los hiperparametros y el aprendisaje deep Q-learning.

- Se debe componetizar mejor la UI para evitar re-renderizados innecesarios como lo seria el background, titulos y demas elementos que no cambian nada, para optimizar el rendimiento (se debe usar siempre primero lo dado por Pygame (consultar esa informacion en internet))
