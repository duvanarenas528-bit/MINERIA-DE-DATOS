                              Análisis y predicción de accidentes de tránsito mediante minería de datos
## 1. Integrantes y roles

### Leni Santiago Murillo Sterling
* Desarrollo Frontend e interfaz de usuario.
* Análisis de datos y Machine Learning.

Responsabilidades principales:

* Diseñar y desarrollar la interfaz web del proyecto.
* Construir el dashboard para la visualización de resultados.
* Implementar gráficos, indicadores y componentes interactivos.
* Participar en la limpieza y transformación de los datos.
* Realizar análisis exploratorio de los datos.
* Participar en la selección, entrenamiento y evaluación de modelos de Machine Learning.
* Interpretar los resultados obtenidos por los modelos.
* Integrar los resultados del análisis y las predicciones dentro de la interfaz.

### Duban Camilo Tibaquira Arenas
* Desarrollo Backend y procesamiento de datos.
* Análisis de datos y Machine Learning.

Responsabilidades principales:

* Diseñar y desarrollar la arquitectura Backend.
* Implementar los servicios necesarios para procesar y consultar los datos.
* Desarrollar los procesos de limpieza, transformación y preparación del dataset.
* Gestionar la comunicación entre el Backend, el modelo predictivo y el Frontend.
* Participar en el análisis exploratorio de los datos.
* Participar en la selección, entrenamiento y evaluación de modelos de Machine Learning.
* Implementar los procesos necesarios para ejecutar las predicciones.
* Optimizar el procesamiento de grandes volúmenes de información.


## 2. Idea del proyecto

Desarrollar una solución de análisis y predicción de accidentes de tránsito mediante técnicas de minería de datos y Machine Learning, utilizando un dataset amplio de registros históricos de accidentes.

El proyecto busca identificar patrones relacionados con la ocurrencia y gravedad de los accidentes, analizando variables como fecha, hora, ubicación, condiciones climáticas, tipo de vía, vehículos involucrados y características del accidente.

A partir de estos datos se desarrollará un sistema capaz de generar información visual mediante un dashboard y aplicar modelos de Machine Learning que permitan estimar la gravedad de un accidente y detectar factores asociados a un mayor nivel de riesgo.

La solución permitirá transformar grandes volúmenes de datos históricos en información útil para comprender el comportamiento de los accidentes de tránsito y apoyar la toma de decisiones relacionadas con la seguridad vial.


## 3. Problema

Los accidentes de tránsito representan un problema que puede estar relacionado con múltiples factores, como las condiciones de la vía, el clima, la hora, el día, la ubicación y las características de los vehículos involucrados. La gran cantidad de información generada por estos eventos puede dificultar la identificación manual de patrones y relaciones entre las diferentes variables.

Actualmente, los datos históricos de accidentes pueden contener información valiosa que, mediante técnicas de minería de datos, puede ser analizada para encontrar patrones y factores asociados con accidentes de mayor gravedad.

Por esta razón, se plantea desarrollar una solución que permita procesar y analizar un conjunto de datos amplio sobre accidentes de tránsito, identificar patrones relevantes y utilizar modelos de Machine Learning para realizar predicciones sobre la gravedad de los accidentes.

### Pregunta problema

¿Es posible utilizar técnicas de minería de datos y Machine Learning para identificar patrones en los accidentes de tránsito y predecir su nivel de gravedad a partir de sus características?


## 4. MVP (Producto Mínimo Viable)

El Producto Mínimo Viable consistirá en una aplicación web que permita visualizar y analizar información histórica sobre accidentes de tránsito y realizar predicciones sobre el nivel de gravedad de un accidente.

El MVP contará inicialmente con las siguientes funcionalidades:

1. **Carga y procesamiento de datos:** integración del dataset de accidentes y preparación de la información para su análisis.

2. **Dashboard de análisis:** visualización de indicadores y gráficos sobre accidentes por fecha, hora, ubicación, condiciones climáticas, tipo de vehículo y nivel de gravedad.

3. **Análisis exploratorio:** identificación de tendencias, relaciones y patrones presentes en los datos.

4. **Modelo predictivo:** implementación de un modelo de Machine Learning capaz de clasificar el nivel de gravedad de un accidente a partir de sus características.

5. **Módulo de predicción:** interfaz donde el usuario pueda ingresar determinadas características de un accidente y obtener una estimación de su nivel de gravedad.

El MVP permitirá demostrar que los datos históricos pueden utilizarse para generar conocimiento y realizar predicciones relacionadas con la seguridad vial.


## 5. Tecnologías previstas

Para el desarrollo del proyecto se contempla utilizar las siguientes tecnologías:

| Tecnología                  | Uso previsto                                                |
| --------------------------- | ----------------------------------------------------------- |
| **Python**                  | Procesamiento, análisis y modelado de datos                 |
| **Pandas**                  | Limpieza, transformación y análisis del dataset             |
| **NumPy**                   | Operaciones y procesamiento numérico                        |
| **Jupyter Notebook**        | Experimentación y documentación del análisis                |
| **SQL**                     | Consulta y gestión de datos, si resulta necesario           |
| **FastAPI o Flask**         | Desarrollo del Backend y exposición del modelo mediante API |
| **React**                   | Desarrollo de la interfaz y dashboard                       |
| **HTML / CSS / JavaScript** | Construcción y diseño de la interfaz web                    |
| **Git y GitHub**            | Control de versiones y trabajo colaborativo                 |

La selección definitiva de tecnologías podrá ajustarse durante el desarrollo de acuerdo con las necesidades del proyecto y las características finales del dataset.

## 6. Estado inicial

Actualmente, el proyecto se encuentra en una etapa inicial de planificación y definición de alcance.

Se ha seleccionado como problemática el análisis de accidentes de tránsito debido a la disponibilidad potencial de grandes volúmenes de datos históricos y a la posibilidad de aplicar diferentes técnicas de minería de datos para identificar patrones y realizar predicciones.

En esta etapa se han definido:

* La idea general del proyecto.
* El problema que se busca abordar.
* La pregunta problema.
* El alcance inicial del MVP.
* Los integrantes y sus responsabilidades.
* Las tecnologías que se consideran utilizar.




