# Entregable Final - Semántica Para Ciencia del Dato
### Máster Universitario en Lógica Computacional e Inteligencia Artificial
#### Universidad de Sevilla
#### John Urena

Este repositorio contiene el entregable final para la asignatura "Semántica Para Ciencia del Dato" del Máster Universitario en Lógica Computacional e Inteligencia Artificial de la Universidad de Sevilla. Los resultados detallados se encuentran en el documento "Memoria de Análisis al Dataset POLE con Neo4j y Scikit-learn".

### Instrucciones para Replicar los Resultados

1. **Instalación de Neo4J**:
   - Descargar e instalar Neo4J localmente desde [aquí](https://neo4j.com/download/neo4j-desktop/?edition=desktop&flavour=unix&release=1.5.9&offline=true&utm_medium=PaidSearch&utm_source=google&utm_campaign=GDB&utm_content=EMEA-X-Conversion-GDB-Text&utm_term=download%20neo4j&gclid=CjwKCAiA44OtBhAOEiwAj4gpOZuCYOG4BW1OJ-csGU3n9ijq9BZ_wMzLCe1qFFfYxBXPU642Ff9AOxoCUesQAvD_BwE).

2. **Carga del Repositorio del Dataset POLE**:
   - El repositorio del dataset se encuentra en [este enlace](https://github.com/neo4j-graph-examples/pole).

3. **Configuración de la Base de Datos**:
   - Configurar los parámetros de la base de datos en el archivo `config.ini`:
     ```
     DB_HOST = <your_db_uri>
     DB_USER = <your_user>
     DB_PASSWORD = <your_password>
     ```

### Explicación del Código

- **Paquete Utils**:
  - En el paquete `Utils` se encuentran la mayoría de las funciones utilizadas a lo largo del cuaderno, y estas están documentadas.

- **Generación de CSV**:
  - El proceso para generar los archivos CSV es manual. La función `find_best_params` imprime en la consola las métricas de los árboles que generen un desempeño mejor al 85%. Para generar el CSV, copiar el resultado impreso en la consola y pegarlo en alguna de las variables de `text_parser` (hay dos ejemplos de cómo se hace en el código utilizado para la creación de este proyecto).

### Contacto

Para cualquier duda o consulta, puede contactar a través del siguiente correo electrónico: ing.jdum@gmail.com
