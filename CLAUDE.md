# ml-case-studies · instrucciones para Claude

Repo **público** de casos de estudio de machine learning con datos públicos (Kaggle y similares).

- Todo en español de España: código, comentarios, commits y documentación.
- **Solo datos públicos.** Ni datos, ni código, ni nombres de tablas, ni cifras de ninguna
  empresa. Antes de cada commit, comprobar que no se cuela nada de eso.
- Una carpeta por caso, con su `README.md` (pregunta, datos, sesiones, conclusiones) y `src/`.
- Los datos van en `datos/<caso>/` (ignorada por git). Nunca subir CSV ni `kaggle.json`: las
  reglas de Kaggle no permiten redistribuir los datos.
- Los scripts imprimen resúmenes (métricas, tablas pequeñas); no volcar dataframes enteros.
- Los notebooks se suben sin salidas.
- El objetivo es enseñar el método, no subir en la clasificación: validación honesta y
  explicaciones claras antes que rascar décimas de AUC.
- Modelo por defecto: Sonnet 5.5. Haiku 5.5 en subagentes.
