# limpieza.py
#
# Script de diagnostico y limpieza que aplica TODAS las tecnicas de la
# Sesion 3 (Limpieza de Datos) sobre las 3 fuentes del proyecto:
#   - isna(), duplicated(), describe()            -> diagnostico()
#   - drop_duplicates()                           -> revisar_duplicados()
#   - Normalizacion de categorias (strip/title)   -> normalizar_categoria()
#   - Clasificacion MCAR / MAR / MNAR              -> clasificar_mecanismo_faltante()
#   - Outliers con IQR                             -> outliers_iqr()
#   - Outliers con z-score (para comparar con IQR)-> outliers_zscore()
#   - Correccion de valores imposibles             -> corregir_valor_imposible()
#   - Pipeline reproducible con validacion final   -> main()
#
# Los 3 datasets limpios se encuentran dentro de la carpeta:
#   limpios/
#
#   - fars_person_2023_limpio.csv
#   - clima_accidentes_limpio.csv
#   - accidentes_limpio.csv


import pandas as pd


# ------------------------------------------------------------------
# PASO 0 - Diagnostico: isna() + duplicated() + describe()
# ------------------------------------------------------------------

def diagnostico(path, columnas_numericas=None):
    """
    Diagnostico inicial de calidad:
    forma, nulos, duplicados y describe().
    """

    df = pd.read_csv(path)

    print(f"Archivo: {path}")
    print(f"Filas: {df.shape[0]}  Columnas: {df.shape[1]}")

    print("Cuanto falta (nulos por columna, top 5):")
    print(df.isna().sum().sort_values(ascending=False).head())

    print(
        f"Que esta repetido -> "
        f"Duplicados exactos: {df.duplicated().sum()}"
    )

    if columnas_numericas:
        print("Que se sale de rango -> describe():")
        print(
            df[columnas_numericas]
            .describe()
            .round(1)
        )

    print("-" * 60)

    return df


# ------------------------------------------------------------------
# PASO 1 - Duplicados: drop_duplicates()
# ------------------------------------------------------------------

def revisar_duplicados(df, subset=None):
    """
    Aplica drop_duplicates() y reporta si hubo cambio.
    """

    antes = len(df)

    df_dedup = (
        df
        .drop_duplicates(subset=subset)
        .reset_index(drop=True)
    )

    despues = len(df_dedup)

    print(
        f"Duplicados "
        f"({'llave ' + str(subset) if subset else 'exactos'}): "
        f"antes={antes} "
        f"despues={despues} "
        f"eliminados={antes - despues}"
    )

    return df_dedup


# ------------------------------------------------------------------
# PASO 2 - Categorias: normalizacion de texto
# ------------------------------------------------------------------

def normalizar_categoria(serie):
    """
    Normaliza una columna categorica:
    quita espacios y unifica mayusculas/minusculas.
    """

    return (
        serie
        .astype(str)
        .str.strip()
        .str.title()
    )


def verificar_categoria_normalizada(df, columna):
    """
    Compara los valores unicos antes y despues de normalizar.
    """

    originales = set(
        df[columna]
        .dropna()
        .unique()
    )

    normalizados = set(
        normalizar_categoria(
            df[columna].dropna()
        ).unique()
    )

    estado = (
        "sin redundancia"
        if len(originales) == len(normalizados)
        else "HABIA redundancia"
    )

    print(
        f"{columna}: "
        f"{len(originales)} categorias originales -> "
        f"{len(normalizados)} tras normalizar "
        f"({estado})"
    )


# ------------------------------------------------------------------
# PASO 3 - Valores faltantes: MCAR / MAR / MNAR
# ------------------------------------------------------------------

def clasificar_mecanismo_faltante(
    df,
    columna_nula,
    columna_evidencia
):
    """
    Busca evidencia de MAR comparando una variable observada
    segun si la columna_nula es NaN o no.
    """

    es_nulo = df[columna_nula].isna()

    resumen = (
        df.groupby(es_nulo)[columna_evidencia]
        .mean()
    )

    print(
        f"Evidencia de mecanismo para {columna_nula} "
        f"(usando {columna_evidencia} como variable observada):"
    )

    print(resumen)

    # Verificamos que existan los dos grupos
    if len(resumen) < 2:
        print(
            "No hay suficientes grupos para determinar "
            "el mecanismo de datos faltantes."
        )
        print("-" * 60)
        return

    diferencia = abs(
        resumen.iloc[-1] - resumen.iloc[0]
    )

    veredicto = (
        f"MAR (la ausencia se explica por "
        f"{columna_evidencia})"
        if diferencia > 0.5
        else "sin evidencia clara, revisar MCAR"
    )

    print(
        f"Diferencia entre grupos: "
        f"{diferencia:.1f} -> {veredicto}"
    )

    print("-" * 60)


# ------------------------------------------------------------------
# PASO 4a - Outliers con IQR
# ------------------------------------------------------------------

def outliers_iqr(df, columna):
    """
    Devuelve los outliers de una columna numerica
    segun la regla IQR de Tukey.
    """

    q1 = df[columna].quantile(0.25)
    q3 = df[columna].quantile(0.75)

    iqr = q3 - q1

    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr

    mask = (
        (df[columna] < lim_inf)
        |
        (df[columna] > lim_sup)
    )

    print(
        f"[IQR]      {columna}: "
        f"Q1={q1:.1f} "
        f"Q3={q3:.1f} "
        f"IQR={iqr:.1f} "
        f"limites=({lim_inf:.1f}, {lim_sup:.1f}) "
        f"outliers={mask.sum()}"
    )

    return df.loc[mask, columna]


# ------------------------------------------------------------------
# PASO 4b - Outliers con z-score
# ------------------------------------------------------------------

def outliers_zscore(df, columna, umbral=3):
    """
    Devuelve los outliers segun z-score.
    Regla clasica: |z| > 3.
    """

    media = df[columna].mean()
    std = df[columna].std()

    if std == 0 or pd.isna(std):
        print(
            f"[Z-SCORE]  {columna}: "
            f"no se puede calcular porque la desviacion "
            f"estandar es 0 o NaN."
        )
        return df.iloc[0:0][columna]

    z = (
        df[columna] - media
    ) / std

    mask = z.abs() > umbral

    print(
        f"[Z-SCORE]  {columna}: "
        f"media={media:.1f} "
        f"std={std:.1f} "
        f"|z|>{umbral} "
        f"outliers={mask.sum()}"
    )

    return df.loc[mask, columna]


# ------------------------------------------------------------------
# PASO 4c - Correccion de valores imposibles
# ------------------------------------------------------------------

def corregir_valor_imposible(
    df,
    columna,
    es_imposible
):
    """
    Reemplaza por la mediana los valores imposibles.
    No elimina filas.
    """

    n_afectados = int(
        es_imposible.sum()
    )

    if n_afectados > 0:

        mediana = (
            df.loc[~es_imposible, columna]
            .median()
        )

        df.loc[
            es_imposible,
            columna
        ] = mediana

        print(
            f"{columna}: "
            f"{n_afectados} valores imposibles corregidos "
            f"con la mediana ({mediana:.1f})"
        )

    else:

        print(
            f"{columna}: "
            f"0 valores imposibles encontrados "
            f"(ya corregido en la version *_limpio.csv)"
        )

    return n_afectados


# ------------------------------------------------------------------
# PIPELINE COMPLETO
# ------------------------------------------------------------------

def main():

    print("=" * 60)
    print("INICIO DEL PROCESO DE LIMPIEZA Y VALIDACION")
    print("=" * 60)

    # --------------------------------------------------------------
    # PASO 0: diagnostico de las 3 fuentes
    # --------------------------------------------------------------

    fars = diagnostico(
        "limpios/fars_person_2023_limpio.csv",
        columnas_numericas=["AGE"]
    )

    clima = diagnostico(
        "limpios/clima_accidentes_limpio.csv",
        columnas_numericas=[
            "Temperature(F)",
            "Pressure(in)",
            "Wind_Speed(mph)"
        ]
    )

    acc = diagnostico(
        "limpios/accidentes_limpio.csv",
        columnas_numericas=[
            "Distance(mi)"
        ]
    )


    # --------------------------------------------------------------
    # PASO 1: duplicados
    # --------------------------------------------------------------

    print("=" * 60)
    print("PASO 1 - REVISION DE DUPLICADOS")
    print("=" * 60)

    fars = revisar_duplicados(fars)
    clima = revisar_duplicados(clima)
    acc = revisar_duplicados(acc)


    # --------------------------------------------------------------
    # PASO 2: categorias normalizadas
    # --------------------------------------------------------------

    print("=" * 60)
    print("PASO 2 - NORMALIZACION DE CATEGORIAS")
    print("=" * 60)

    verificar_categoria_normalizada(
        clima,
        "Wind_Direction"
    )

    verificar_categoria_normalizada(
        acc,
        "State"
    )


    # --------------------------------------------------------------
    # PASO 3: mecanismo de valores faltantes
    # --------------------------------------------------------------

    print("=" * 60)
    print("PASO 3 - MCAR / MAR / MNAR")
    print("=" * 60)

    clasificar_mecanismo_faltante(
        clima,
        "Wind_Chill(F)",
        "Temperature(F)"
    )

    clasificar_mecanismo_faltante(
        clima,
        "Precipitation(in)",
        "Temperature(F)"
    )


    # --------------------------------------------------------------
    # PASO 4: OUTLIERS
    # --------------------------------------------------------------

    print("=" * 60)
    print("PASO 4 - OUTLIERS IQR Y Z-SCORE")
    print("=" * 60)

    # FARS
    # Se excluyen los codigos 998/999 porque representan
    # valores especiales y no edades reales.

    fars_edad_valida = (
        fars[
            fars["AGE"] < 998
        ]
        .copy()
    )

    outliers_iqr(
        fars_edad_valida,
        "AGE"
    )

    outliers_zscore(
        fars_edad_valida,
        "AGE"
    )


    # CLIMA

    outliers_iqr(
        clima,
        "Temperature(F)"
    )

    outliers_zscore(
        clima,
        "Temperature(F)"
    )


    # --------------------------------------------------------------
    # PASO 4c: validar valores imposibles
    # --------------------------------------------------------------

    print("=" * 60)
    print("PASO 4c - VALORES IMPOSIBLES")
    print("=" * 60)

    corregir_valor_imposible(
        fars_edad_valida,
        "AGE",
        fars_edad_valida["AGE"] > 110
    )


    # --------------------------------------------------------------
    # PASO 5: VALIDACION FINAL
    # --------------------------------------------------------------

    print("=" * 60)
    print("VALIDACION FINAL")
    print("=" * 60)

    print(
        f"FARS Person : "
        f"filas={len(fars)}  "
        f"duplicados={fars.duplicated().sum()}"
    )

    print(
        f"Clima       : "
        f"filas={len(clima)} "
        f"duplicados={clima.duplicated().sum()}"
    )

    print(
        f"Accidentes  : "
        f"filas={len(acc)}   "
        f"duplicados={acc.duplicated().sum()}"
    )

    print("=" * 60)
    print("PROCESO TERMINADO CORRECTAMENTE")
    print("=" * 60)


# ------------------------------------------------------------------
# EJECUCION
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()