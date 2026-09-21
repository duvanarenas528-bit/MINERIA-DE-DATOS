# Transformacion y reduccion con los datos de accidentes + clima
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
# Agarra la carpeta donde esta guardado este archivo, asi no toca escribir la ruta completa
RUTA = Path(__file__).resolve().parent

#  ABRIR LOS DOS ARCHIVOS Y PEGARLOS EN UNA SOLA TABLA
# Son dos tablas distintas pero hablan de los mismos accidentes: una tiene el choque
# y la otra el clima de ese momento. Las dos traen la columna ID, entonces la usamos
# para emparejarlas (como cuando cruzas dos listas por el numero de cedula).
acc = pd.read_csv(RUTA / "LIMPIOS/accidentes_limpio.csv")
clima = pd.read_csv(RUTA / "LIMPIOS/clima_accidentes_limpio.csv")
# Le quito a la tabla de clima unas columnas que ya venian en la otra, para no repetirlas
df = acc.merge(clima.drop(columns=["Start_Time", "City", "State"]), on="ID")
print("Accidentes:", acc.shape, "| Clima:", clima.shape, "| Unidos:", df.shape)

# Estas son las columnas de numeros con las que vamos a trabajar
NUMERICAS = [
    "Start_Lat", "Start_Lng", "Distance(mi)",
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)",
]

# Wind_Chill y Precipitation no las meti: vienen vacias en mas del 90% de las filas,
# o sea que casi no hay dato ahi y no sirve de nada rellenarlas.
# A las que si quedaron les tapo los poquitos huecos con la mediana (el valor del medio).
X = df[NUMERICAS].fillna(df[NUMERICAS].median())
print("Matriz numerica:", X.shape, "| nulos restantes:", int(X.isna().sum().sum()))


# LA PARTE DONDE SE VE POR QUE HAY QUE ESCALAR
# Esta funcioncita mide que tan lejos esta un accidente de otro. Es el mismo teorema
# de Pitagoras del colegio pero con varias columnas en vez de dos lados.
def distancia(a, b):
    return ((a - b) ** 2).sum() ** 0.5


# Agarro los dos primeros accidentes y miro cuanto pone cada columna a esa diferencia
c1, c2 = X.iloc[0], X.iloc[1]
print("\n--- Distancia entre dos accidentes SIN escalar ---")
for col in NUMERICAS:
    print(f"  {col:<18} aporta {abs(c1[col] - c2[col]):>10.2f}")
print("Distancia total sin escalar :", round(distancia(c1, c2), 2))

# Aqui esta el chiste: casi toda la diferencia la pone la humedad, solo porque
# se mide de 0 a 100 y las demas manejan numeros chiquitos. No es que la humedad
# sea mas importante, es que sus numeros son mas grandes y se roban el show.

# MinMax deja todas las columnas entre 0 y 1, asi ninguna pesa mas que otra por el tamano
X_minmax = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=NUMERICAS)
print("Distancia total con MinMax  :", round(distancia(X_minmax.iloc[0], X_minmax.iloc[1]), 3))

# PROBANDO LAS TRES FORMAS DE ESCALAR QUE VIMOS EN CLASE
print("\n--- MinMaxScaler (todo a [0, 1]) ---")
print("Antes  : min", X.min().round(2).tolist())
print("Después: min", X_minmax.min().round(3).tolist(), "max", X_minmax.max().round(3).tolist())

# Standard deja todo centrado en 0. Los que estan por encima del promedio quedan
# en positivo y los de abajo en negativo.
print("\n--- StandardScaler (media=0, desviación=1) ---")
X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=NUMERICAS)
print("Medias tras escalar:", X_std.mean().round(3).tolist())
print("Desv. estándar     :", X_std.std().round(3).tolist())
# Si imprimo los maximos se ve el problema: hay valores rarisimos que quedan altisimos
print("Máximos (outliers) :", X_std.max().round(1).tolist())

# Robust hace algo parecido pero se apoya en el valor del medio, entonces los datos
# raros no lo despeinan tanto
print("\n--- RobustScaler (usa mediana e IQR, resiste outliers) ---")
X_rob = pd.DataFrame(RobustScaler().fit_transform(X), columns=NUMERICAS)
print("Mediana tras escalar:", X_rob.median().round(3).tolist())

# Antes de decidir cual usar, cuento cuantos valores raros tiene cada columna.
# La regla que nos dieron: lo que se salga bastante del monton, cuenta como raro.
print("\n--- % de outliers por variable (justifica el escalador elegido) ---")
q1, q3 = X.quantile(0.25), X.quantile(0.75)
iqr = q3 - q1
fuera = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).mean() * 100
print(fuera.round(2).to_string())

# Me di cuenta de algo feo probando: Robust divide entre el IQR, y si ese numero
# es cero o casi cero, la division se dispara y salen valores absurdos (me salio
# un 5113 en Distance). Pasa cuando casi todas las filas tienen el mismo valor.
print("\n--- IQR de cada variable (aviso para RobustScaler) ---")
print(iqr.round(3).to_string())
DEGENERADAS = iqr[iqr < 0.05].index.tolist()
print("Variables con IQR ~ 0 (no escalar, mejor discretizar):", DEGENERADAS)

# Entonces a esas dos mejor las vuelvo pregunta de si o no (1 o 0) y listo,
# asi no hay que escalarlas y no dañan nada
df["Con_distancia"] = (df["Distance(mi)"] > 0).astype(int)   # hubo via bloqueada?
df["Baja_visibilidad"] = (X["Visibility(mi)"] < 10).astype(int)  # se veia mal?

NUMERICAS_OK = [c for c in NUMERICAS if c not in ("Distance(mi)", "Visibility(mi)")]
X_rob = pd.DataFrame(
    RobustScaler().fit_transform(X[NUMERICAS_OK]), columns=NUMERICAS_OK
)
print("Robust aplicado solo a:", NUMERICAS_OK)
print("Máximos tras Robust:", X_rob.max().round(1).tolist())

# PASAR EL TEXTO A NUMEROS
# El computador no entiende "Clear" ni "Overcast", toca volverlo numeros.
# Con get_dummies cada categoria se vuelve su propia columna de 1 y 0, asi no queda
# pensando que un clima es "mas" que otro.
print("\n--- One-hot con get_dummies ---")
# Weather_Condition tiene un monton de categorias distintas, si las dejo todas
# me salen como 40 columnas nuevas. Entonces dejo las 5 mas repetidas y el resto
# lo junto en "Otro".
top5 = df["Weather_Condition"].value_counts().head(5).index
df["Clima_grupo"] = df["Weather_Condition"].where(df["Weather_Condition"].isin(top5), "Otro")
df["Sunrise_Sunset"] = df["Sunrise_Sunset"].fillna("Day")  # habia una sola fila vacia

categoricas = ["Clima_grupo", "Sunrise_Sunset"]
X_cat = pd.get_dummies(df[categoricas], prefix=categoricas, dtype=int)
print("Columnas nuevas:", X_cat.columns.tolist())
print(X_cat.head(3).to_string())

# Estas columnas ya venian como Verdadero/Falso, o sea que ya son de si o no.
# Solo toca cambiarlas a 1 y 0 y quedan listas.
BANDERAS = ["Crossing", "Junction", "Traffic_Signal", "Stop", "Station",
            "Con_distancia", "Baja_visibilidad"]
X_bool = df[BANDERAS].astype(int)
print("Banderas de vía (proporción de 1s):", X_bool.mean().round(3).tolist())

# PARTIR LA TEMPERATURA EN GRUPOS
# En vez de manejar el numero exacto, la parto en 4 grupos (tipo frio, fresco,
# tibio, caliente). Con "quantile" cada grupo queda con mas o menos la misma
# cantidad de filas, no con el mismo rango de grados.
print("\n--- KBinsDiscretizer: Temperature(F) -> 4 categorías ---")
discretizador = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
temp_bins = discretizador.fit_transform(X[["Temperature(F)"]]).ravel()
print("Bordes de los rangos:", discretizador.bin_edges_[0].round(1))
print(pd.Series(temp_bins).value_counts().sort_index().to_string())

# PCA, O SEA TRATAR DE RESUMIR LAS COLUMNAS
# La idea es ver si 8 columnas se pueden resumir en menos sin perder casi nada.
# Ojo: esto solo funciona si los datos ya estan escalados, por eso uso X_std.
print("\n--- PCA: ¿cuántas dimensiones resumen los datos? ---")
pca = PCA()
pca.fit(X_std)
razones = pca.explained_variance_ratio_      # cuanto aporta cada pedazo nuevo
acumulada = razones.cumsum().round(3)        # los voy sumando uno tras otro
print("Varianza explicada por componente :", razones.round(3))
print("Varianza acumulada                :", acumulada)
# Cuento cuantos pedazos necesito para llegar al 80% y al 90% de la informacion
para_80 = int((acumulada < 0.80).sum()) + 1
para_90 = int((acumulada < 0.90).sum()) + 1
print(f"Componentes para explicar >=80%: {para_80} | para >=90%: {para_90} (de {len(NUMERICAS)})")
# Conclusion: aqui casi no comprime nada, toca casi las mismas columnas. Segun lo
# que explicaron, eso pasa cuando las columnas no se parecen entre ellas.

# LA TABLA FINAL, YA LISTA
# Junto lo escalado + las columnas de clima en 1 y 0 + las de si o no.
X_final = pd.concat([X_rob, X_cat, X_bool], axis=1)
y = df["Severity"]  # esto es lo que se quiere predecir, va aparte y NUNCA dentro de X
print("\n--- Matriz final para modelado ---")
print("Forma X:", X_final.shape, "| y:", y.shape)
print("Columnas:", X_final.columns.tolist())
print("Distribución de y (Severity):")
print(y.value_counts(normalize=True).round(3).to_string())
# Aqui se ve que casi todo es nivel 2 y 3, de los otros hay poquisimos casos