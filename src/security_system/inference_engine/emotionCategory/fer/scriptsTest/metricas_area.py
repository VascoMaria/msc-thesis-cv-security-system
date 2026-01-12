import pandas as pd

EXCEL_PATH = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\HefestoPhotos\Areas\faces_ratios.xlsx"   # caminho do teu Excel
RAW_SHEET = "Resultados"
OUT_SHEET_METRICS = "Metrics"
OUT_SHEET_DEFS = "Definitions"


# Carregar dados
df = pd.read_excel(EXCEL_PATH, sheet_name=RAW_SHEET)
df["Label"] = df["Label"].str.lower().str.strip()

# Totais
total_samples = len(df)
positives = (df["Label"] == "positivos").sum()
negatives = (df["Label"] == "negativos").sum()

# Métricas básicas
grouped = (
    df.groupby("Label")["Ratio(%)"]
      .agg(["mean", "std", "median", "min", "max", "count"])
)

# Calcular métricas ignorando zeros
mean_no_zeros = {}
median_no_zeros = {}
for lbl in ["positivos", "negativos"]:
    vals = df.loc[df["Label"] == lbl, "Ratio(%)"]
    vals_no0 = vals[vals > 0]
    if len(vals_no0) > 0:
        mean_no_zeros[lbl] = vals_no0.mean()
        median_no_zeros[lbl] = vals_no0.median()
    else:
        mean_no_zeros[lbl] = 0.0
        median_no_zeros[lbl] = 0.0

grouped["mean_no_zeros"] = grouped.index.map(mean_no_zeros.get)
grouped["median_no_zeros"] = grouped.index.map(median_no_zeros.get)

# Garantir que ambas classes existem
for lbl in ["positivos", "negativos"]:
    if lbl not in grouped.index:
        grouped.loc[lbl] = [0]*len(grouped.columns)

# Gap entre médias
mean_gap = grouped.loc["positivos","mean"] - grouped.loc["negativos","mean"]

# Construir tabela final
metrics_table = grouped.reset_index().rename(columns={"Label":"Classe"})
metrics_table.loc[len(metrics_table)] = ["TOTAL", None, None, None, None, None, total_samples, None, None]
metrics_table.loc[len(metrics_table)] = ["mean_gap", mean_gap, None, None, None, None, None, None, None]

# Definições
definitions = pd.DataFrame({
    "Métrica": [
        "mean","std","median","mean_no_zeros","median_no_zeros",
        "min","max","count","mean_gap","TOTAL"
    ],
    "Definição": [
        "Média dos valores do Ratio(%).",
        "Desvio padrão (mede a variação em relação à média).",
        "Mediana: valor central quando os dados são ordenados (inclui zeros).",
        "Média ignorando valores 0 (quando a segunda face não foi detetada).",
        "Mediana ignorando valores 0 (quando a segunda face não foi detetada).",
        "Valor mínimo observado.",
        "Valor máximo observado.",
        "Número de amostras nessa classe.",
        "Diferença entre a média dos positivos e a média dos negativos.",
        "Número total de frames (positivos + negativos)."
    ]
})

# Guardar
with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    metrics_table.to_excel(writer, sheet_name=OUT_SHEET_METRICS, index=False)
    definitions.to_excel(writer, sheet_name=OUT_SHEET_DEFS, index=False)

print("✅ Guardado: 'Metrics' (linhas=classes) e 'Definitions' (explicações')")
