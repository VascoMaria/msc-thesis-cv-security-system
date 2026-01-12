# rf_5x2cv.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CSV_PATH = "saida_so_confidencias.csv"  # o teu ficheiro
TARGET   = "alarme"                     # coluna alvo (1=alarme, 0=não alarme)
RPT      = 5                            # repetições
N_SPLITS = 2                            # 5×2 CV -> 2 folds por repetição
SEED0    = 42                           # base para reprodutibilidade
N_BOOT   = 10_000                       # reamostragens bootstrap para IC95%
# Random Forest (só para análise, não produção)
RF_KW = dict(
    n_estimators=500,
    max_depth=None,
    random_state=0,
    n_jobs=-1,
    class_weight="balanced"
)
# ----------------------------

# === utilidade: 5×2 CV estratificado, com seeds diferentes por repetição ===
def five_by_two_splits(y, n_repeats=RPT, seed0=SEED0):
    for r in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=2,
            shuffle=True,
            random_state=seed0 + r
        )
        # gera (train_idx, test_idx) duas vezes por repetição
        yield from skf.split(np.zeros_like(y), y)

# === utilidade: média e IC95% via bootstrap percentile ===
def mean_ci_boot(values, B=N_BOOT, alpha=0.05, seed=0):
    vals = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(B, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return vals.mean(), lo, hi

# === carregar dados ===
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=[TARGET]).to_numpy()
y = df[TARGET].to_numpy()
feat_names = df.drop(columns=[TARGET]).columns

# === coletores ===
recalls, fnrs, f1s = [], [], []
all_importances = []   # (n_models treinos = 10)

# === loop 5×2 CV ===
split_iter = five_by_two_splits(y, n_repeats=RPT, seed0=SEED0)
fold_id = 0
for train_idx, test_idx in split_iter:
    fold_id += 1
    rf = RandomForestClassifier(**RF_KW)
    rf.fit(X[train_idx], y[train_idx])

    y_pred = rf.predict(X[test_idx])

    rec = recall_score(y[test_idx], y_pred, pos_label=1)
    f1  = f1_score(y[test_idx], y_pred, pos_label=1)
    fnr = 1.0 - rec

    recalls.append(rec)
    f1s.append(f1)
    fnrs.append(fnr)
    all_importances.append(rf.feature_importances_)

# === métricas com média ± IC95% (bootstrap) ===
metrics = {
    "Recall": np.array(recalls),
    "FNR":    np.array(fnrs),
    "F1":     np.array(f1s)
}
summary_rows = []
print("=== 5×2 Cross-Validation (10 avaliações) ===")
for name, arr in metrics.items():
    mean, lo, hi = mean_ci_boot(arr, B=N_BOOT, alpha=0.05, seed=123)
    half_width = (hi - lo) / 2
    print(f"{name}: {100*mean:.2f}% ± {100*half_width:.2f}%  (IC95% [{100*lo:.2f}%, {100*hi:.2f}%])")
    summary_rows.append({
        "metric": name,
        "mean": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "pm_halfwidth": half_width
    })

# guardar métricas por fold e resumo
out_dir = Path(".")
pd.DataFrame({
    "fold": np.arange(1, 10+1),
    "recall": recalls,
    "fnr": fnrs,
    "f1": f1s
}).to_csv(out_dir / "rf_metrics_5x2cv.csv", index=False)

pd.DataFrame(summary_rows).to_csv(out_dir / "rf_metrics_summary.csv", index=False)

# === importâncias (média ± desvio-padrão ao longo dos 10 treinos) ===
all_importances = np.vstack(all_importances)  # (10, n_features)
mean_imps = all_importances.mean(axis=0)
std_imps  = all_importances.std(axis=0)

feat_imp_rf = (pd.DataFrame({
    "feature": feat_names,
    "mean_importance": mean_imps,
    "std_importance": std_imps
})
    .sort_values("mean_importance", ascending=False)
    .reset_index(drop=True)
)

feat_imp_rf.to_excel(out_dir / "rf_feature_importances.xlsx", index=False)

# === gráfico igual ao teu (para manter a tua figura/legenda) ===
plt.figure(figsize=(12, 8))
plt.barh(
    feat_imp_rf["feature"],
    feat_imp_rf["mean_importance"],
    xerr=feat_imp_rf["std_importance"],
    align="center"
)
plt.xlabel("Importância média")
plt.title("Importância das Features - Random Forest (5×2 CV)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(out_dir / "random_forest_importances.png", dpi=300, bbox_inches="tight")
plt.show()
print("-> Guardado: rf_metrics_5x2cv.csv, rf_metrics_summary.csv, rf_feature_importances.xlsx, random_forest_importances.png")
