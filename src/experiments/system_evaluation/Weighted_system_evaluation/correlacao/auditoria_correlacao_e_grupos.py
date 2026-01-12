
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def infer_categories_from_names(columns: List[str]) -> Dict[str, List[str]]:
    cats = {"weapons": [], "emotions": [], "environment": [], "other": []}
    for c in columns:
        cl = c.lower()
        if cl.startswith("weapon_"):
            cats["weapons"].append(c)
        elif cl.startswith("emotion_"):
            cats["emotions"].append(c)
        elif cl.startswith("environment_"):
            cats["environment"].append(c)
        else:
            cats["other"].append(c)
    if not cats["other"]:
        cats.pop("other", None)
    return cats


def load_category_map(path: str, columns: List[str]) -> Dict[str, List[str]]:
    if path is None or not os.path.exists(path):
        return infer_categories_from_names(columns)
    with open(path, "r", encoding="utf-8") as f:
        user_map = json.load(f)
    flat = [m for lst in user_map.values() for m in lst]
    missing = [m for m in flat if m not in columns]
    if missing:
        raise ValueError(f"Modelos em categories_map inexistentes nas colunas: {missing}")
    return user_map


def compute_correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corr_p = df.corr(method="pearson")
    corr_s = df.corr(method="spearman")
    return corr_p, corr_s


def list_high_corr_pairs(corr_p: pd.DataFrame, corr_s: pd.DataFrame, threshold: float) -> pd.DataFrame:
    cols = corr_p.columns.tolist()
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rp = corr_p.iloc[i, j]
            rs = corr_s.iloc[i, j]
            if abs(rp) > threshold or abs(rs) > threshold:
                rows.append(
                    {
                        "model_a": cols[i],
                        "model_b": cols[j],
                        "pearson": float(rp),
                        "spearman": float(rs),
                        "abs_max": float(max(abs(rp), abs(rs))),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["model_a","model_b","pearson","spearman","abs_max"])
    return df.sort_values("abs_max", ascending=False).reset_index(drop=True)


def connected_components_from_threshold(corr_p: pd.DataFrame, corr_s: pd.DataFrame, threshold: float):
    cols = corr_p.columns.tolist()
    n = len(cols)
    R = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rp = corr_p.iloc[i, j]
            rs = corr_s.iloc[i, j]
            if abs(rp) > threshold or abs(rs) > threshold:
                R[i, j] = True

    visited = set()
    comps = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        comp = set([i])
        while stack:
            u = stack.pop()
            for v in range(n):
                if R[u, v] and v not in comp:
                    comp.add(v)
                    stack.append(v)
        visited |= comp
        if len(comp) > 1:
            comps.append([cols[k] for k in sorted(list(comp))])
    return comps


def save_heatmap(corr: pd.DataFrame, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Auditoria de correlação e agrupamento por categorias/grupos.")
    parser.add_argument("-i", "--input_csv", required=True, help="Caminho para o CSV com as saídas dos modelos.")
    parser.add_argument("-o", "--out_dir", default="auditoria_out", help="Diretório de saída para relatórios.")
    parser.add_argument("-g", "--ground_truth_col", default="alarme", help="Nome da coluna de ground truth (será excluída das features).")
    parser.add_argument("-t", "--threshold", type=float, default=0.8, help="Limiar de correlação para formar grupos (padrão=0.8).")
    parser.add_argument("-m", "--categories_map_json", default=None, help="JSON opcional com mapeamento manual de categorias.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if args.ground_truth_col in df.columns:
        features = df.drop(columns=[args.ground_truth_col])
    else:
        features = df.copy()

    categories = load_category_map(args.categories_map_json, features.columns.tolist())

    corr_p, corr_s = compute_correlations(features)
    corr_p.to_csv(os.path.join(args.out_dir, "corr_pearson.csv"))
    corr_s.to_csv(os.path.join(args.out_dir, "corr_spearman.csv"))

    high_pairs = list_high_corr_pairs(corr_p, corr_s, args.threshold)
    high_pairs_path = os.path.join(args.out_dir, "high_corr_pairs.csv")
    high_pairs.to_csv(high_pairs_path, index=False)

    comps = connected_components_from_threshold(corr_p, corr_s, args.threshold)
    groups = {f"group_{i+1}": comp for i, comp in enumerate(comps)}
    with open(os.path.join(args.out_dir, "groups_by_correlation.json"), "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "categories.json"), "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)

    save_heatmap(corr_p, os.path.join(args.out_dir, "heatmap_pearson.png"), "Correlação (Pearson)")
    save_heatmap(corr_s, os.path.join(args.out_dir, "heatmap_spearman.png"), "Correlação (Spearman)")

    stats_rows = []
    for cat, mods in categories.items():
        if len(mods) >= 2:
            sub = corr_p.loc[mods, mods].copy()
            mask = ~np.eye(len(mods), dtype=bool)
            mean_intra = sub.values[mask].mean()
        else:
            mean_intra = np.nan
        stats_rows.append({"categoria": cat, "n_modelos": len(mods), "corr_pearson_media_intra": mean_intra})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(args.out_dir, "categoria_stats.csv"), index=False)

    print("=== Auditoria de Correlação ===")
    print(f"Ficheiro: {args.input_csv}")
    print(f"Modelos: {', '.join(features.columns)}")
    print(f"Limiar (threshold): {args.threshold}")
    print("\nCategorias inferidas/fornecidas:")
    for cat, mods in categories.items():
        print(f"  - {cat}: {mods}")
    print("\nGrupos por correlação (ρ>|threshold|):")
    if groups:
        for g, mods in groups.items():
            print(f"  * {g}: {mods}")
    else:
        print("  (nenhum grupo com correlação alta)")
    print(f"\nPares com alta correlação guardados em: {high_pairs_path}")
    print(f"Outputs em: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
