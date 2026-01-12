# -*- coding: utf-8 -*-
"""
Validação via /detect_batch (batch=1), parser ITERATIVO (sem recursão)
- Espera TODOS os 7 modelos por imagem antes de avançar
- Guarda só a MAIOR confiança das classes alarmantes por modelo
- Exporta Excel: imagem, 7 colunas (modelos), alarme (1/0)
"""

import os, sys, json, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests, pandas as pd

# ---------- CAMINHOS ----------
DATASET_DIR = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\DatasetValidacao\sintetico"
OUTPUT_XLSX = os.path.join(DATASET_DIR, "resultado_validacao.xlsx")
DEBUG_DIR = os.path.join(DATASET_DIR, "_debug_responses")

# ---------- MODELOS ----------
MODEL_ORDER = [
    "Modelo Best",
    "Modelo Best v2",
    "Modelo Modal",
    "Modelo Efficient-net",
    "Modelo DeepFace",
    "Modelo Fer",
    "Modelo Violence",
]
MODELS = {
    "Modelo Best":          "http://localhost:8000/detect_batch",
    "Modelo Best v2":       "http://localhost:8007/detect_batch",
    "Modelo Modal":         "http://localhost:8006/detect_batch",
    "Modelo Efficient-net": "http://localhost:8010/detect_batch",
    "Modelo DeepFace":      "http://localhost:8001/detect_batch",
    "Modelo Fer":           "http://localhost:8012/detect_batch",
    "Modelo Violence":      "http://localhost:8009/detect_batch",
}
ALARM_LABELS = {
    "Modelo Best":          {"guns", "gun", "knife"},
    "Modelo Best v2":       {"gun"},
    "Modelo Modal":         {"gun", "knife", "pistol", "grenade", "rifle", "handgun"},
    "Modelo Efficient-net": {"weapon"},
    "Modelo DeepFace":      {"fear", "angry"},
    "Modelo Fer":           {"fear", "angry"},
    "Modelo Violence":      {"violence"},
}

# ---------- CONFIG ----------
REQUEST_TIMEOUT = 30
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def extract_max_alarm_conf_iterative(resp_json, allowed_labels: set) -> float:
    """
    Parser SEM recursão: percorre o JSON com uma pilha.
    Procura pares (label-like, confidence) e também dicionários de probabilidades.
    """
    allowed = {str(x).strip().lower() for x in allowed_labels}
    LABEL_KEYS = ("label", "class", "class_name", "name", "emotion", "category", "type")

    # Se for batch top-level unitário, usa o primeiro; senão usa como veio
    root = resp_json[0] if isinstance(resp_json, list) and len(resp_json) == 1 else resp_json

    best = None
    stack = [root]
    seen_ids = set()
    processed = 0
    MAX_NODES = 200000

    def try_add(label_val, conf_val):
        nonlocal best
        if label_val is None or conf_val is None:
            return
        try:
            lbl = str(label_val).strip().lower()
            if lbl in allowed:
                c = float(conf_val)
                best = c if (best is None or c > best) else best
        except Exception:
            pass

    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen_ids: 
            continue
        seen_ids.add(nid)
        processed += 1
        if processed > MAX_NODES:
            break

        if isinstance(node, dict):
            # item com confidence + rótulo
            if "confidence" in node:
                lbl = None
                for k in LABEL_KEYS:
                    if k in node:
                        lbl = node[k]; break
                try_add(lbl, node.get("confidence"))

            # probabilidades como dict
            for prob_key in ("probabilities", "probs", "scores", "softmax", "logits"):
                if prob_key in node and isinstance(node[prob_key], dict):
                    for lbl, sc in node[prob_key].items():
                        try_add(lbl, sc)

            # labels/scores paralelos
            if (
                "labels" in node and "scores" in node and
                isinstance(node["labels"], list) and isinstance(node["scores"], list)
            ):
                for lbl, sc in zip(node["labels"], node["scores"]):
                    try_add(lbl, sc)

            # estruturas comuns
            for key in ("detections", "predictions", "objects", "emotions", "results", "data"):
                if key in node and isinstance(node[key], (list, dict)):
                    stack.append(node[key])

            # varrer o resto
            for v in node.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)

        elif isinstance(node, list):
            # empurra todos os itens; não usamos node[0] cegamente
            for it in node:
                if isinstance(it, (dict, list)):
                    stack.append(it)

    return float(best) if best is not None else 0.0

def send_one(url: str, img_path: Path):
    files = [("files", (img_path.name, open(img_path, "rb"), "application/octet-stream"))]
    try:
        r = requests.post(url, files=files, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return json.loads(r.text)
    except Exception:
        print(f"[AVISO] Falha em {url} para {img_path.name}", file=sys.stderr)
        traceback.print_exc()
        return None
    finally:
        try: files[0][1][1].close()
        except Exception: pass

def maybe_dump_debug(model_name: str, img_name: str, payload):
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        with open(os.path.join(DEBUG_DIR, f"{model_name}__{img_name}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def process_image(img_path: Path, executor: ThreadPoolExecutor) -> dict:
    futs = {name: executor.submit(send_one, url, img_path) for name, url in MODELS.items()}
    row = {"imagem": img_path.name}
    for model_name in MODEL_ORDER:
        resp = futs[model_name].result()
        conf = 0.0 if resp is None else extract_max_alarm_conf_iterative(resp, ALARM_LABELS[model_name])
        if conf == 0.0 and resp is not None:
            maybe_dump_debug(model_name, img_path.name, resp)
        row[model_name] = conf
    return row

def main():
    print(">>> Versão SEM RECURSÃO ativa (extract_max_alarm_conf_iterative).")
    base = Path(DATASET_DIR)
    pos_dir = base / "positivo"
    neg_dir = base / "negativo"
    if not pos_dir.exists() or not neg_dir.exists():
        print("Pastas 'positivo' e/ou 'negativo' não encontradas.", file=sys.stderr)
        sys.exit(1)

    rows = []
    with ThreadPoolExecutor(max_workers=7) as executor:
        for img in sorted(pos_dir.rglob("*")):
            if is_image(img):
                row = process_image(img, executor)
                row["alarme"] = 1
                rows.append(row)
        for img in sorted(neg_dir.rglob("*")):
            if is_image(img):
                row = process_image(img, executor)
                row["alarme"] = 0
                rows.append(row)

    cols = ["imagem"] + MODEL_ORDER + ["alarme"]
    pd.DataFrame(rows, columns=cols).to_excel(OUTPUT_XLSX, index=False)
    print(f"✅ Concluído. Excel salvo em: {OUTPUT_XLSX}")
    print(f"ℹ️  Se alguma confiança ficou 0.0, o JSON bruto foi salvo (quando recebido) em: {DEBUG_DIR}")

if __name__ == "__main__":
    main()
