import cv2
import numpy as np
from pydantic import BaseModel
import json
import os
from COMMON.logging_config import logger
from functools import lru_cache
import sys

def get_config_path() -> str:
    """
    Retorna o caminho do config.json:
      - se for um executável PyInstaller (sys.frozen), usa a pasta do .exe
      - senão, usa a pasta do módulo
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(__file__)
    return os.path.join(base, "config.json")

CONFIG_FILE = get_config_path()

# MEMORIA - Guardar em memoria como o Paulo tinha pedido
@lru_cache(maxsize=1)
def load_full_config() -> dict:
    """
    Carrega e guarda em cache o conteúdo de config.json na primeira vez que for chamado.
    Nas chamadas seguintes, devolve o mesmo dict sem reabrir o ficheiro.
    """
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_config = load_full_config()
CATEGORY_MODEL_CLASSES    = {}
CATEGORY_MODEL_THRESHOLDS = {}
CATEGORY_MODEL_WEIGHTS    = {}

for category_cfg in _config.get("categories", []):
    cat_name = category_cfg.get("name")
    if not cat_name:
        continue

    rule = category_cfg.get("decision_rule", "")
    classes_map = {}
    thresh_map  = {}
    weight_map  = {}

    for m in category_cfg.get("models", []):
        m_name = m.get("name")
        if not m_name:
            continue
        # classes (lowercase das keys)
        raw_classes = m.get("classes", {})
        # 1) se for dict: mantém peso original
        if isinstance(raw_classes, dict):
            classes_map[m_name] = {k.lower(): v for k, v in raw_classes.items()}
        # 2) se for lista: atribui peso 1 a cada label
        elif isinstance(raw_classes, list):
            classes_map[m_name] = {str(label).lower(): 1 for label in raw_classes}
        else:
            # fallback: sem classes
            classes_map[m_name] = {}
        # thresholdDetection (default 0.5)
        thresh_map[m_name] = m.get("thresholdDetection", 0.5)
        # weight (só para regra consensus)
        weight_map[m_name] = m.get("weight", 0)

    CATEGORY_MODEL_CLASSES[cat_name]    = classes_map
    CATEGORY_MODEL_THRESHOLDS[cat_name] = thresh_map
    CATEGORY_MODEL_WEIGHTS[cat_name]    = weight_map

# Acesso aos mapas em memória

def get_model_classes(category_name: str) -> dict[str, int]:
    return CATEGORY_MODEL_CLASSES.get(category_name, {})


def get_model_thresholds(category_name: str) -> dict[str, float]:
    return CATEGORY_MODEL_THRESHOLDS.get(category_name, {})


def get_model_weights(category_name: str) -> dict[str, int]:
    return CATEGORY_MODEL_WEIGHTS.get(category_name, {})


def get_faces_areas_ratio_threshold():
    config = load_full_config()
    return config.get("faces_areas_ratio_threshold")

################################################################################### 

def get_active_categories():
    config = load_full_config()
    # Substituição de print por logger.debug
    logger.debug("Config carregada: %s", config)
    active = []

    for category in config.get("categories", []):
        if category.get("active"):
            active.append(category.get("name"))

    return active

def get_models_by_category():
    config = load_full_config()
    result = {}

    for category in config.get("categories", []):
        cat_name = category.get("name")
        models = category.get("models", [])
        result[cat_name] = {}

        for model in models:
            model_name = model.get("name")
            model_url = model.get("url")
            if model_name and model_url:
                result[cat_name][model_name] = model_url

    return result

# Regra de decisão global (entre categorias)
def get_global_decision_rule():
    config = load_full_config()
    return config.get("categories_decision_rule", "consensus")  # default = consensus

# Threshold global (só usado se for 'scoring' ou 'prioridade')
def get_global_threshold():
    config = load_full_config()
    return config.get("threshold")

# Pesos entre categorias (só se for 'scoring' ou 'prioridade')
def get_global_weights():
    config = load_full_config()
    map_global_weights = {}
    for category in config.get("categories", []):
        map_global_weights[category.get("name")] = category.get("weight")
    return map_global_weights

# Regra de decisão por categoria (ex: weapon_detection)
def get_category_decision_rule(category_name):
    config = load_full_config()
    for category in config.get("categories", []):
        if category.get("name") == category_name:
            return category.get("decision_rule", "consensus")
    return None

# Threshold da categoria
def get_category_threshold(category_name):
    config = load_full_config()
    for category in config.get("categories", []):
        if category.get("name") == category_name:
            return category.get("threshold")
    return None

# Pesos dos modelos da categoria
def get_category_weights(category_name):
    config = load_full_config()
    for category in config.get("categories", []):
        if category.get("name") == category_name:
            return category.get("weight", {})
    return {}

from typing import List, Dict, Any

def apply_category_decision_rule(category_name: str, model_results: List[Dict[str, Any]]) -> bool:
    """
    Aplica a regra de decisão da categoria (scoring, consensus, prioridade...)
    e retorna True (alarme) ou False (sem alarme).

    model_results: Lista de resultados APENAS da categoria 'category_name'.
      Cada item normalmente é do tipo:
        {
          "model": "best",
          "category": "weapon_detection",
          "status": "success",
          "detections": [
             { "label": "gun", "confidence": 0.78 },
             { "label": "knife", "confidence": 0.45 }
          ],
          ...
        }
    """
    logger.debug("model_results para '%s': %s", category_name, model_results)

    # 1) Buscar do config a rule, threshold, weights, e as classes de cada modelo
    rule = get_category_decision_rule(category_name)
    logger.debug("Regra obtida para '%s': %s", category_name, rule)
    threshold = get_category_threshold(category_name)

     # 2) Pega os mapas pré-carregados em memória
    model_classes_map   = get_model_classes(category_name)
    model_threshold_map = get_model_thresholds(category_name)
    model_weights_map   = get_model_weights(category_name)

    logger.debug("model_classes_map: %s", model_classes_map)
    logger.debug("model_threshold_map: %s", model_threshold_map)
    logger.debug("model_weights_map: %s", model_weights_map)
    logger.info("Regras de modelo para '%s': %s", category_name, model_classes_map)

    # 2) Dependendo da rule
    if rule == "scoring":
        return apply_scoring_rule(
            model_results,
            model_classes_map,
            model_threshold_map,
            model_weights_map,
            threshold
        )
    elif rule == "consensus":
        return apply_consensus_rule(
            model_results,
            model_classes_map,
            model_threshold_map
        )
    elif rule == "prioridade":
        return apply_prioridade_rule(
            model_results,
            model_classes_map,
            model_threshold_map,
            model_weights_map,
            threshold
        )
    else:
        # Se não houver rule definida, ou for algo desconhecido, retorna False
        logger.warning("Regra desconhecida '%s' para '%s'", rule, category_name)
        return False

def apply_scoring_rule(
    model_results: List[Dict[str, Any]],
    model_classes_map: Dict[str, Dict[str, int]],
    model_threshold_map: Dict[str, float],
    model_weights_map: Dict[str, int],
    threshold: int
) -> bool:
    """
    Regra scoring: soma pesos dos modelos que detetaram classes alarmantes
    (com confidence >= thresholdDetection).
    """
    if threshold is None:
        threshold = 1

    sum_score = 0
    for r in model_results:
        if r.get("status") == "success":
            detections = r.get("detections", [])
            model_name = r.get("model")

            if isinstance(detections, dict):
                detections = [detections]
            elif isinstance(detections, list) and any(isinstance(d, list) for d in detections):
                detections = [item for sublist in detections for item in sublist]


            for d in detections:
                label_lower = d["emotion" if r.get("category") == "emotion_recognition" else "label"].lower()
                conf = d.get("confidence", 1.0)

                if (
                    label_lower in model_classes_map.get(model_name, []) and
                    conf >= model_threshold_map.get(model_name, 0.5)
                ):
                    weight = model_weights_map.get(model_name, 1)
                    logger.debug("Modelo '%s' detetou '%s' com confiança %.2f — peso aplicado: %s", model_name, label_lower, conf, weight)
                    sum_score += weight
                    break


    logger.debug("Sum_score calculado: %s | Threshold: %s", sum_score, threshold)
    return (sum_score >= threshold)

def apply_consensus_rule(
    model_results: List[Dict[str, Any]],
    model_classes_map: Dict[str, List[str]],
    model_threshold_map: Dict[str, float]
) -> bool:
    """
    Regra consensus: se mais de metade dos modelos tiverem alarme, é True.
    Só conta como alarme se a deteção tiver confidence >= thresholdDetection do modelo.
    """
    total_models = len(model_results)
    logger.debug("Total de modelos na categoria: %s", total_models)
    if total_models == 0:
        return False

    alarm_models = 0
    for r in model_results:
        if r.get("status") != "success":
            continue

        detections = r.get("detections", [])
        model_name = r.get("model")

        # Unifica todos os formatos possíveis:
        # Pode ser:
        # - uma lista de dicts => ok
        # - uma lista de listas => achata
        # - um dict (ex: emoções) => transforma em lista
        if isinstance(detections, dict):
            detections = [detections]
        elif isinstance(detections, list) and any(isinstance(d, list) for d in detections):
            detections = [item for sublist in detections for item in sublist]

        found_alarm = False
        for d in detections:
            label_key = "emotion" if r.get("category") == "emotion_recognition" else "label"
            label = d.get(label_key, "").lower()
            conf = d.get("confidence", 1.0)

            if (
                label in model_classes_map.get(model_name, []) and
                conf >= model_threshold_map.get(model_name, 0.5)
            ):
                found_alarm = True
                break

        if found_alarm:
            alarm_models += 1

    logger.debug("Modelos com alarme: %s | Necessário >= metade para True", alarm_models)
    return (alarm_models >= (total_models / 2.0))

def apply_prioridade_rule(
    model_results: List[Dict[str, Any]],
    model_classes_map: Dict[str, List[str]],
    model_threshold_map: Dict[str, float],
    weights: Dict[str, int],
    threshold: int
) -> bool:
    """
    Regra 'prioridade'  VALOR POR CLASSE ALARMANTE:
    1) Se algum modelo (alarmante) tiver weight >= threshold, retornamos True imediatamente.
    2) Se não, somamos peso de todos os modelos que alarmaram:
       - Se a soma >= threshold, também retorna True.
    """
    if threshold is None:
        threshold = 1

    sum_score = 0
    for r in model_results:
        if r.get("status") == "success":
            detections = r.get("detections", [])
            model_name = r.get("model")

            if isinstance(detections, dict):
                detections = [detections]
            elif isinstance(detections, list) and any(isinstance(d, list) for d in detections):
                detections = [item for sublist in detections for item in sublist]

                
            for d in detections:
                label_lower = d["emotion" if r.get("category") == "emotion_recognition" else "label"].lower()
                conf = d.get("confidence", 1.0)

                if (
                    label_lower in model_classes_map.get(model_name, []) and
                    conf >= model_threshold_map.get(model_name, 0.5)
                ):
                    alarm_classe = model_classes_map.get(model_name).get(label_lower)
                    logger.debug("alarm_classe encontrado: %s", alarm_classe)
                    sum_score += alarm_classe
                    # Encontrou alarme para esse modelo, não é preciso verificar mais deteções
                    break

    logger.debug("Sum_score calculado: %s | Threshold: %s", sum_score, threshold)
    return (sum_score >= threshold)


def check_camera_status(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = gray.size

    dark_pixels = np.sum(hist[:50])
    bright_pixels = np.sum(hist[200:])

    if dark_pixels > total_pixels * 0.9:
        return "Camera Bloqueada"
    elif bright_pixels > total_pixels * 0.9:
        return "Excesso de Luz"

    return "Camera Normal"



@lru_cache(maxsize=1)
def _build_alarm_labels_map() -> dict[str, set[str]]:
    """
    Constrói um mapa {categoria: set(de labels alarmantes em lowercase)} a partir
    do _config já carregado em memória.
    Suporta:
      - classes definidas como dict: keys são labels e values podem ser pesos
      - classes definidas como lista: cada item é uma label
    """
    alarm_map: dict[str, set[str]] = {}
    for cat_cfg in _config.get("categories", []):
        cat_name = cat_cfg.get("name")
        if not cat_cfg.get("active") or not cat_name:
            continue

        labels: set[str] = set()
        for m in cat_cfg.get("models", []):
            raw = m.get("classes", {})
            if isinstance(raw, dict):
                # usa as chaves do dict
                labels.update(k.lower() for k in raw.keys())
            elif isinstance(raw, list):
                # cada item da lista é uma label
                labels.update(str(item).lower() for item in raw)
            else:
                # fallback: ignora formatos inesperados
                continue

        alarm_map[cat_name] = labels

    return alarm_map

def get_alarm_labels(category_name: str) -> set[str]:
    """
    Retorna o set de labels alarmantes (lowercase) para a categoria,
    ou set() se categoria não existir.
    """
    return _build_alarm_labels_map().get(category_name, set())

"""
class Settings(BaseModel):
    detect_environment: bool = False
    detect_emotions: bool = False
    detect_weapons: bool = True

global_settings = Settings()

def update_global_settings(new_settings: Settings):
    global global_settings
    global_settings = new_settings
    
    save_global_settings()

def get_global_settings():
    return load_global_settings()


#def save_global_settings():
#    with open(CONFIG_FILE, "w") as f:
#       json.dump(global_settings.dict(), f, indent=4)

#def load_global_settings():
#    try:
#        with open(CONFIG_FILE, "r") as f:
#            data = json.load(f)
#            return Settings(**data)
#    except (FileNotFoundError, json.JSONDecodeError):
#        return Settings()

"""
