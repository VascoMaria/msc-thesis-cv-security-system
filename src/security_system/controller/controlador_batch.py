import json
import time
import asyncio
import aiohttp
from COMMON.common_batch import (
    get_active_categories, 
    get_models_by_category, 
    apply_category_decision_rule, 
    get_global_decision_rule, 
    get_global_threshold, 
    get_global_weights,
    get_alarm_labels,
    get_faces_areas_ratio_threshold
)
from COMMON.logging_config import logger

_session: aiohttp.ClientSession | None = None

async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def send_request(session, file_bytes_list, url, model_name, category):
    """ 
    Envia múltiplas imagens como batch para o modelo.
    """
    form_data = aiohttp.FormData()
    for idx, file_bytes in enumerate(file_bytes_list):
        form_data.add_field(
            name="files",
            value=file_bytes,
            filename=f"image_{idx}.jpg",
            content_type="image/jpeg"
        )

    start_time = time.perf_counter()

    try:
        async with session.post(url, data=form_data, timeout=10) as response:
            if response.status == 200:
                response_data = await response.json()
            else:
                response_data = {"error": await response.text()}
    except Exception as e:
        logger.error("Erro ao enviar request para %s: %s", url, e)
        response_data = {"error": str(e), "url": url}

    return {
        "model": model_name,
        "category": category,
        "time": round(time.perf_counter() - start_time, 4),
        "status": response_data.get("status", "error"),
        "detections": response_data.get(
            "emotions" if category == "emotion_recognition" else "detections", []
        ),
        "faces_areas": response_data.get("faces_areas") if category == "emotion_recognition" else None, 
        "error": response_data.get("error", None),
        "url": url if "error" in response_data else url
    }


async def process_frame(frame_bytes_list: list[bytes]):
    """
    Envia o batch completo para cada modelo (1 request por modelo).
    """
    active_categories = get_active_categories()
    model_urls = get_models_by_category()
    session = await _get_session()

    start_time = time.perf_counter()

    # Envia todas as imagens de uma vez para cada modelo
    tasks = [
        asyncio.create_task(
            send_request(session, frame_bytes_list, url, model_name, category)
        )
        for category, models in model_urls.items()
        if category in active_categories
        for model_name, url in models.items()
    ]
    model_results = await asyncio.gather(*tasks)

    total_time = round(time.perf_counter() - start_time, 4)
    formatted_results = [r for r in model_results if r is not None]

    # DEBUG pois pode ser um volume grande de dados e eu quero sempre ver 
    logger.debug("Formatted results: %s", formatted_results)

    # ============================
    # INFO ADICIONAL: proximidade de faces (não mexe no alarme)
    # ============================
    extra_info = []
    extra_info_detail = {}

    ratio_threshold = get_faces_areas_ratio_threshold()
    if ratio_threshold is not None:
        emo_results = [r for r in formatted_results 
                    if r["category"] == "emotion_recognition" and r.get("status") == "success"]
        for r in emo_results:
            pair = r.get("faces_areas")
            if isinstance(pair, list) and len(pair) == 2:
                user_area = int(pair[0] or 0)
                intruder_area = int(pair[1] or 0)
                if user_area > 0 and intruder_area > 0:
                    ratio = (intruder_area / user_area) * 100
                    if ratio >= ratio_threshold:
                        flag_name = "second_face_near_user"
                        extra_info.append(flag_name)
                        extra_info_detail[flag_name] = {
                            "user_area": user_area,
                            "intruder_area": intruder_area,
                            "ratio": round(ratio, 1),
                            "threshold": ratio_threshold,
                            "model": r.get("model")
                        }
                        break

    logger.debug("Extra info: %s | detail: %s", extra_info, extra_info_detail)



    # ============================
    # 1) DECISÃO POR CATEGORIA
    # ============================
    category_alarms = {}
    for cat in active_categories:
        # filtra apenas os resultados desta categoria
        cat_results = [res for res in formatted_results if res["category"] == cat]
        logger.debug("Resultados da categoria %s: %s", cat, cat_results)

        cat_alarm = apply_category_decision_rule(cat, cat_results)
        category_alarms[cat] = cat_alarm

    logger.info("Category alarms gerados: %s", category_alarms)

    # ============================
    # 2) DECISÃO GLOBAL
    # ============================
    global_rule = get_global_decision_rule()
    global_threshold = get_global_threshold()
    global_weights = get_global_weights()

    logger.debug("Global rule: %s | threshold: %s | weights: %s",
                 global_rule, global_threshold, global_weights)

    if global_rule == "scoring":
        sum_score = 0
        for cat, is_alarm in category_alarms.items():
            if is_alarm:
                w = global_weights.get(cat, 1)
                logger.debug("Somando peso da categoria '%s': %s", cat, w)
                sum_score += w
        alarm = (sum_score >= (global_threshold or 1))

    elif global_rule == "consensus":
        total_cats = len(category_alarms)
        if total_cats == 0:
            alarm = False
        else:
            alarm_cats = sum(1 for cat, is_alarm in category_alarms.items() if is_alarm)
            alarm = (alarm_cats >= total_cats / 2.0)
    else:
        # default => se pelo menos uma der alarme => alarm
        alarm = any(category_alarms.values())
    """ pensar no nome class_weighted_sum para a prioridade (soma entres as classes dos modelos)
        elif global_rule == "prioridade":
            # Exemplo: soma pesos das categorias que alarmaram
            sum_score = 0
            for cat, is_alarm in category_alarms.items():
                if is_alarm:
                    w = global_weights.get(cat, 1)
                    sum_score += w
            alarm = (sum_score >= (global_threshold or 1))
    """
    alarm_detections = []
    if alarm:
        # 3) EXTRAIR DETEÇÕES DE ALARME (uma classe por categoria)
        for cat, is_alarm in category_alarms.items():
            if not is_alarm:
                continue

            # labels permitidas para esta categoria, em lowercase
            allowed = get_alarm_labels(cat)

            # reúne deteções brutas por categoria
            candidates = []
            for res in formatted_results:
                if res["category"] != cat:
                    continue
                dets = res["detections"]
                # emoções retornam dict
                if isinstance(dets, dict):
                    label = dets.get("emotion", "").lower()
                    conf  = dets.get("confidence", 0)
                    if label:
                        candidates.append((label, conf))
                else:
                    if any(isinstance(d, list) for d in dets):
                        dets = [item for sublist in dets for item in sublist]
                    for d in dets:
                        if not isinstance(d, dict):
                            continue
                        lbl = d.get("label", "").lower()
                        cnf = d.get("confidence", 0)
                        if lbl:
                            candidates.append((lbl, cnf))

            # **Filtra só as classes que constam no config (alarmantes)**
            candidates = [(lbl, cnf) for lbl, cnf in candidates if lbl in allowed]

            # seleciona a detecção com maior confiança
            if candidates:
                best = max(candidates, key=lambda x: x[1])
                alarm_detections.append(best[0])

        # remove duplicados, mas mantém ordem
        seen, unique = set(), []
        for x in alarm_detections:
            if x not in seen:
                unique.append(x)
                seen.add(x)
        alarm_detections = unique

    response = {
        "status": "success",
        "data": formatted_results,
        "processing_time": total_time,
        "alarm": alarm,
        "detections": alarm_detections,
        "extra_info": extra_info,                  # flags simples
        "extra_info_detail": extra_info_detail     # detalhe técnico opcional

    }


    # Se quiseres destacar em nível WARNING quando há alarme:
    if alarm:
        logger.warning("ALARME DETECTADO! %s", response)
    else:
        logger.info("Sem alarme. Resultado: %s", response)

    # DEBUG final ou INFO para mostrar a response em formato pretty
    logger.debug("Resposta formatada: %s", json.dumps(response, indent=4))

    return [response] * len(frame_bytes_list)
