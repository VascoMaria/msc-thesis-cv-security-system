import logging
import os
import sys

# Configuração do logger
logger = logging.getLogger('SecuritySystem_logger')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    # Determina o diretório base:
    # - para executável PyInstaller, usa a pasta do .exe
    # - para script normal, usa a pasta do módulo
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(__file__)

    # Garante que o diretório existe
    os.makedirs(base_dir, exist_ok=True)

    # Handler para arquivo de log em app.log ao lado do executável
    log_path = os.path.join(base_dir, 'app.log')
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(filename)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Handler para saída no terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
