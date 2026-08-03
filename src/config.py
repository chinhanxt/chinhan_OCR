import os
import logging
import torch

MODEL_NAME = os.getenv("MODEL_NAME", "baidu/Unlimited-OCR")

# Configure TF32 and PyTorch GPU optimizations
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chinhan_ocr")
