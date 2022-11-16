from pathlib import Path

import gdown

from config.config import MODELS_DIR

dest_path = Path(MODELS_DIR, "mobilenet_v2")

url="https://drive.google.com/drive/folders/?usp=sharing"
file_id = "1a7U05ttb693hx7CR82Sn__P-t_NIUX2B"

gdown.download_folder(
    id=file_id,
     output=str(dest_path),
     use_cookies=False
)
