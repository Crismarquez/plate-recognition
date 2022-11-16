
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORE_DIR = Path(BASE_DIR, "store")
MODELS_DIR = Path(STORE_DIR, "models")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_CONFIGURATION ={
    "motorcycle":{
        "detector_name": "mobilenet_v2", #"mobilenet_v2",
        "ocr_method": "paddle"
    },
    "car":{
        "detector_name": "mobilenet_v2",
        "ocr_method": "paddle"
    }
}

URL_MODELS = {
    "mobilenet_v1": "1nw_uPnm14hzFVjp1E34oK1ObyzBdvk9x",
    "mobilenet_v2": "1a7U05ttb693hx7CR82Sn__P-t_NIUX2B",
    "onnx":"1lunDZ1nlc5On3AUst_LGFVaNpJhg24Rt",
    "pt":"1CS5pJ-lqJgY1-czcU0atflhzNizZ9qCL"
}

# hard rule
TRANSFORM_CHAR = {
    "motorcycle":{
        "last_char":{
            "O": "D",
            "0": "D",
            "4": "A",
            "1": "A"
        },
        "middle_char":{
            "D": "0",
            "A": "4",
            "Z": "2",
            "B": "8"

        },
        "initial_char":{
            "0": "O",
            "4": "A",
            "3": "B"
        }
        
    },
    "car":{
        "last_char":{
            "D": "0",
            "A": "4",
            "Z": "2"
        },
        "initial_char":{
            "0": "O",
            "4": "A",
            "3": "B"
        }
    }
}
