# license plate recognition

## System requirements
ubuntu 18 - 20

python >= 3.7

## Clone repo
<pre>
git clone https://github.com/bluelabsai/plate-recognition.git
cd plate-recognition
</pre> 

## Virtual enviroment
<pre>
python3 -m venv .venv
source .venv/bin/activate
</pre> 

## Install dependencies
<pre>
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:${PWD}"
</pre> 

## Run demo
<pre>
python3 recoplate/main.py interactive "motorcycle" 0 
</pre> 

**Commands: webcam**

**Args: "type_vehicle"   "device"**

allow vehicles [
    "motorcycle",
    "car"
]
