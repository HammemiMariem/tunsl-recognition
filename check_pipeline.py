import json
import random
from pathlib import Path

import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("models/tunsl_model.keras")
with open("models/classes.json", encoding="utf-8") as f:
    classes = json.load(f)

exts = {".jpg", ".jpeg", ".png"}
files = [p for d in Path("data/flat_dataset").iterdir() if d.is_dir()
         for p in d.iterdir() if p.suffix.lower() in exts]

random.seed(0)
sample = random.sample(files, 200)

correct = 0
for p in sample:
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR, same as the webcam
    rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
    probs = model(rgb[None].astype("float32"), training=False).numpy()[0]
    correct += classes[int(probs.argmax())] == p.parent.name

print(f"{correct}/{len(sample)} correct")