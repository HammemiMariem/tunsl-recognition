"""Real-time sign recognition from the webcam."""
import argparse
import json
from collections import deque

import cv2
import numpy as np
import tensorflow as tf


def center_square(frame):
    """Crop the largest centered square, like the square dataset photos."""
    h, w = frame.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return frame[y0:y0 + s, x0:x0 + s], (x0, y0, s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/tunsl_model.keras")
    parser.add_argument("--classes", default="models/classes.json")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Below this confidence, show 'unknown'")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    with open(args.classes, encoding="utf-8") as f:
        classes = json.load(f)

    history = deque(maxlen=5)  # average the last few frames to reduce flicker
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        crop, (x0, y0, s) = center_square(frame)
        # OpenCV gives BGR, the model was trained on RGB.
        rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
        x = rgb[None].astype("float32")
        probs = model(x, training=False).numpy()[0]
        history.append(probs)
        avg = np.mean(history, axis=0)

        i = int(avg.argmax())
        text = f"{classes[i]} ({avg[i]:.0%})" if avg[i] >= args.threshold else "unknown"
        cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (255, 255, 255), 2)
        cv2.putText(frame, text, (x0 + 10, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)
        cv2.imshow("TunSL recognition (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()