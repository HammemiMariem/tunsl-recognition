# Tunisian Sign Language (TunSL) Recognition

Image classifier for 57 Tunisian Sign Language words, built with TensorFlow/Keras
(MobileNetV2 transfer learning) and a real-time OpenCV webcam demo.

## Dataset

[(First ever) Tunisian Sign Language Dataset](https://doi.org/10.17632/fbjjgzgv7f.1)
by Chakroun & Jerbi (2023), Mendeley Data: 4,423 images, 57 signs, 7 signers.
It is not included in this repo. Download it separately and respect its license.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Usage

```bash
# 1. Flatten category/sign/images into sign/images
python flatten_dataset.py --src "data/(First ever) Tunisian Sign Language Dataset/Data" --dst data/flat_dataset

# 2. Train (saves models/tunsl_model.keras and models/classes.json)
python train.py --epochs 30

# 3. Check the preprocessing pipeline on dataset images
python check_pipeline.py

# 4. Live webcam demo (press q to quit)
python webcam.py
```

## Model

- MobileNetV2 (ImageNet weights, frozen) + global pooling + dropout + softmax
- Augmentation: rotation, zoom, translation, contrast, brightness (no horizontal flip,
  since mirroring can change the meaning of a sign)
- 80/20 random train/validation split

## Results and limitations

- Validation accuracy: about 90%. `check_pipeline.py` gets 181/200 on a sample of
  dataset images, confirming the preprocessing is correct.
- **The model does not generalize well to a new webcam.** The dataset has only 7 signers
  and the validation split is random, so the same people and rooms appear in training
  and validation. Validation accuracy therefore overestimates real-world performance.
  On a new camera and room, the live demo often predicts one favorite class.
- Signs are really motion, and single images lose information.

## Ideas for improvement

- Add your own photos per sign to match your camera and room
- MediaPipe hand landmarks to ignore background and clothing
- Video clips with an LSTM or Transformer over landmark sequences
- Convert to TFLite for a mobile app

## Credits

Chakroun, W., Jerbi, R. (2023), "(First ever) Tunisian Sign Language Dataset",
Mendeley Data, V1, doi: 10.17632/fbjjgzgv7f.1