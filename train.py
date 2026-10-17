"""Train a MobileNetV2 transfer-learning classifier on the TunSL dataset."""
import argparse
import json
from pathlib import Path

import tensorflow as tf

IMG_SIZE = (224, 224)


def build_model(num_classes: int) -> tf.keras.Model:
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(0.15, 0.15),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.RandomBrightness(0.3),
        # No horizontal flip: mirroring can change the meaning of a sign.
    ])

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = tf.keras.Input(IMG_SIZE + (3,))
    h = augment(inputs)
    h = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(h)
    h = base(h, training=False)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dropout(0.5)(h)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(h)
    return tf.keras.Model(inputs, outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/flat_dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    common = dict(validation_split=0.2, seed=42,
                    image_size=IMG_SIZE, batch_size=args.batch_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data, subset="validation", **common)
    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model(len(class_names))
    model.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "tunsl_model.keras")
    (out / "classes.json").write_text(json.dumps(class_names, ensure_ascii=False), encoding="utf-8")
    print(f"Saved model and classes to {out}/")


if __name__ == "__main__":
    main()