import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(model, test_dataset):
    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        predicted_classes = tf.argmax(predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_names = test_dataset.class_names

    cm = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=len(class_names)
    ).numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(
        np.arange(len(class_names)),
        class_names,
        rotation=45
    )
    plt.yticks(
        np.arange(len(class_names)),
        class_names
    )

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


class Classifier:
    def __init__(self):
        self.data_dir = "datasets/gitignored"
        self.batch_size = 32
        self.img_height = 256
        self.img_width = 256

    def load_dataset(self, directory):
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            color_mode="grayscale",
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

    def train(self):
        train_dataset = self.load_dataset(self.data_dir + "/train")
        test_dataset = self.load_dataset(self.data_dir + "/test")

        model = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 1)),

            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Conv2D(64, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.15),

            tf.keras.layers.Conv2D(128, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(256, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=10
        )
        # Evaluate final performance
        test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        plot_confusion_matrix(model, test_dataset)
