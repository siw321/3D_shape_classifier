import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt





class Classifier:
    def __init__(self, data_dir = "datasets/gitignored", batch_size = 32):
        self.model = None

        self.train_dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir + "/train",
            color_mode="grayscale",
            batch_size=batch_size
        )
        self.test_dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir + "/test",
            color_mode="grayscale",
            batch_size=batch_size
        )

    def plot_confusion_matrix(self):
        y_true = []
        y_pred = []

        for images, labels in self.test_dataset:
            predictions = self.model.predict(images, verbose=0)
            predicted_classes = tf.argmax(predictions, axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted_classes.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        class_names = self.test_dataset.class_names

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

    def build(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Resizing(64, 64),
            tf.keras.layers.Rescaling(1. / 255),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.GlobalMaxPooling2D(),

            tf.keras.layers.Dense(3, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def train(self):
        self.model.fit(
            self.train_dataset,
            validation_data=self.test_dataset,
            epochs=10
        )
