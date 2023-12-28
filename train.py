import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    labels = []

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(1 if "yes" in folder else 0)

    return images, labels


def create_dataset():
    yes_images, yes_labels = load_images_from_folder("assets/yes")
    no_images, no_labels = load_images_from_folder("assets/no")

    images = np.concatenate((yes_images, no_images), axis=0)
    labels = np.concatenate((yes_labels, no_labels), axis=0)

    images = np.array(images)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def build_model(input_shape=(128, 128, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    X_train, X_test, y_train, y_test = create_dataset()

    print("Eğitim seti şekli:", X_train.shape)
    print("Test seti şekli:", X_test.shape)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    model = build_model(input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test doğruluğu: {test_acc}")

    model.save('models/brain_model.keras')

if __name__ == "__main__":
    main()