import pandas
import tensorflow as tf
from matplotlib import pyplot as plt

# obtention des des dataset à partir des fichiers d'entrès
data_train = pandas.read_csv("train.csv", sep=",")
label = data_train.pop("label")
data_train = data_train.astype("float32")
data_train_norm = data_train / 255
data_train_norm = tf.reshape(data_train_norm, [42000, 28, 28, 1])



# division du dataset pour séparer le jeu de test du jeu de validation
data_train = tf.convert_to_tensor(data_train_norm)
label = tf.convert_to_tensor(label)
train_label = label[:29400]
train_data = data_train[:29400]
val_label = label[29400:]
val_data = data_train[29400:]


# layer d'augmenation de données (rotation et translation)
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

# réseau de neurones pour l'aprentissage
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    augmentation,

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

# affichage des différentes métriques obtenus au cours de l'apprentissage
history = model.fit(train_data, train_label, validation_data=(val_data, val_label), epochs=100)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.9, 1])
plt.legend(loc='lower right')
plt.show()


# écriture des résultats pour soumettre la réponse à kaggle
data_test = pandas.read_csv("test.csv", sep=",")
data_test = data_test.astype("float32")
data_test = data_test / 255
data_test = tf.reshape(data_test, [28000, 28, 28, 1])
res = tf.argmax(model.predict(data_test), axis=1)

with open("res.csv", "w") as fi:
    fi.write("ImageId,Label\n")
    res = res.numpy()
    for i in range(1, 28001):
        fi.write(str(i) + "," + str(res[i - 1]) + "\n")
