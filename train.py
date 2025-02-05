import tensorflow as tf
import tensorflow_datasets as tfds

# 데이터 로드 및 전처리
dataset, info = tfds.load("oxford_flowers102", as_supervised=True, with_info=True)
train_dataset, test_dataset, validation_dataset = dataset["train"], dataset["test"], dataset["validation"]

# 데이터 증강
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

# 데이터 전처리 함수
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

train_dataset = train_dataset.map(preprocess_image).shuffle(1020).batch(32)
validation_dataset = validation_dataset.map(preprocess_image).batch(32)

# 모델 생성
base_model = tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(102, activation="softmax")
])

# 학습률 스케줄링 및 조기 종료
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True)

# 학습
model.fit(train_dataset, validation_data=validation_dataset, epochs=27, callbacks=[early_stopping, checkpoint_cb])

# 모델 저장
model.save("oxford_flowers_model.h5")