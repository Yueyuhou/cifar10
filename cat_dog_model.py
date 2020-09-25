from cat_dog import data_preprocessing
from cat_dog import model
from cat_dog import show_result
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = r"G:\DeepLearningPractise\keras_get_started\oldCode\cifar-10-python\cifar-10-batches-py"
num_class = 10
batch_size = 100
data_augment = False

data_pre = data_preprocessing.DataLoading(path, num_class)
train_data_img, train_data_label, test_data_img, test_data_label = data_pre.load_data()
print(train_data_img.shape)

if data_augment:
    pass
else:
    train_data_img_gen = ImageDataGenerator(rescale=1. / 255)
    test_data_img_gen = ImageDataGenerator(rescale=1. / 255)

train_data_img_gen.fit(train_data_img)
test_data_img_gen.fit(test_data_img)
train_generator = train_data_img_gen.flow(train_data_img, train_data_label, batch_size=batch_size)
test_generator = test_data_img_gen.flow(test_data_img, test_data_label, batch_size=batch_size)

model = model.Model_1(train_data_img.shape[1:], num_class)
model = model.get_model()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=train_data_img.shape[0]/batch_size, epochs=30,
                              validation_data=test_generator, validation_steps=test_data_img.shape[0]/batch_size)

model.save('cifar10_model_1.h5')

show_result.show_result(history.history)




