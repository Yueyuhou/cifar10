import data_preprocessing
import model
import show_result
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# define parameters
with open(r'config.json', 'r') as f:
    config_par = json.load(f)

path = config_par["config_par"]["path"]
num_class = config_par["config_par"]["num_class"]
batch_size = config_par["config_par"]["batch_size"]
epoch_num = config_par["config_par"]["epoch_num"]
rescale = config_par["config_par"]["rescale"]
rotation_range = config_par["config_par"]["rotation_range"]
width_shift_range = config_par["config_par"]["width_shift_range"]
height_shift_range = config_par["config_par"]["height_shift_range"]
shear_range = config_par["config_par"]["shear_range"]
lr = config_par["config_par"]["lr"]
decay = config_par["config_par"]["decay"]
horizontal_flip = config_par["config_par"]["horizontal_flip"]
data_augment = config_par["config_par"]["data_augment"]
check_point_path = config_par["config_par"]["check_point_path"]
my_log_file = config_par["config_par"]["my_log_file"]


if not os.path.exists(my_log_file):
    os.makedirs(my_log_file)



# get data
data_pre = data_preprocessing.DataLoading(path, num_class)
train_data_img, train_data_label, test_data_img, test_data_label = data_pre.load_data()
print(train_data_img.shape)

# data augment
if data_augment:
    train_data_img_gen = ImageDataGenerator(rescale=rescale, rotation_range=rotation_range,
                                            width_shift_range=width_shift_range,
                                            height_shift_range=0.2, shear_range=shear_range,
                                            horizontal_flip=horizontal_flip,
                                            fill_mode='nearest')
else:
    train_data_img_gen = ImageDataGenerator(rescale=rescale)

test_data_img_gen = ImageDataGenerator(rescale=rescale)

train_data_img_gen.fit(train_data_img)
test_data_img_gen.fit(test_data_img)
train_generator = train_data_img_gen.flow(train_data_img, train_data_label, batch_size=batch_size)
test_generator = test_data_img_gen.flow(test_data_img, test_data_label, batch_size=batch_size)

# get model and fit
model = model.Model_1(train_data_img.shape[1:], num_class)
model = model.get_model()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=lr, decay=decay),
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, mode='min',
                                              restore_best_weights=True),
             tf.keras.callbacks.ModelCheckpoint(check_point_path, monitor='val_loss', mode='min', save_best_only=True,
                                                save_freq='epoch'),
             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, mode='min',
                                                  min_delta=0.01, cooldown=3),
             tf.keras.callbacks.TensorBoard(log_dir=my_log_file, histogram_freq=1, update_freq='epoch')]

history = model.fit_generator(train_generator, steps_per_epoch=train_data_img.shape[0] / batch_size, epochs=epoch_num,
                              validation_data=test_generator, validation_steps=test_data_img.shape[0] / batch_size,
                              callbacks=callbacks)

# save model and show result
# if data_augment:
#     model.save("cifar10_model_aug.h5")
# else:
#     model.save('cifar10_model_1.h5')

#show_result.show_result(history.history)
