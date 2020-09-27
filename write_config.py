import json

config = {
  "config_par":
  {
    "path":"G:\\DeepLearningPractise\\keras_get_started\\oldCode\\cifar-10-python\\cifar-10-batches-py",
    "num_class" : 10,
    "batch_size" : 100,
    "epoch_num" : 100,
    "rescale" : 1./255,
    "rotation_range" : 30,
    "width_shift_range" : 0.2,
    "height_shift_range" : 0.2,
    "shear_range" : 0,
    "lr" : 0.0001,
    "decay" : 1e-6,
    "horizontal_flip" : True,
    "data_augment" : False,
    "check_point_path" : ".\\check_point\\weights_{epoch:02d}-{val_loss:.2f}.hdf5",
    "my_log_file" : ".\\cat_dog\\log_file"
  }

}

with open('config.json', 'w') as f:
    json.dump(config, f)