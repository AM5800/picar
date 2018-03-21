from model import create_model, input_image_height, input_image_width, input_image_channels
from donkeycar.parts.datastore import TubHandler, TubGroup
import keras

verbose = 1
train_split = .8

X_keys = ['cam/image_array']
y_keys = ['user/angle']

tubgroup = TubGroup("data")
train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, batch_size=128, train_frac=train_split)

save_best = keras.callbacks.ModelCheckpoint('model_{val_loss:.2f}.hdf5',
                                            monitor='val_loss',
                                            verbose=verbose,
                                            save_best_only=True,
                                            mode='min')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=.0005,
                                           patience=5,
                                           verbose=verbose,
                                           mode='auto')

hist = create_model().fit_generator(
    train_gen,
    epochs=100,
    verbose=1,
    validation_data=val_gen,
    callbacks=[save_best, early_stop])

