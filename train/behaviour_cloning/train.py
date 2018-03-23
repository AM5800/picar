import keras
from donkeycar.parts.datastore import TubGroup
from model import create_model

verbose = 2
train_split = .8

X_keys = ['cam/image_array']
y_keys = ['user/angle']

tubgroup = TubGroup("..\..\..\picar_sync\wide-36.6-day,..\..\..\picar_sync\wide-36.6-evening-ccw-recovery")
train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, batch_size=10, train_frac=train_split)

save_best = keras.callbacks.ModelCheckpoint('model_{val_loss:.4f}.hdf5',
                                            monitor='val_loss',
                                            verbose=verbose,
                                            save_best_only=True,
                                            mode='min')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=.0005,
                                           patience=5,
                                           verbose=verbose,
                                           mode='auto')
steps_per_epoch = 100

model = create_model()
model.summary()

hist = model.fit_generator(
    train_gen,
    epochs=100,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
    validation_data=val_gen,
    validation_steps=steps_per_epoch * (1.0 - train_split),
    callbacks=[save_best, early_stop])
