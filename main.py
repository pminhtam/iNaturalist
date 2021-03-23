from data.data_provider import data_generator
from model.efficient import make_model
from tensorflow.keras import optimizers
import tensorflow as tf
import os
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

if __name__ == "__main__":
    img_size = 256
    batch_size = 64
    n_class = 10000
    nb_epochs = 1000
    checkpoint_dir = "checkpoint"
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    model = make_model(img_size,n_class)
    model.compile(optimizers.Adam(lr=0.0001, decay=1e-6), loss='categorical_crossentropy',
                        metrics=['accuracy'])
    train_generator,val_generator = data_generator("/root/kaggle/train_mini.json","/root/kaggle",img_size,batch_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0,
            patience=10
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, verbose=1, min_delta=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'efficientb3.h5'),verbose=1,
            monitor="val_loss", save_weights_only=True, save_freq='epoch'
        )
    ]
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=nb_epochs,
                        callbacks=callbacks,
                        verbose=1)
    with open('history.json', 'w') as f:
        json.dump(history.history, f)
