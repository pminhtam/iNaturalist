from efficientnet.model import EfficientNetB3
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

def make_model(img_size,n_class):
    # img_size = 256
    model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    print(model)
    model.trainable = False
    # Adding custom layers
    x = model.output
    x = Flatten()(x)
    x = Dense(16384, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_class, activation="softmax")(x)
    model_final = Model(model.input, predictions)
    return model_final

