import json
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# label_file_path = "/home/dell/train_mini.json"
def data_generator(label_file_path,image_folder_path,img_size,batch_size,val_file_path,val_folder_path):
    with open(label_file_path) as data_file:
        train_anns = json.load(data_file)
    with open(val_file_path) as data_file:
        val_anns = json.load(data_file)
    train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id', 'category_id']]
    train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'})
    df_train_file_cat = pd.merge(train_img_df, train_anns_df, on='image_id')
    df_train_file_cat['category_id'] = df_train_file_cat['category_id'].astype(str)

    val_anns_df = pd.DataFrame(val_anns['annotations'])[['image_id', 'category_id']]
    val_img_df = pd.DataFrame(val_anns['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'})
    df_val_file_cat = pd.merge(val_img_df, val_anns_df, on='image_id')
    df_val_file_cat['category_id'] = df_val_file_cat['category_id'].astype(str)
    # print(df_train_file_cat.head())
    df_train_file_cat.sample(frac=1)
    train_datagen=ImageDataGenerator(rescale=1./255,
        horizontal_flip = True,
        vertical_flip =True,
        rotation_range=10,
        zoom_range = 0.3,
        width_shift_range = 0.3,
        height_shift_range=0.3,
        preprocessing_function = None
        )
    val_datagen=ImageDataGenerator(rescale=1./255,
        preprocessing_function = None
        )
    train_generator=train_datagen.flow_from_dataframe(
        dataframe=df_train_file_cat,
        directory=image_folder_path,
        x_col="file_name",
        y_col="category_id",
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",
        target_size=(img_size,img_size))

    val_generator=val_datagen.flow_from_dataframe(
        dataframe=df_val_file_cat,
        directory=val_folder_path,
        x_col="file_name",
        y_col="category_id",
        batch_size=batch_size,
        shuffle=False,
        class_mode="categorical",
        target_size=(img_size,img_size))
    return train_generator,val_generator
# print(train_anns)

# train_generator = data_generator("/root/kaggle/train_mini.json","/root/kaggle",256,64)

# print(next(iter(train_generator)))