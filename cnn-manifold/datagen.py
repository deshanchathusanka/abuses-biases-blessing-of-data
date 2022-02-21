# DO NOT CHANGE - Press to run
def gen_data(df, model, FILE_PATH, Image_Size):
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.image import ImageDataGenerator,load_img


    earlystop = EarlyStopping(patience = 10)
    learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.00001)
    callbacks = [earlystop,learning_rate_reduction]

    # DO NOT CHANGE
    df["category"] = df["category"].replace({0:'cat',1:'dog'})
    train_df,validate_df = train_test_split(df,test_size=0.2,
      random_state=42)

    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    total_train=train_df.shape[0]
    total_validate=validate_df.shape[0]
    batch_size=16


    #DO NOT CHANGE
    train_datagen = ImageDataGenerator(rotation_range=15,
                                    rescale=1./255,
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1
                                    )

    train_generator = train_datagen.flow_from_dataframe(train_df,
                                                     FILE_PATH,x_col='filename',y_col='category',
                                                     target_size=Image_Size,
                                                     class_mode='categorical',
                                                     batch_size=batch_size)

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        FILE_PATH, 
        x_col='filename',
        y_col='category',
        target_size=Image_Size,
        class_mode='categorical',
        batch_size=batch_size
    )

    test_datagen = ImageDataGenerator(rotation_range=15,
                                    rescale=1./255,
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1)

    test_generator = train_datagen.flow_from_dataframe(train_df,
                                                     FILE_PATH,x_col='filename',y_col='category',
                                                     target_size=Image_Size,
                                                     class_mode='categorical',
                                                     batch_size=batch_size)
                                                     
    return validation_generator, test_generator, train_generator, total_validate, batch_size, total_train, callbacks, learning_rate_reduction, earlystop