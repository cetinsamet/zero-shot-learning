import numpy as np
np.random.seed(123)
import pickle

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.utils import to_categorical


WORD2VECPATH    = "../data/class_vectors.npy"
DATAPATH        = "../data/zeroshot_data.pkl"

def load_data():
    """read data, create datasets"""
    # READ DATA
    with open(DATAPATH, 'rb') as infile:
        data = pickle.load(infile)

    # SHUFFLE DATA
    np.random.shuffle(data)
    data_classnames, data_features = map(list, zip(*data))

    # ONE-HOT-ENCODE DATA
    label_encoder = LabelEncoder()
    label_encoder.fit(data_classnames)

    ### SPLIT DATA FOR TRAINING
    train_val_data = [(data_features[i], data_classnames[i]) for i in range(len(data)) if data[i][0] in train_classes]
    train_size = 300
    train_data, valid_data = list(), list()
    for class_ in train_classes:
        train_ct = 0

        for data_ in train_val_data:
            if data_[1] == class_:
                if train_ct < train_size:
                    train_data.append(data_)
                    train_ct+=1
                    continue
                valid_data.append(data_)

    # SHUFFLE TRAINING AND VALIDATION DATA
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)
    train_data = [(data[0], to_categorical(label_encoder.transform([data[1]]), num_classes=20))for data in train_data]
    valid_data = [(data[0], to_categorical(label_encoder.transform([data[1]]), num_classes=20)) for data in valid_data]

    # FORM X_TRAIN AND Y_TRAIN
    x_train, y_train    = zip(*train_data)
    x_train, y_train    = np.squeeze(np.asarray(x_train)), np.squeeze(np.asarray(y_train))

    # FORM X_VALID AND Y_VALID
    x_valid, y_valid = zip(*valid_data)
    x_valid, y_valid = np.squeeze(np.asarray(x_valid)), np.squeeze(np.asarray(y_valid))


    ### SPLIT DATA FOR ZERO-SHOT
    zsl_data = [(data_features[i], data_classnames[i]) for i in range(len(data)) if data[i][0] in zsl_classes]
    # SHUFFLE ZERO-SHOT DATA
    np.random.shuffle(zsl_data)
    zsl_data = [(data[0], to_categorical(label_encoder.transform([data[1]]), num_classes=20)) for data in zsl_data]

    # FORM X_ZSL AND Y_ZSL
    x_zsl, y_zsl = zip(*zsl_data)
    x_zsl, y_zsl = np.squeeze(np.asarray(x_zsl)), np.squeeze(np.asarray(y_zsl))

    return (x_train, x_valid, x_zsl), (y_train, y_valid, y_zsl)

def custom_kernel_init(shape):
    class_vectors       = sorted(np.load(WORD2VECPATH), key=lambda x:x[0])
    classnames, vectors = zip(*class_vectors)
    vectors             = np.asarray(vectors, dtype=np.float)
    vectors             = vectors.T
    return vectors

def  build_model():
    model = Sequential()
    model.add(Dense(4096, input_shape=(4096,), activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.7))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.3))
    #model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(NUM_ATTR, activation='relu'))
    model.add(Dense(NUM_CLASS, activation='softmax', trainable=False, kernel_initializer=custom_kernel_init))

    print("-> model building is completed.")
    return model


def train_model(model, train_data, valid_data):
    x_train, y_train = train_data
    x_valid, y_valid = valid_data

    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics   = ['categorical_accuracy', 'top_k_categorical_accuracy'])

    history = model.fit(x_train, y_train,
                        validation_data = (x_valid, y_valid),
                        verbose         = 2,
                        epochs          = EPOCH,
                        batch_size      = BATCH_SIZE,
                        shuffle         = True)

    print("model training is completed.")
    return history

def main():
    global train_classes
    with open('train_classes.txt', 'r') as infile:
        train_classes = [str.strip(line) for line in infile]

    global zsl_classes
    with open('zsl_classes.txt', 'r') as infile:
        zsl_classes = [str.strip(line) for line in infile]

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # SET HYPERPARAMETERS

    global NUM_CLASS, NUM_ATTR, EPOCH, BATCH_SIZE
    NUM_CLASS   = 20
    NUM_ATTR    = 300
    BATCH_SIZE  = 32
    EPOCH       = 20

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # TRAINING PHASE
    (x_train, x_valid, x_zsl), (y_train, y_valid, y_zsl) = load_data()
    #model = build_model()
    #train_model(model, (x_train, y_train), (x_valid, y_valid))


if __name__ == '__main__':
    main()
