import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_fer2013(csv_file):
    df = pd.read_csv(csv_file)
    
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].values
    
    X = []
    for pixel_sequence in pixels:
        image = np.array(pixel_sequence.split(), dtype='float32')
        image = image.reshape(48, 48, 1)
        X.append(image)
    
    X = np.array(X, dtype='float32')
    X = X / 255.0  
    
    y = np.array(emotions, dtype='int32')
    
    return X, y, df



def split_data(X, y, df):
    train_mask = df["Usage"] == "Training"
    val_mask = df["Usage"] == "PublicTest"
    test_mask = df["Usage"] == "PrivateTest"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    csv_path = "data/fer2013.csv"

    X, y, df = load_fer2013(csv_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, df)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    print("Pixel range:", X.min(), "to", X.max())
