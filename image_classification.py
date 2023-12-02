from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class ImageClassifier:

    img_width = 350
    img_height = 350
    num_label = 0
  

    def __init__(self, dataset):
      self.readDataset(dataset)
  
      self.num_label = len(self.data.columns[2:])
      self.labels = self.data.columns[2:]
      print(f"Image labels: {self.labels}")
      X = self.getImageData()
      y = self.data.drop(['Id', 'Genre'], axis = 1)
      y = y.to_numpy()
      X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size = 0.1)
      self.model = self.constructNN(X_train[0].shape)
      self.train(self.model, X_train, y_train, X_test, y_test)

    def readDataset(self, path):
        data = pd.read_csv(path)
        self.data = data.drop(data.index[700:], axis=0)
        print("Data imported successfully")

    def processImg(self, path):
        img = image.load_img(path, target_size=(self.img_width, self.img_height, 3))
        img = image.img_to_array(img)
        img = img/255.0
        return img
    
    def getImageData(self):
        X = []
        for i in tqdm(range(self.data.shape[0])):
            path = './Images/'+self.data['Id'][i] + ".jpg"
            img = self.processImg(path)
            X.append(img)

        X = np.array(X)
        return X
        
    def constructNN(self, input_shape):
        ## Nerual Network
        model = Sequential()
        # input - conv - BN - ReLu - Max Pool
        # 1st layer
        model.add(Conv2D(16, (3,3), activation='relu', input_shape = input_shape ))
        model.add(BatchNormalization())
        model.add(MaxPool2D(2,2))
        model.add(Dropout(0.1))
        # 2nd layer
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(2,2))
        model.add(Dropout(0.2))
        # 3rd layer
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(2,2))
        model.add(Dropout(0.3))
        # 4th layer
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(2,2))
        model.add(Dropout(0.4))

        model.add(Flatten())
        # fully connected layers
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.50))

        # output layer -> sigmoid for multilabel
        model.add(Dense(self.num_label, activation='sigmoid'))
        # model.summary()
        return model

    def train(self, model, X_train, y_train, X_test, y_test):
        print("Training model...")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    def predict(self, img_path):
        img = self.processImg(img_path)
        img = img.reshape(1, self.img_width, self.img_height, 3)
        y_prob = self.model.predict(img)
        predictions = np.argsort(y_prob[0])[:-4:-1]
        predictions = [ f"{self.labels[predictions[i]]}: {y_prob[0][predictions[i]]:.2}"  for i in range(3) ]
        print("\n" + "#"*90 + "\n")
        print(f"Prediction Result: {predictions}")
        print("\n"+ "#"*90 + "\n")

def main():
  image_classifier = ImageClassifier('./train.csv')
  path = './fast.jpg'
  image_classifier.predict(path)

if __name__ == "__main__":
  main()