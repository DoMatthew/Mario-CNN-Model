import pickle
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import torch 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

chars = ['Mario0.pickle', 'BabyDaisy1.pickle', 'BabyLuigi2.pickle', 'BabyMario3.pickle', 'BabyPeach4.pickle', 'BabyRosalina5.pickle', 'Bowser6.pickle', 'BowserJr7.pickle', 
          'Daisy8.pickle', 'DiddyKong9.pickle', 'DonkeyKong10.pickle']
chars2 =  ['Mario0.pickle', 'BabyDaisy1.pickle']
images = []
label = []
image_dir = "/Users/mdo/Desktop/marioframe/rgbData/"   
for folder in chars:  #os.listdir(image_dir)
    print('start processing', folder)
    with open(image_dir + '/' + folder, 'rb') as handle:
        b = pickle.load(handle)
        #print(len(b['tensors']))
        #print(b['tensors'][0].shape)
        images += b['tensors'][:10]
        label += (b['labels'][:10])

print("load df")
df = pd.DataFrame.from_dict({"tensors": images, "labels": label})

#print(df.shape)
#print(df.loc[0]['tensors'])
#df.to_csv('df.csv')
df = df.sample(frac=1)

train_set, test_set = train_test_split(df,test_size=0.3) 
#print('train', train_set.shape)
#train_set.to_csv('training.csv')
#test_set.to_csv('testing.csv')
print('finished train/test split')
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(6300, 128) 
        #maybe another fc layer
        self.fc2 = nn.Linear(128, 11)
        self.Softmax = nn.Softmax(dim=0)

    def forward(self, x):
        
        # Convolution 1
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # Convolution 2 
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Convolution 3
        x = self.cnn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = torch.flatten(x, start_dim=0)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.Softmax(x)
        return output
    
    def one_hot_encode(self, index):
        one_hot = torch.zeros(11)
        one_hot[index] = 1
        return one_hot 

    def train(self, epochs, x):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for e in range(epochs):
            for i in x.index:
                tens = x.loc[i]['tensors']
                optimizer.zero_grad()
                prediction = self.forward(tens)
                y = self.one_hot_encode(x.loc[i]['labels'])
                loss_val = criterion(prediction, y)
                loss_val.backward()
                optimizer.step()
                print(loss_val.item(), e, 'label is:', x.loc[i]['labels'], 'prediction is:', torch.argmax(prediction))
        return loss_val

  
model = CNN()
out = model.train(10, train_set)
print(f'output: {out}')

correct = 0
total = 0
with torch.no_grad():
    for i in test_set.index:
        out = model.forward(test_set.loc[i]['tensors'])
        prediction = torch.argmax(out)
        total += 1 
        if prediction == test_set.loc[i]['labels']:
            correct += 1
accuracy = 100 * correct / total
print('# correct:', correct, 'total is:', total, 'accuary is:', accuracy)





#print("My tensor : ", tensor)






    # with open('tensorCSV.csv', 'a') as csv_file:  
    #     writer = csv.writer(csv_file)
    #     for i in range(len(b["tensors"])):
    #         writer.writerow((index, b["tensors"][i].numpy().tolist(), b["labels"][i]))

