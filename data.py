
# Import required libraries 
import os
import torch 
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 
from torchvision.transforms.functional import crop
import pickle
from IPython.display import display

# Read the image 
def image_to_tensor(image_path, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     #convert
    tensor = transform(image)
    return tensor

def crop_char(image):
    return crop(image, 150, 200, 150, 250)

transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Lambda(crop_char)
]) 

image_dir = "/Users/mdo/Desktop/marioframe/characters/"      # Change File Per

labels = []
tensors = []
char_num = {'Mario': 0, 'Baby Daisy': 1, 'Baby Luigi': 2, 'Baby Mario': 3, 'Baby Peach': 4, 
            'Baby Rosalina': 5, 'Bowser': 6,'Bowser Jr': 7, 'Daisy': 8, 'Diddy Kong': 9, 'Donkey Kong': 10, 
            'Dry Bones': 11, 'Dry Bowser': 12, 'Funky Kong': 13,'Isabelle': 14, 'King Boo': 15, 
            'Koopa Troopa': 16, 'Link': 17, 'Luigi': 18, 'Peach': 19, 'Rosalina': 20,
            'Shy Guy': 21, 'Toad': 22, 'Toadette': 23, 'Waluigi': 24, 'Wario': 25, 'Yoshi': 26,}
data_tensor = {'tensors': [], 'labels': []} 
df = pd.DataFrame(data_tensor)

for folder in os.listdir(image_dir):
    for filename in os.listdir(image_dir + '/' + folder):
        if filename.endswith(".png"): 
            image_path = os.path.join(image_dir, folder, filename)
            print(image_path)
            name = os.path.basename(os.path.normpath(image_dir))
            labelNum = char_num[folder]
            tensor = image_to_tensor(image_path, transform)
            df = df._append({'tensors': tensor, 'labels': labelNum}, ignore_index=True)
            display(df)
display(df)
        #print(tensor)
        #tensors.append(tensor)
        #labels.append(26)       #change label per file



#print(tensors)

for name in char_num.keys():
    with open(name + str(char_num[name]) + '.pickle', 'wb') as fh:      #change Per label
        pickle.dump({'tensors': tensors, 'labels' : labels}, fh, protocol=pickle.HIGHEST_PROTOCOL)

print("file saved successfully.")
