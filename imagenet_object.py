import os
import zipfile
import pandas as pd

source = "/home/claudio/Documentos/imagenet-object-localization-challenge.zip"
destination = "/home/claudio/Documentos/ImageNet_objects/"

imagenet_classes_file = "imagenet classes.csv"

df = pd.read_csv(imagenet_classes_file)
names = df['folder_name'].tolist()
# print(df['class'].sort_values(ascending=False))
# exit()
zip_folders = ["ILSVRC/Data/CLS-LOC/train/", "ILSVRC/Data/CLS-LOC/test/"]

files = []
for zip_folder in zip_folders:
    for i in range(len(names)):
        files.append(zip_folder + names[i])


with zipfile.ZipFile(source) as archive:
    i = 0
    for f in archive.namelist():
        for f2 in files:
            if f2 in f:
                archive.extract(f, destination)