import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
import csv
import json

from matplotlib import pyplot as plt
import numpy as np

class PretrainDataset_Contact(Dataset):
    def __init__(self, mode='train'):
        self.datalist = []

        # 定义数据集根目录和csv文件路径
        datasets = [  
            # {  
            #     'root_dir': '/data/tactile_datasets/S1_dataset/Tacquad/',  
            #     'csv_file': '/data/tactile_datasets/S1_dataset/Tacquad/Tacquad.csv'  
            # },  
            # {  
            #     'root_dir': '/data/tactile_datasets/S1_dataset/TAG/',  
            #     'csv_file': '/data/tactile_datasets/S1_dataset/TAG/TAG.csv'  
            # }  
            {  
                'root_dir': '/data/tactile_datasets/S1_dataset/Tacquad/',  
                'csv_file': '/home/shipeng/PhysioTouch/Tacquad.csv'  
            }
        ]  

        # 同一加载所有数据集
        for dataset_info in datasets:
            root_dir = dataset_info['root_dir']  
            csv_file = dataset_info['csv_file'] 

            if not os.path.exists(csv_file):
                print(f"警告：CSV文件不存在：{csv_file}")
                continue
            with open(csv_file, 'r', encoding='utf-8-sig') as file:  
                csv_reader = csv.reader(file)  

                for row in csv_reader:  
                    folder_name = row[0]  
                    start_frame = int(row[1])  
                    end_frame = int(row[2])  

                    # 从第4帧开始(因为需要前3帧作为历史)  
                    for t in range(start_frame + 3, end_frame + 1):                            
                        # # 构建完整路径  
                        png_path_0 = root_dir + folder_name  + '/' + str(t-3) + '.png' 
                        png_path_1 = root_dir + folder_name + '/' + str(t-2) + '.png'  
                        png_path_2 = root_dir + folder_name + '/' + str(t-1) + '.png'
                        png_path_3 = root_dir + folder_name + '/' + str(t) + '.png' 
                          
                        # 验证文件存在  
                        if os.path.exists(png_path_0) and os.path.exists(png_path_3):  
                            self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])  
        print(f"Total samples loaded: {len(self.datalist)}")


        if mode == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):  
        img0 = Image.open(self.datalist[index][0]).convert('RGB')  
        img1 = Image.open(self.datalist[index][1]).convert('RGB')  
        img2 = Image.open(self.datalist[index][2]).convert('RGB')  
        img3 = Image.open(self.datalist[index][3]).convert('RGB')  
    
        img0 = self.to_tensor(img0).unsqueeze(0)  
        img1 = self.to_tensor(img1).unsqueeze(0)  
        img2 = self.to_tensor(img2).unsqueeze(0)  
        img3 = self.to_tensor(img3).unsqueeze(0)  
        img = torch.cat([img0, img1, img2, img3])  
        img = self.transform(img)  
        return img
