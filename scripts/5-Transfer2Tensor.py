import os 
import cv2 
import torch

def normalize_image(image):

    return (image - image.min())/(image.max()-image.min())

folder_paths = [x[0] for x in list(os.walk('/home/obanmarcos/Balseiro/DeepOPT/datasets/x10')) if 'red' in x[0]]
imgs_jpg_paths = []
imgs_torch_paths = []

for folder_path in folder_paths:

    imgs_jpg_paths += [folder_path+'/'+x for x in os.listdir(folder_path)]

for x in imgs_jpg_paths:

    imgs_torch_paths.append(x.replace('jpg', 'pt'))

for str_img, str_torch in zip(imgs_jpg_paths, imgs_torch_paths):

    print('written ', str_torch)
    img = cv2.imread(str_img, cv2.IMREAD_GRAYSCALE)

    img_tensor = torch.FloatTensor(img).unsqueeze(0)
    
    torch.save(img_tensor, str_torch)
