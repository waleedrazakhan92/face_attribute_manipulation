from PIL import Image
import numpy as np
def load_img(img_path, resize_dims=None):
    original_image = Image.open(img_path)
    original_image = original_image.convert("RGB")
    if resize_dims !=None:
        original_image = original_image.resize(resize_dims)

    return original_image

def preprocess_img(img, transforms):
    transformed_image = transforms(input_image)
    return transformed_image

def display_images(*images, img_size=(256,256)):
    res = np.array(images[0].resize(img_size))
    for i in range(1,len(images)):
        res = np.concatenate((res, np.array(images[i].resize(img_size))), axis=1)
        
    return Image.fromarray(res) 

def display_images_2(images, img_size=(256,256)):
    res = np.array(images[0].resize(img_size))
    for i in range(1,len(images)):
        res = np.concatenate((res, np.array(images[i].resize(img_size))), axis=0)
        
    return Image.fromarray(res)

def read_matched_images(matched_img_paths):
    matched_imgs = []
    for i in range(0,len(matched_img_paths)):
        matched_imgs.append(load_img(matched_img_paths[i]))

    return matched_imgs