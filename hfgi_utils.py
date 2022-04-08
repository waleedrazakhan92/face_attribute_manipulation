import sys
sys.path.insert(1, 'HFGI/')

from face_alignment_utils import align_face
from load_and_preprocess_utils import load_img
from argparse import Namespace
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms

from multiprocessing import Pool, Process, Manager
from datetime import datetime, timedelta
from utils.common import tensor2im
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.
from tqdm import tqdm
from glob import glob
import os

def split_name(img_name):
    return img_name.split('/')[-1].split('.')[0]

def split_latent(img_name):
    return img_name.split('/')[-1].split('_latent')[0]

def create_new_latent_list(path_images, path_latents):
    name_imgs = glob(os.path.join(path_images,'*'))
    name_latents = glob(os.path.join(path_latents,'*.npy'))

    name_imgs_split = list(map(split_name, name_imgs))
    name_latents_split = list(map(split_latent, name_latents))

    print('Total images',len(name_imgs))
    print('Total latents',len(name_latents))
    idx_list = []
    for i in range(0,len(name_latents_split)):
        idx_list.append(name_imgs_split.index(name_latents_split[i]))


    new_path_list = list(np.delete(np.array(name_imgs),idx_list))
    print('Latents remaining',len(new_path_list))

    return new_path_list

tranform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])                                     


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes

def load_hfgi_model(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['is_train'] = False
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    return net


def load_lat_np(file):
    lat = np.load(file)
    return lat

def load_lat_multi(path_latents):
    pool = Pool(8)
    all_latents = pool.map(load_lat_np, path_latents)
    pool.close()
    pool.join()
    return all_latents

"""
The function takes a preprocessed aligend and transcfomed image tensor and convert it into its latent vector
Using the pretrained HFGI model
Then returns an image along with its latent vector
"""
def img_2_latent_proj(transformed_img, net):
    with torch.no_grad():
        x = transformed_img.unsqueeze(0).cuda()

        # tic = time.time()
        latent_codes = get_latents(net, x)
        
        # calculate the distortion map
        imgs, _ = net.decoder([latent_codes[0].unsqueeze(0).cuda()],None, input_is_latent=True, randomize_noise=False, return_latents=True)
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

        # ADA
        img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)

        result_image, lat = net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        # toc = time.time()
        # print('Inference took {:.4f} seconds.'.format(toc - tic))
    
    return tensor2im(result_image[0]), latent_codes

def latent_2_img_move(transformed_img, net, latent_move):
    with torch.no_grad():
        x = transformed_img.unsqueeze(0).cuda()

        # tic = time.time()
        latent_codes = get_latents(net, x)
        # ------------------------------------------
        latent_codes = latent_move
        # latent_codes[:8] = (latent_codes + torch.tensor(intensity*direction).cuda())[:8]
        # ------------------------------------------
        # calculate the distortion map
        imgs, _ = net.decoder([latent_codes[0].unsqueeze(0).cuda()],None, input_is_latent=True, randomize_noise=False, return_latents=True)
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

        # ADA
        img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)

        # ------------------------------------------
        # latent_codes[:8] = (latent_codes + torch.tensor(intensity*direction).cuda())[:8]
        # ------------------------------------------

        result_image, lat = net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        # toc = time.time()
        # print('Inference took {:.4f} seconds.'.format(toc - tic))
    
    return tensor2im(result_image[0])#, latent_codes

def process_with_hfgi(img, net, predictor, detector):
    # #img = img.resize((256,256))
    aligned_img = align_face(img, predictor, detector)
    transformed_img = tranform_img(aligned_img)
    projected_img, latent_vec = img_2_latent_proj(transformed_img, net)

    return aligned_img, transformed_img, projected_img, latent_vec

def interpolate_faces(img, nearest_img, net, predictor, detector, steps=10):
    aligned_img, transformed_img, projected_img, latent_vec = process_with_hfgi(img, net, predictor, detector)
    aligned_img_nearest, transformed_img_nearest, projected_img_nearest, latent_vec_nearest = process_with_hfgi(nearest_img, net, predictor, detector)
    
    interpolated_vectors = np.linspace(latent_vec.cpu(), latent_vec_nearest.cpu(), steps)

    moved_imgs_all = []
    moved_imgs_all.append(projected_img)
    moved_img = latent_2_img_move(tranform_img(projected_img), net, torch.tensor(interpolated_vectors[0]).cuda())

    for i in tqdm(range(1,len(interpolated_vectors))):
        moved_img = latent_2_img_move(tranform_img(moved_img), net, torch.tensor(interpolated_vectors[i]).cuda())
        moved_imgs_all.append(moved_img)

    moved_imgs_all.append(projected_img_nearest)
    return moved_imgs_all, aligned_img, aligned_img_nearest, projected_img, projected_img_nearest

def project_and_save_latent(path_images_data, path_projections, net, predictor, detector, save_latents=True, save_projections=True):
    
    if not os.path.isdir(path_projections):
        os.mkdir(path_projections)

    path_images = create_new_latent_list(path_images_data, path_projections)
    # all_paths = glob(path_images+'*')
    for i in tqdm(range(0,len(path_images))):
        # img = load_img(all_paths[i])
        img = load_img(path_images[i])
        img_name = path_images[i].split('/')[-1]
        # -------------------------------------------
        # img = img.resize((256,256))
        # -------------------------------------------
        try:
            aligned_img = align_face(img, predictor, detector)
            transformed_image = tranform_img(aligned_img)
            # img_name = all_paths[i].split('/')[-1]
            result_img, result_latent = img_2_latent_proj(transformed_image, net)
            
            if save_projections==True:
                # save_size = result_img.size
                # combined_img = PIL.Image.fromarray(np.hstack((aligned_img.resize(save_size),result_img)))
                # combined_img.save(path_projections+img_name)
                result_img.save(path_projections+img_name)
            
            if save_latents==True:
                np.save(f"{path_projections+img_name.split('.')[0]}_latent.npy", result_latent.cpu())
        except:
            print('Problem with {}. Skipping the image!!!'.format(path_images[i]))


def img_2_latent_proj_change(transformed_img, net, direction, intensity):
    with torch.no_grad():
        x = transformed_img.unsqueeze(0).cuda()

        # tic = time.time()
        latent_codes = get_latents(net, x)
        # ------------------------------------------
        # latent_codes[:8] = (latent_codes + torch.tensor(intensity*direction).cuda())[:8]
        # ------------------------------------------
        # calculate the distortion map
        imgs, _ = net.decoder([latent_codes[0].unsqueeze(0).cuda()],None, input_is_latent=True, randomize_noise=False, return_latents=True)
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

        # ADA
        img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)

        # ------------------------------------------
        latent_codes[:8] = (latent_codes + torch.tensor(intensity*direction).cuda())[:8]
        # ------------------------------------------

        result_image, lat = net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        # toc = time.time()
        # print('Inference took {:.4f} seconds.'.format(toc - tic))
    return tensor2im(result_image[0])#, latent_codes
    
def create_video(img_array, vid_path='interp_video.avi'):
    import cv2
    size = np.shape(img_array[0])[:2]
    out = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    
    for i in range(len(img_array)):
        out.write(cv2.cvtColor(np.array(img_array[i]),cv2.COLOR_RGB2BGR))
    out.release()
