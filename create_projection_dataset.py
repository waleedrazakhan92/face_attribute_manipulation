from hfgi_utils import project_and_save_latent, load_hfgi_model
import dlib
import sys
import PIL
import argparse
import os

if __name__ == "__main__":
    # Load the models
    predictor = dlib.shape_predictor("HFGI/checkpoints/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    # face_rec_model = dlib.face_recognition_model_v1('HFGI/checkpoints/dlib_face_recognition_resnet_model_v1.dat')
    net = load_hfgi_model('HFGI/checkpoints/ckpt.pt')

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dataset", help="Path to images folder to project")
    parser.add_argument("--path_projections", help="Path to save projections", default='image_projections/')
    parser.add_argument("--save_latents", help="Save latents or not", action='store_true')
    parser.add_argument("--save_proj_imgs", help="Save projected images or not", action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.path_projections):
        os.mkdir(args.path_projections)

    project_and_save_latent(args.path_dataset, args.path_projections, net, predictor, detector, args.save_latents, args.save_proj_imgs)