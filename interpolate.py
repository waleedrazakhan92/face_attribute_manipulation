# from load_and_preprocess_utils import *
# from face_alignment_utils import *
# from face_recognition_utils import *
from hfgi_utils import interpolate_faces, load_hfgi_model, create_video
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
    parser.add_argument("--path_img_1", help="Path to first image")
    parser.add_argument("--path_img_2", help="Path to second image")
    parser.add_argument("--steps", help="Path to new person to add", default = 20)
    parser.add_argument("--path_results", help="Path to save results", default = 'results_and_data/')
    parser.add_argument("--vid_name", help="Name of interpolation video", default='interpolation_video.avi')
    parser.add_argument("--save_aligned_imgs", help="Save aligned and cropped images", action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.path_results):
        os.mkdir(args.path_results)

    img_1 = PIL.Image.open(args.path_img_1)
    img_2 = PIL.Image.open(args.path_img_2)

    interpolated_faces, img_1_aligned, img_2_aligned, img_1_projected, img_2_projected = interpolate_faces(img_1, img_2, net, predictor, detector, steps=int(args.steps))
    
    if args.save_aligned_imgs==True:
        print('Saving images to', args.path_results)
        name_1 = args.path_img_1.split('/')[-1].split('.')[0]
        name_2 = args.path_img_2.split('/')[-1].split('.')[0]
        img_1_aligned.save(os.path.join(args.path_results,name_1+'_aligned.jpg'))
        img_2_aligned.save(os.path.join(args.path_results,name_2+'_aligned.jpg'))
        img_1_projected.save(os.path.join(args.path_results,name_1+'_projected.jpg'))
        img_2_projected.save(os.path.join(args.path_results,name_2+'_projected.jpg'))


    print('Saving video to',os.path.join(args.path_results,args.vid_name))
    create_video(interpolated_faces,os.path.join(args.path_results,args.vid_name))
