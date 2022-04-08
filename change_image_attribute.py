from hfgi_utils import load_hfgi_model, tranform_img, img_2_latent_proj, img_2_latent_proj_change
from load_and_preprocess_utils import load_img
from face_alignment_utils import align_face
import dlib
import argparse
import joblib
import numpy as np
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_img", help="Path to input image")
    parser.add_argument("--path_attribute_model", help="path to trained classifier")
    parser.add_argument("--change_intensity", help="path to trained classifier", default=1.5, type=float)
    parser.add_argument("--save_combined_img", help="save conbined image for comparison", action='store_true')
    args = parser.parse_args()

    # Load the models
    predictor = dlib.shape_predictor("HFGI/checkpoints/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    # face_rec_model = dlib.face_recognition_model_v1('HFGI/checkpoints/dlib_face_recognition_resnet_model_v1.dat')
    net = load_hfgi_model('HFGI/checkpoints/ckpt.pt')

    # load classifier
    clf = joblib.load(args.path_attribute_model)
    attribute_direction = clf.coef_.reshape((18, 512))

    img = load_img(args.path_img)
    aligned_img = align_face(img, predictor, detector)

    transformed_img = tranform_img(aligned_img)
    projected_img,_ = img_2_latent_proj(transformed_img, net)

    manipulated_img = img_2_latent_proj_change(transformed_img, net, attribute_direction, args.change_intensity)
    print('Saving image to:',args.path_img.split('/')[-1].split('.')[0]+'_manipulated.png')
    manipulated_img.save(args.path_img.split('/')[-1].split('.')[0]+'_manipulated.png')
    
    if args.save_combined_img==True:
        combined = np.concatenate((np.array(img.resize((1024,1024))),np.array(aligned_img.resize((1024,1024))) ,np.array(projected_img), np.array(manipulated_img)), axis=1)

        combined = Image.fromarray(combined)
        combined.save(args.path_img.split('/')[-1].split('.')[0]+'_combined.png')
