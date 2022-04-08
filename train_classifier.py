from hfgi_utils import load_lat_multi
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import joblib 
import pickle
import argparse
import os
from glob import glob
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_positive_latents", help="Path to positive attribute latents")
    parser.add_argument("--path_negative_latents", help="Path to negative attribute latents")
    parser.add_argument("--save_model_dir", help="directory to save trained classifier", default='trained_classiifier/')
    parser.add_argument("--save_model_name", help="trained_classifier_name",default="attribute.joblib")
    parser.add_argument("--max_iters", help="maximum number of training iterations", default=10000, type=int)
    args = parser.parse_args()

    if not os.path.isdir(args.save_model_dir):
        os.mkdir(args.save_model_dir)

    pos_paths = glob(args.path_positive_latents+'*.npy')
    neg_paths = glob(args.path_negative_latents+'*.npy')

    pos_latents = load_lat_multi(pos_paths)
    neg_latents = load_lat_multi(neg_paths)

    pos_labels = np.ones(len(pos_latents))
    neg_labels = np.zeros(len(neg_latents))

    dlatent_data = np.vstack((pos_latents, neg_latents)) 
    all_labels = np.hstack((pos_labels, neg_labels))

    dlatent_data = dlatent_data.reshape((-1, 18*512))

    print('Number of latents : ',len(dlatent_data))
    print('Number of labels : ',len(all_labels))

    clf = LogisticRegression(class_weight=None, max_iter=args.max_iters)
    clf.fit(dlatent_data.reshape((-1, 18*512)), all_labels)
    attribute_direction = clf.coef_.reshape((18, 512))
    print('classifier accuracy',clf.score(dlatent_data, all_labels))

    filename = os.path.join(args.save_model_dir,args.save_model_name)
    pickle.dump(clf, open(filename, 'wb'))
    joblib.dump(clf, filename)


if __name__ == "__main__":
    main()
