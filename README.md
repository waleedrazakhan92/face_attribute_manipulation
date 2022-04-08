# Face_attribute_manipulation
We know that generative models generate images from a latent space and we can control semantic changes in a image using that latent space. But what if we want to make changes in a real image? For us to be able to make changes in a real life image, we need to first have the latent representation of that image. The official stylegan repository, https://github.com/NVlabs/stylegan2-ada, has a code to convert an input image into it's latent space 'projector.py'. What it does is it takes generates a random latent vector then generates an image out of it and then compare that image with the real image to calculate the loss. Then it backpropagates that error to make changes in the latent vector. Iteratively like this, it finds a latent represntation of that image. But its a slow process.

For this project we a using the HFGI: High-Fidelity GAN Inversion for Image Attribute Editing repository https://github.com/Tengfei-Wang/HFGI. What it does is, it can canculate the embeddings of the input image in one forward pass with a very reasonable accuracy. Which makes it practical for image editing tools. You can read further details about the architecture in the paper https://arxiv.org/abs/2109.06590.

![](<repository_images/HFGI.jpg>)

## Requirements


## Downloading the resources
You can find all the models and files used in this repository here. https://drive.google.com/drive/folders/1puK3hPxDWfHCJ0fbxFAMx2tOfSSDBJhK?usp=sharing. All the resources are also shared on the bucket as well. In case the **gdown** command doesn't work, you can download models and files from either the drive or the bucket, and place them accordingly in their right spots.  

```
# install ninja
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```
```
# Clone the repository:
git clone https://github.com/waleedrazakhan92/face_attribute_manipulation.git
cd face_attribute_manipulation/
```
Create a folder **checkpoints** in **HFGI** directory. And place **ckpt.pt**, **shape_predictor_68_face_landmarks.dat** in the checkpoints folder.

```
# place all the checkpoint files in the directory in HFGI/checkpoints/ 
# you can either place them manually or download them directely in the folder
cd HFGI/
mkdir checkpoints/

# download the pretrained HFGI model checkpoint
# ckpt.pt
# you can also find this model in https://github.com/Tengfei-Wang/HFGI
gdown --id 19y6pxOiJWB0NoG3fAZO9Eab66zkN9XIL 

# download the dlib landmarks detection model
# shape_predictor_68_face_landmarks.dat
wget https://raw.github.com/ageitgey/face_recognition_models/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat

cd ../../
```
## Download the data 
For experimentation, a dataset of handpicked images of different celebrities is also given. You can find this data in https://drive.google.com/drive/folders/11tjrev0KC7obWyFAp4712Ewnk52S7Ynr?usp=sharing. There are two zip files **face_recognition_dataset.zip** and **face_recognition_dataset_2.zip**

```
# download face recognition data
gdown --id 11BrTyAP2ldqxnXXj2dO3V9jaC_eBM96u
unzip -q face_recognition_dataset.zip

gdown --id 1eR0JGP0tWorRV0owU_ORdTNj5cTq3Tjv
unzip -q face_recognition_dataset_2.zip
```
## Playing with the latent space
An input image is first cropped and aligned in the preprocessing step to prepare it as an input to the HFGI model. The HFGI model then converts that image into its latent representation. Once we have this latent representation, we can manipulate its vector by moving it in any direction we want. Now this manipulated vector is then put into the decoder to generated the resulant image.
![](<repository_images/pipeline_2.png>)

You can also interpolate between latent vectors of two different images to make smooth transition videos between two faces. Increasing the number of steps samples more latent points between two latent vectors and makes the transition smooth.
![](<repository_images/interp.gif>)

Run the following command to make interpolation video between two faces:
```
python3 interpolate.py --path_img_1 /path/to/img --path_img_2 /path/to/img  --steps 200 --vid_name 'interpolation_video.avi' --save_aligned_imgs 
```
## Attribute classification
You can also train a classifier on a desired attribute conversion(male to female or female to male etc.). Once trained, this classifier can be used to manipulate an image in one direction or the other. To train this classifier you first need to have latent vectors of the desired images arranged accordingly. For example you can put images of positive attribute images and negative attribute images (males and females) in separate folders and then run the script ```create_projection_database.py``` file on both the folders to have latent vectors for positive and negatives separately. 
```
python3 create_projection_dataset.py --path_dataset 'path/to/positive images/folder/' --path_projections 'positive_projections/' --save_latents 
python3 create_projection_dataset.py --path_dataset 'path/to/negative images/folder/' --path_projections 'negative_projections/' --save_latents 

```
Once you have both sets of latent vectors, you can train the classifier using ```train_classifier.py``` file. Running this code will save the classifier in a *.joblib* format. 
```
python3 train_classifier.py --path_positive_latents '/path/to/positive_latents/' --path_negative_latents '/path/to/negative_latents/'
```
Then you can change the attribute of an image using the trained classifier using ```change_image_attribute.py```. You can set the intensity of the change in either positive or negative direction. Ideally the value can lie between +5 and -5. But that is subjective to each attribute.
```
python3 change_image_attribute.py --path_img '/path/to/img.png' --path_attribute_model 'trained_classiifier/attribute.joblib' --change_intensity -5 #--save_combined_img
```

## Face Manipulation Results
### Age:

![](<repository_images/age_2.png>)
![](<repository_images/age_3.png>)

### Gender:

![](<repository_images/gender_1.png>)
![](<repository_images/gender_3.png>)

### Ethnicity:

![](<repository_images/ethnicity_1.png>)
![](<repository_images/ethnicity_2.png>)

