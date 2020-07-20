"""
This Script Tests the Models on real world Data
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from model import Network


def get_roi(image, haar_cascade_path):
    """
    Args:
        path: image file path
        haar_cascade: haarcascade detection file
    Returns: image, faces matrix detected for n-faces, offsets coordinates
    """
    # Import a single image
    # image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # implementing haarCascade face detector classifier
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    faces = face_cascade.detectMultiScale(image, 1.02, 30)
    image_with_detection = image.copy()

    # Draw a box around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(image_with_detection, (x, y), (x+w, y+h), (255, 0, 0), 3)
    return image, faces


def scale_keypoints(keypts, orig_img, new_img):
    keypts[:, 0] *= new_img.shape[0] / orig_img.shape[0]
    keypts[:, 1] *= new_img.shape[1] / orig_img.shape[1]
    return keypts


def predict(image, faces, display=False):
    """
    This function predicts keypoints, and plots them
    Args:
        image: image being used to pass into the neural net
        faces: detected faces
    Returns: None
    """
    # initialize the network with gpu support
    net = Network().cuda()
    model_dir = 'saved_models/'
    model_name = '_model_save.pt'

    # loading in the model
    net.load_state_dict(torch.load(model_dir+model_name))

    # setting the model to evaluate mode.
    net.eval()
    image_copy = np.copy(image)
    # flipping the image, so that coordinates corelate with pftracks
    roi_l = []
    # detecting the faces in the image
    for face in faces:
        (x, y, w, h) = face
        # initializing padding
        # padding = 10
        # getting the region of the face with padding
        roi = image_copy[y:y+h, x:x+w]
        roi = cv2.flip(roi, 0)
        roi_l.append(roi)
        new_y = image.shape[0] - h - y
        # converting the image to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        # Normalizing the image to become [0, 1]
        img = (gray/255.0).astype('float32')
        # resizing the image to the model dimensions
        img_resize = cv2.resize(img, (224, 224))
        img = img_resize.reshape(1,
                                 1,
                                 img_resize.shape[0],
                                 img_resize.shape[1])

        # pylint: disable=maybe-no-member
        img = torch.from_numpy(img)
        img = torch.FloatTensor(img)
        keypts = net(img.cuda())
        # unNormalzing the keypoints
        keypts = keypts.view(5, 2).data.cpu().numpy()
        keypts = keypts * 50.0 + 100.
        # scaling up the keypoints to ROI size
        keypts = scale_keypoints(keypts, img_resize, roi)
        x_pos = keypts[:, 0] + x
        y_pos = keypts[:, 1] + new_y
        if display is True:
            flip_image = cv2.flip(image, 0)
            plt.imshow(flip_image, cmap='gray', origin='lower')
            plt.scatter(x_pos,
                        y_pos,
                        s=20,
                        marker='.',
                        c='yellow')
            plt.axis('off')
        return roi, x_pos, y_pos
