import matplotlib.image as mpimg
import json
import argparse
import os
import cv2
import numpy as np


def getFileNames(path):
    """
    """
    names_ = []
    if os.path.exists(path):
        names = os.listdir(path)
        for name in names:
            if name.endswith('.jpg'):
                names_.append(name)
    return names_


def import_images(path):
    names_ = getFileNames(path)
    images = []
    for image in names_:
        img = mpimg.imread(path+image)
        images.append(cv2.flip(img, 0))
    return np.array(images)


def get_targetData(target_path):
    with open(target_path, 'r') as f:
        data = json.load(f)
        len_data = len(data)
    target = []
    for i in range(len_data):
        target.append(data['Tracker'+str(i+1)])
    target = np.array(target)
    return target


def create_dateSet(img_path, target_path):
    targets = get_targetData(target_path)
    t = []
    for i in range(targets.shape[1]):
        t.append(targets[:, i, :])
    images = import_images(img_path)
    return images, np.array(t)


def generateDataset(root_dir='data/', save_file=False, test_set=False):
    print(root_dir+'trackers.json')
    x, y = create_dateSet(root_dir,
                          root_dir+'trackers.json')
    if test_set is True:
        percent = np.round(x.shape[0] * 0.1).astype(np.int)
        x_test = x[:percent]
        y_test = y[:percent]
        x = x[percent:]
        y = y[percent:]
        print(x.shape, x_test.shape)
    if save_file is True:
        np.savez(root_dir+'data', a=x, b=y)
        np.savez(root_dir+'data_test', a=x_test, b=y_test)
        print("Saving out numpy file..")
    else:
        return x, y


generateDataset(root_dir='data/train_data/', save_file=True, test_set=True)

'''
def Main():
    parser = argparse.ArgumentParser(description='creates npz dataset')
    parser.add_argument('-f', type=str, help='file to a root dir')
    parser.add_argument('-s', type=bool, help='Save file')
    args = parser.parse_args()
    generateDataset(args.f, args.s)


if __name__ == '__main__':
    Main()
'''