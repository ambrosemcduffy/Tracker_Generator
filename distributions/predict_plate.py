import os
import numpy as np
import sys
import cv2
import argparse
from inference import predict, get_roi


def predict_plate(directory, frame_stop):
    names_l = [name for name in os.listdir(directory) if name.endswith('.jpg')]
    haar_cascade = 'Cascade/haarcascade_frontalface_default.xml'
    trackers_l = []
    cnt = 0
    for name in names_l:
        plate = cv2.imread(directory+name)
        image, faces = get_roi(plate, haar_cascade_path=haar_cascade)
        try:
            roi, x, y = predict(image, faces)
            trackers_l.append([x, y])
        except TypeError:
            print('Caught an error continuing')
            x = np.zeros(5)
            y = np.zeros(5)
            trackers_l.append([x, y])
        print("Predicting image {}".format(cnt))
        if cnt == frame_stop:
            break
        cnt += 1
    return trackers_l


def export_trackers(directory, frame_stop=10):
    tracker_list = predict_plate(directory, frame_stop)
    orig_stdout = sys.stdout
    f = open('{}/out.txt'.format(directory), 'w')
    sys.stdout = f

    for i in range(5):
        print('\n"Tracker{:04d}"'.format(i+1))
        print('1')
        print(frame_stop)
        cnt = 0
        for x, y in tracker_list:
            print(cnt, x[i], y[i],  "1.000000")
            cnt += 1
    sys.stdout = orig_stdout
    f.close()


export_trackers('plates/', 290)

'''
def Main():
    parser = argparse.ArgumentParser(description='export tracking points')
    parser.add_argument('-f', type=str, help='file plate path')
    parser.add_argument('-fn', type=int, help='frame numbers')
    args = parser.parse_args()
    export_trackers(args.f, args.fn)


if __name__ == '__main__':
    Main()
'''