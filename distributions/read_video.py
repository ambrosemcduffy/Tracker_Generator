import cv2
import numpy
import matplotlib.pyplot as plt
from inference import predict, get_roi


haar_cascade = 'Cascade/haarcascade_frontalface_default.xml'
# image  = cv2.imread('data/picard_07.jpg')
# image, faces = get_roi(image, haar_cascade_path=haar_cascade)
#predict(image, faces)

cap = cv2.VideoCapture("data/picard_speech_02.mp4")
cnt = 0
trackers_l = []
while True:
    # Reading in the Video
    ret, frame = cap.read()
    if ret:
        image, faces = get_roi(frame, haar_cascade_path=haar_cascade)
        roi, x, y = predict(image, faces)
        #plt.axis('off')
        trackers_l.append([x[0],y[0]])
        print(cnt)
        cnt+=1
        #plt.pause(0.01)
        #plt.clf()
        #plt.show()
        #cnt+=1
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    else:
        break
#cap.release()
#cv2.destroyAllWindows()

print('"Tracker0001"')
print("1")
print(len(trackers_l))
for i in range(len(trackers_l)):
    print(i, trackers_l[i][0], trackers_l[i][1], "1.000000")