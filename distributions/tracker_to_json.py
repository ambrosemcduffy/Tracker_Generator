"""
This script is to export pftrack Trackers into a compatible format to then
be processed into a Neural Network.
"""


import os
import json
from collections import defaultdict

import pfpy


def get_keypoints():
    """
    This Obtains the keypoints from Pftrack
    """
    data = defaultdict(list)
    clip = pfpy.getClipRef(0)
    start = clip.getInPoint()
    end = clip.getOutPoint()
    num = pfpy.getNumTrackers()
    cnt = 0
    for i in range(0, num):
        tracker = pfpy.getTrackerRef(i)
        for frame in range(start, end+1):
            x_trans, y_trans = tracker.getTrackPosition(frame)
            data['Tracker'+str(cnt+1)].append([x_trans, y_trans])
        cnt += 1
    return data


def key_to_directory():
    """
    This Function exports Keypts into a Json to the designated folder.
    Args: None
    Returns: None
    """
    data = get_keypoints()
    path = str(raw_input("Paste file path"))
    name = 'data.json'
    if os.path.exists:
        with open(os.path.join(path, name), 'w') as file:
            json.dump(data, file)
        print("file exported")
    else:
        print("Path does not exist")


key_to_directory()
