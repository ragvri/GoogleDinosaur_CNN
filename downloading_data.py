import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
from pynput import keyboard
from pynput.keyboard import Key
import pickle
import time

mon = {'top': 300, 'left': 0, 'width': 300, 'height': 300}
sct = mss()

try:
    with open('count.pickle', 'rb') as handle:
        count = pickle.load(handle)
except FileNotFoundError:
    count = 0

first = True


def jumping_training_data(key):
    global count, first
    if key == Key.up:
        if count is not 0 and first is not True:
            print("up arrow pressed")
            sct.get_pixels(mon)
            img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            cv2.imwrite("/home/raghav/Dropbox/coding/python/google_dinosaur_ANN/train_data/jump_images/testimage{}"
                        ".jpg".format(count), np.array(img))
        count += 1
        first = False
    elif key == Key.caps_lock:
        print(count)
        with open('count.pickle', 'wb') as handle:
            pickle.dump(count, handle)
        quit()

i = 1


def test_images(key):
    global i
    if key == Key.shift:
        sct.get_pixels(mon)
        img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        print("shift pressed")
        cv2.imwrite("/home/raghav/Dropbox/coding/python/google_dinosaur_ANN/test_data/no_jump_testimage{}.jpg".format(i),
                    np.array(img))
        i += 1
    elif key == Key.caps_lock:
        quit()

with keyboard.Listener(on_press=test_images) as listener:
    listener.join()
# with open('count.pickle', 'wb') as f:
#     pickle.dump(count, f)

