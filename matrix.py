import cv2
import numpy as np
image=cv2.imread("C:/Users/advip/Desktop/ENDO/test/c (4).JPG")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_image=cv2.resize(image_rgb,(5,5))
red_channel=resized_image[:,:,0]
green_channel=resized_image[:,:,1]
blue_channel=resized_image[:,:,2]
print("Red Channel:\n",red_channel)
print("Green Channel:\n",green_channel)
print("Blue Channel:\n",blue_channel)
