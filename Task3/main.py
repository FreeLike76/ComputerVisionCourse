import cv2
import numpy as np

FPS = 24

marker = cv2.imread("Task3/data/marker.jpg")
video = cv2.VideoCapture("Task3/data/find_chocolate.mp4")

orb = cv2.ORB_create()

while(video.isOpened()):
    ret, frame = video.read()
    
    kp1, des1 = orb.detectAndCompute(marker, None)
    kp2, des2 = orb.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matching_result = cv2.drawMatches(marker, kp1, frame, kp2, matches[:20], None)
    cv2.imshow("Display", matching_result)

    if cv2.waitKey(1000//FPS) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
