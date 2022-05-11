import cv2
import numpy as np

FPS = 10

marker = cv2.imread("Task3/data/marker.jpg")
video = cv2.VideoCapture("Task3/data/find_chocolate.mp4")

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(marker, None)

# points on marker
pts = np.float32([
    [0, 0],
    [0, marker.shape[0] - 1],
    [marker.shape[1] - 1, marker.shape[0] - 1],
    [marker.shape[1] - 1, 0]
    ]).reshape(-1, 1, 2)

while(video.isOpened()):
    ret, frame = video.read()
    
    # find key points
    kp2, des2 = orb.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # find matches
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # find homography
    src_pts  = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts  = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # apply transform to border box
    dst = cv2.perspectiveTransform(pts, M)

    # display
    frame = cv2.polylines(frame, [dst.astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
    matching_result = cv2.drawMatches(marker, kp1, frame, kp2, matches[:10], None)
    cv2.imshow("Display", matching_result)

    if cv2.waitKey(1000//FPS) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
