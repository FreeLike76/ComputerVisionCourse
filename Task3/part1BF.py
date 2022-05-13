import cv2
import numpy as np

FPS_LIM = 60
MIN_MATCHES = 120
DRAW_PREV_IF_FAIL = True

marker = cv2.imread("Task3/data/marker.jpg", cv2.IMREAD_GRAYSCALE)
video = cv2.VideoCapture("Task3/data/find_chocolate.mp4")

orb = cv2.ORB_create(WTA_K=3, nlevels=14)
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

kp1, des1 = orb.detectAndCompute(marker, None)

# points on marker
border = np.float32([
    [0, 0],
    [0, marker.shape[0] - 1],
    [marker.shape[1] - 1, marker.shape[0] - 1],
    [marker.shape[1] - 1, 0]
    ]).reshape(-1, 1, 2)

while(video.isOpened()):
    ret, frame = video.read()
    if not ret: break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find key points
    kp2, des2 = orb.detectAndCompute(frame, None)

    # find matches
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    print("Matches:", len(matches))
    if len(matches) < MIN_MATCHES:
        print("Not enought matches")
        if DRAW_PREV_IF_FAIL and transformed_border is not None:
            frame = cv2.polylines(frame, [transformed_border], True, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # find homography
        src_pts  = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts  = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # apply transform to border box
        transformed_border = cv2.perspectiveTransform(border, M).astype(np.int32)

        # draw tracker
        frame = cv2.polylines(frame, [transformed_border], True, (255, 255, 255), 2, cv2.LINE_AA)
    
    # display
    matching_result = cv2.drawMatches(marker, kp1, frame, kp2, matches, None)
    cv2.imshow("Display", matching_result)

    if cv2.waitKey(1000 // FPS_LIM) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
