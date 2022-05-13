import cv2
import numpy as np

FPS_LIM = 60
MIN_GOOD_MATHCES = 10

marker = cv2.imread("Task3/data/marker.jpg", cv2.IMREAD_GRAYSCALE)
video = cv2.VideoCapture("Task3/data/find_chocolate.mp4")

orb = cv2.ORB_create(WTA_K=3, nlevels=16)

index_params = dict(
    algorithm = 6,          # FLANN_INDEX_LSH
    table_number = 12,
    key_size = 20,
    multi_probe_level = 2)

search_params = dict(checks=100)

flann = cv2.FlannBasedMatcher(index_params, search_params)

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
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    try:
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)
    except:
        pass
    
    print("Matches:", len(good_matches))
    if len(good_matches) < MIN_GOOD_MATHCES:
        print("Not enought good matches")
    
    else:
        # find homography
        src_pts  = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts  = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # apply transform to border box
        transformed_border = cv2.perspectiveTransform(border, M).astype(np.int32)

        # draw tracker
        frame = cv2.polylines(frame, [transformed_border], True, (255, 255, 255), 2, cv2.LINE_AA)

    # display
    matching_result = cv2.drawMatches(marker, kp1, frame, kp2, good_matches, None)
    cv2.imshow("Display", matching_result)

    if cv2.waitKey(1000 // FPS_LIM) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()