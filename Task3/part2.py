import cv2
import numpy as np


# CONFIGURES
FPS_LIM = 60
UPDATE_IF_LESS_THAN = 4
UPDATE_EVERY_N_FRAMES = 120

SAVE_RESULT = False
RESULT_FPS = 30


# FUNCTIONS
def get_tracker_points(frame):
    print("Updating with ORB")
    # Frame features
    kp2, des2 = orb.detectAndCompute(frame, None)
    # Find matches
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]
    # Points to track
    src = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    return cv2.perspectiveTransform(border, M)


# Read image
marker = cv2.imread("Task3/data/marker.jpg", cv2.IMREAD_GRAYSCALE)

# Read video
video = cv2.VideoCapture("Task3/data/find_chocolate.mp4")
if not video.isOpened():
    print("Error: unable to open video!")
    exit(0)
ret, old_frame = video.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Def marker borders
border = np.float32([
    [0, 0],
    [0, marker.shape[0] - 1],
    [marker.shape[1] - 1, marker.shape[0] - 1],
    [marker.shape[1] - 1, 0]
    ]).reshape(-1, 1, 2)

# ORB & Brute-Force Matcher
orb = cv2.ORB_create(WTA_K=3, nlevels=18)
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# Finding keypoints to track
kp1, des1 = orb.detectAndCompute(marker, None)
t_border0 = get_tracker_points(old_gray)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize = (81, 81),
    maxLevel = 4,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Stuff for trajectories
color = np.random.randint(0, 255, (t_border0.shape[0], 3))
mask = np.zeros_like(old_frame)

if SAVE_RESULT:
    result = cv2.VideoWriter("Task3/result/result2.mp4", -1,
    RESULT_FPS, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frames = 0
while(True):
    frames += 1
    # Read next frame
    ret, frame = video.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    t_border1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, t_border0, None, **lk_params)
    
    # Select good points
    if t_border1 is not None:
        good_new = t_border1[st==1]
        good_old = t_border0[st==1]
    lost = st.shape[0] - np.sum(st)
    if lost > 0: print("Lost:", lost)

    # Draw the tracker and save is specified
    cv2.polylines(frame, [t_border0.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
    if SAVE_RESULT:
        result.write(frame)

    # Draw the trajectory
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    
    cv2.imshow("frame", img)

    old_gray = frame_gray.copy()
    t_border0 = good_new.reshape(-1, 1, 2)

    # Get new points if need
    if t_border0.shape[0] < UPDATE_IF_LESS_THAN or frames % UPDATE_EVERY_N_FRAMES == 0:
        t_border0 = get_tracker_points(old_gray)
    
    if cv2.waitKey(1000 // FPS_LIM) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()