import pandas as pd
import numpy as np
import cv2
from filterpy.common import kinematic_kf
from filterpy.kalman import IMMEstimator
from filterpy.stats import mahalanobis
from copy import deepcopy
import math
from tqdm import tqdm as tqdm
import random

MATCH_ID = 27
PER_FRAME = 40
HEIGHT = 1080
WIDTH = 1920
COURT = np.array([(351,128), (858,144), (1268,339), (1268,606), (17, 606), (24,339)]).reshape(-1, 2)
plot = False

def diff_image(im_tm1, im_t, im_tp1):
    delta_plus = cv2.absdiff(im_t, im_tm1)
    delta_minus = cv2.absdiff(im_t, im_tp1)

    sp = cv2.meanStdDev(delta_plus)
    sm = cv2.meanStdDev(delta_minus)

    th = [
        sp[0][0, 0] + 3 * math.sqrt(sp[1][0, 0]),
        sm[0][0, 0] + 3 * math.sqrt(sm[1][0, 0]),
    ]

    ret, dbp = cv2.threshold(delta_plus, th[0], 255, cv2.THRESH_BINARY)
    ret, dbm = cv2.threshold(delta_minus, th[1], 255, cv2.THRESH_BINARY)

    return cv2.bitwise_and(dbp, dbm)

def detect_components_in_region(nd, region, MIN_AREA=5, MAX_AREA=300, TOP_N=100, psi=0.7, gamma=0.7):
    components = []
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(nd, ltype=cv2.CV_16U)

    candidates = list()
    for stat in stats:
        area = stat[cv2.CC_STAT_AREA]
        if area < MIN_AREA or area > MAX_AREA:
            continue  # Skip small/large objects (noise)

        lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
        rb = (lt[0] + stat[cv2.CC_STAT_WIDTH], lt[1] + stat[cv2.CC_STAT_HEIGHT])

        candidates.append((lt, rb, area))

    candidates.sort(key=lambda tup: tup[2], reverse=True)

    for i, candidate in enumerate(candidates):
        if i>= TOP_N:
            break

        # The first two elements of each `candidate` tuple are
        # the opposing corners of the bounding box.
        x1, y1 = candidate[0]
        x2, y2 = candidate[1]
        # The third element of the tuple is the area.
        actual_area = candidate[2]

        # For each candidate, estimate the "radius" using a distance transform.
        # The transform is computed on the (small) bounding rectangle.
        cand = nd[y1:y2, x1:x2]
        dt = cv2.distanceTransform(cand, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
        radius = np.amax(dt)

        # "Thinning" of pixels "close to the center" to estimate a
        # potential FOM path.
        ret, Pt = cv2.threshold(dt, psi * radius, 255, cv2.THRESH_BINARY)

        # For now, we estimate it as the max possible length in the bounding box, its diagonal.
        w = x2 - x1
        h = y2 - y1
        path_len = math.sqrt(w * w + h * h)
        expected_area = radius * (2 * path_len + math.pi * radius)

        area_ratio = abs(actual_area / expected_area - 1)
        # is_fom = area_ratio < gamma and y1 > 180 and y1 < 400 and x1>500 and x1<750
        is_fom = area_ratio < gamma and cv2.pointPolygonTest(region, (x1,y1), False) == 1 and cv2.pointPolygonTest(region, (x2, y2), False) ==1
        if is_fom:
            components.append((x1, y1, x2, y2, radius))
    return components

def moving_blob_detector(frames):
    img_tm1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    img_t = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
    img_tp1 = cv2.cvtColor(frames[2], cv2.COLOR_BGR2GRAY)
    diff = diff_image(img_tm1, img_t, img_tp1)
    diff = cv2.erode(diff, np.ones((2, 2), np.uint8), iterations=1)  # To reduce the noises
    cv2.imshow('diff', diff)
    # cv2.waitKey(0)
    blobs = detect_components_in_region(diff, COURT)
    return blobs


# add noise
def add_noise(df, frac, noise='FP', L=1):
    if noise == 'FP': # add FP
        for index in random.sample(range(0, len(df)), int(len(df) * frac) // L):
            for i in range(L):
                df.loc[index + i, 'x'] = random.randint(0, WIDTH)
                df.loc[index + i, 'y'] = random.randint(0, HEIGHT)
    elif noise == 'FN': # add FN
        for index in random.sample(range(0, len(df)), int(len(df) * frac) // L):
            for i in range(L):
                df.loc[index + i, 'x'] = -1
                df.loc[index + i, 'y'] = -1
                df.loc[index + i, 'score'] = 0
    return df


curPt = []
def click_point(event, x, y, flags, param):
    global curPt
    if event == cv2.EVENT_LBUTTONDOWN:
        curPt = [x, y]
        # print('curPt ', curPt)
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', click_point)
thresh1 = 0.9
thresh3 = 7

df_gt = pd.read_csv('../csv_files/m-0' + str(MATCH_ID) + '-ball-gt.csv')
df_test_1 = pd.read_csv('../csv_files/m-0' + str(MATCH_ID) + '-single.csv')
result = []
frames = []

for ball_sn in tqdm(df_gt.ball_sn.unique()):
    # TAs, Tracks for false Alarms
    fa_trackers = []
    ta_template = kinematic_kf(dim=2, order=1, dt=1.0, order_by_dim=False)
    ta_template.Q = np.diag([1.5, 1.5, 1.5, 1.5])
    ta_template.R = np.diag([4.5, 4.5])
    TA_GATING = 3.0
    TA_COV_THRESHOLD = 400.0

    # TBs, Tracks for the Ball
    tb1 = kinematic_kf(dim=2, order=2, dt=1.0, order_by_dim=False)
    tb1.Q = np.diag([13.5] * 6)
    tb1.R = np.diag([14.5, 14.5])
    tb1.x = np.array([[WIDTH / 2, HEIGHT / 2, 0., 0., 0.0, 0.0]]).T
    tb1.P = np.eye(6) * 100**2

    tb2 = kinematic_kf(dim=2, order=2, dt=1.0, order_by_dim=False)
    tb2.F = np.array([[1., 0, 0, 0, 0, 0],
                      [0., 1, 0, 0, 0, 0],
                      [0., 0, 0, 0, 0, 0],
                      [0., 0, 0, 0, 0, 0],
                      [0., 0, 0, 0, 0, 0],
                      [0., 0, 0, 0, 0, 0]])
    tb2.Q = np.diag([900] * 6)
    tb2.R = np.diag([14.5, 14.5])
    tb2.x = np.array([[WIDTH / 2, HEIGHT / 2, 0., 0., 0.0, 0.0]]).T
    tb2.P = np.eye(6) * 100 ** 2

    filters = [tb1, tb2]
    M = np.array([[0.8, 0.2],
                  [0.7, 0.3]])
    mu = np.array([0.5, 0.5])
    bank = IMMEstimator(filters, mu, M)
    avg_F = 0.7 * filters[0].F + 0.3 * filters[1].F
    max_Q = filters[0].Q + filters[1].Q

    TB_GATING = np.array([9.0, 6.0])

    # TS, Track for Smoother buffer
    ts_buffer_xs, ts_buffer_Ps = [], []
    df_temp_gt = df_gt[df_gt.ball_sn == ball_sn]
    df_temp_1 = df_test_1[df_test_1.ball_sn == ball_sn]
    START_MS, END_MS = min(df_temp_gt.time_ms.unique()), max(df_temp_gt.time_ms.unique())

    for t in range(START_MS, END_MS + 1, PER_FRAME):
        # step 1
        bank.predict()
        frame = cv2.imread('../train_imgs/%d_%d.jpg' % (MATCH_ID, t))
        ball_centers_1 = df_temp_1[(df_temp_1.time_ms == t) & (df_temp_1.score > thresh1)].to_numpy()[:, 3:]
        ball_centers = np.array([])

        # Uncomment the following code to see the x,P for each TB
        x, y = int(round(filters[0].x[0, 0])), int(round(filters[0].x[1, 0]))
        r1, r2 = int(round(math.sqrt(filters[0].P[0, 0]))), int(round(math.sqrt(filters[0].P[1, 1])))
        cv2.ellipse(frame, (x, y), (r1, r2), 0, 0, 360, (0, 255, 0), 1)
        x, y = int(round(filters[1].x[0, 0])), int(round(filters[1].x[1, 0]))
        r1, r2 = int(round(math.sqrt(filters[1].P[0, 0]))), int(round(math.sqrt(filters[1].P[1, 1])))
        cv2.ellipse(frame, (x, y), (r1, r2), 0, 0, 360, (255, 0, 0), 1)

        # step 2, delete TA[i] if its cov is too large
        for i in range(len(fa_trackers))[::-1]:
            ta = fa_trackers[i]
            ta.predict()
            # Uncomment the following code to see the x,P for each TA
            x, y = int(round(ta.x[0, 0])), int(round(ta.x[1, 0]))
            r1, r2 = int(round(math.sqrt(ta.P[0, 0]))), int(round(math.sqrt(ta.P[1, 1])))
            cv2.ellipse(frame, (x, y), (r1, r2), 0, 0, 360, (0, 0, 255), 1)
            if max(ta.P[0,0], ta.P[1,1]) > TA_COV_THRESHOLD: # threshold set here
                del fa_trackers[i]
                print('false alarm tracks: ', len(fa_trackers))

        # step 3, for each TA, assosiate with the nearest candidate, remove it, single frame
        if ball_centers_1.size>0:
            for i in range(len(fa_trackers)):
                ta = fa_trackers[i]
                if ball_centers_1.size > 0:
                    unassigned_dts = [i for i in range(ball_centers_1.shape[0])]
                    distances = [mahalanobis(ball_centers_1[j, 0:2], ta.x[0:2], ta.P[0:2, 0:2]) for j in
                                 unassigned_dts]
                    nearest = np.argmin(distances)
                    if distances[nearest] < TA_GATING:
                        ta.update(ball_centers_1[nearest, 0:2])
                        unassigned_dts.remove(nearest)
                        # print(ball_centers_1[nearest, 0:2], 'is false alarm, by ', ta.x[0,0], ta.x[1,0],
                        #       ta.P[0,0], ta.P[1,1])
                        ball_centers_1 = ball_centers_1[unassigned_dts]

        updated = False
        # step 4, B.update() using average ball in gate, single frame
        if ball_centers_1.size > 0:
            un_associated, gated_balls, weights = [], [], []
            for i in range(ball_centers_1.shape[0]):
                distances = np.array(
                    [mahalanobis(ball_centers_1[i, 0:2], kf.x[0:2], kf.P[0:2, 0:2]) for kf in filters])
                if np.all((distances - TB_GATING)<0):
                    gated_balls.append(ball_centers_1[i, 0:2])
                    weights.append(ball_centers_1[i, 2] * math.exp(-1*np.min(distances)))
                    print(ball_centers_1[i, 0:2], 'is a tracked ball')
                else:
                    un_associated.append(i)
                    print('distances:', distances)

            if len(gated_balls)>0:
                updated = True
                gated_balls = np.array(gated_balls).reshape(-1, 2)
                weights = weights / sum(weights)
                avg_ball = np.average(gated_balls, axis=0, weights=weights)
                bank.update(avg_ball)

                for ball_pos in ball_centers_1:
                    cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 5, (255, 0, 255), 3)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (int(ball_pos[0]), int(ball_pos[1]))
                    fontScale = 0.5
                    fontColor = (255, 255, 255)
                    lineType = 2
                    cv2.putText(frame, str(round(ball_pos[2], 5)),
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)

            ball_centers = ball_centers_1[un_associated]

        # step 5, create additional TAs, single frame
        if updated and ball_centers_1.size > 0:
            for i in range(ball_centers_1.shape[0]):
                ds = np.array(
                    [mahalanobis(ball_centers_1[i, 0:2], kf.x[0:2], kf.P[0:2, 0:2]) for kf in filters])
                print(ds)
                print((ds - TB_GATING*0.5))
                if np.all((ds - TB_GATING*0.5)>0):  # set the margin here
                    ta = deepcopy(ta_template)
                    ta.x = np.array([[ball_centers_1[i, 0], ball_centers_1[i, 1], 0., 0.]]).T
                    fa_trackers.append(ta)
                    print('Add a false alarm tracks: ', len(fa_trackers))
                    print(ball_centers_1[i, 0:2], 'is false alarm')

        # step 6
        ts_buffer_xs.append(bank.x.copy())
        ts_buffer_Ps.append(bank.P.copy())

        # code to show the result
        x, y = int(round(bank.x[0, 0])), int(round(bank.x[1, 0]))
        if bank.mu[0] > 0.8:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        r = max(2, int(round(math.sqrt(bank.P[0,0]))))
        cv2.circle(frame, (x, y), r, color, 2)

        if plot:
            cv2.imshow('frame', frame)
            k = cv2.waitKey(0)
            if k == ord('q'):
                break

    # step 6, do the smoothing at the end of the batch
    # smoothed_xs, smoothed_Ps, _, _ = ts.rts_smoother(np.array(ts_buffer_xs).reshape(-1, 6, 1), np.array(ts_buffer_Ps).reshape(-1, 6, 6))
    smoothed_xs, smoothed_Ps, _, _ = filters[0].rts_smoother(np.array(ts_buffer_xs).reshape(-1, 6, 1), np.array(ts_buffer_Ps).reshape(-1, 6, 6),
                                                             [avg_F] * len(ts_buffer_xs), [max_Q] * len(ts_buffer_Ps))

    for i, t in enumerate(range(START_MS, END_MS + 1, 40)):
        result.append([MATCH_ID, t, smoothed_xs[i][0][0], smoothed_xs[i][1][0]])

    sequence_pos = []
    for t in range(START_MS, END_MS + 1, PER_FRAME):
        frame = cv2.imread('../train_imgs/%d_%d.jpg' % (MATCH_ID, t))
        ball_center = smoothed_xs[int(round((t - START_MS) / PER_FRAME))][0:2].reshape(2)
        P = smoothed_Ps[int(round((t - START_MS) / PER_FRAME))]
        if len(sequence_pos) >= 6:
            del sequence_pos[0]
        sequence_pos.append([ball_center[0], ball_center[1]])
        # sequence_pos = [[ball_center[0], ball_center[1]]]
        # r = max(8, int(round(math.sqrt(P[0,0]))))
        for pos in sequence_pos:
            cv2.circle(frame, (int(round(pos[0])), int(round(pos[1]))), 10, (0, 255, 255), 1)
        cv2.circle(frame, (int(round(sequence_pos[-1][0])), int(round(sequence_pos[-1][1]))), 10, (0, 255, 255), 2)
        # cv2.circle(frame, (int(round(pos[0])), int(round(pos[1]))), r, (0, 255, 255), 1)
        cv2.imshow('frame', frame)
        frames.append(frame)
        curPt = []
        k = cv2.waitKey(20)

out = cv2.VideoWriter('../demo_videos/m-0' + str(MATCH_ID) + '-ball-tracker-1f-final.mp4',
                      cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080))
for i in range(len(frames)):
    out.write(frames[i])
out.release()

def get_dist(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

df_result = pd.DataFrame(result, columns=['match_id', 'time_ms', 'x', 'y'])
tp, fp, fn = 0, 0, 0
for index, row_gt in tqdm(df_gt.iterrows()):
    rows = df_result[(df_result.time_ms == row_gt.time_ms)]
    if len(rows) == 0:
        fn += 1
        continue
    row = rows.iloc[0]
    if get_dist(row.x, row_gt.x, row.y, row_gt.y) < 20:
        tp += 1
    else:
        fp += 1

print("tp, fp, fn:", tp, fp, fn)
print("Accuracy:", round(tp / (tp + fp + fn), 4))
print(len(df_test_1[df_test_1.score != 0]))
