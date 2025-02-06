import numpy as np
import cv2

def reconstruct_3d(images):
    sift = cv2.SIFT_create()
    keypoints_list, descriptors_list = [], []

    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_list[0], descriptors_list[1], k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    points_3d = []
    for match in good_matches:
        pt1 = keypoints_list[0][match.queryIdx].pt
        pt2 = keypoints_list[1][match.trainIdx].pt
        points_3d.append([pt1[0], pt1[1], np.linalg.norm(np.array(pt1) - np.array(pt2))])

    return points_3d
