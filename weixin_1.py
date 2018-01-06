import time
import numpy as np
import skimage.io, skimage.feature, skimage.transform, skimage.draw, skimage.morphology, skimage.color, skimage.measure, \
    skimage.filters, skimage.exposure
import matplotlib.pyplot as plt
import subprocess


def get_frame():
    subprocess.run(['adb', 'shell', '/system/bin/screencap', '/sdcard/screenshot.png'])
    subprocess.run(['adb', 'pull', '/sdcard/screenshot.png', 'screenshot.png'])
    img = skimage.io.imread('screenshot.png')
    return img


def apply_swipe(x, y, delay_ms):
    swipe = '{} {} {} {}'.format(x, y, x, y)
    delay_ms = '{}'.format(int(delay_ms))
    subprocess.run(['adb', 'shell', 'input', 'swipe', swipe, delay_ms])


def project_3d_2d(img=None, points=None):
    model = skimage.transform.ProjectiveTransform()
    # hm, wm = 960, 540
    hm, wm = 800, 700
    model.estimate(np.array([(wm - 64, hm + 64), (wm + 64, hm + 64), (wm + 64, hm - 64), (wm - 64, hm - 64)]),
                   np.array([(210, 1052), (435.3, 1182.3), (661, 1052), (435.5, 921.5)]))
    if img is not None:
        return skimage.transform.warp(img, model)
    else:
        return model(points)


def project_2d_3d(img=None, points=None):
    model = skimage.transform.ProjectiveTransform()
    hm, wm = 800, 700
    model.estimate(np.array([(210, 1052), (435.3, 1182.3), (661, 1052), (435.5, 921.5)]),
                   np.array([(wm - 64, hm + 64), (wm + 64, hm + 64), (wm + 64, hm - 64), (wm - 64, hm - 64)]))
    if img is not None:
        return skimage.transform.warp(img, model)
    else:
        return model(points)


def contour_area(contour):
    a = 0
    prev = contour[-1]
    for pt in contour:
        a += prev[0] * pt[1] - prev[1] * pt[0]
        prev = pt
    a *= 0.5
    return a


def check_if_contour_is_square(contour):
    top, bottom = np.min(contour[:,1]), np.max(contour[:,1])
    left, right = np.min(contour[:,0]), np.max(contour[:,0])


def check_board(edge):
    def check_bounding_rect(points):
        top, bottom = np.floor(np.min(points[:, 0])), np.ceil(np.max(points[:, 0]))
        left, right = np.floor(np.min(points[:, 1])), np.ceil(np.max(points[:, 1]))
        width, height = right - left, bottom - top
        ss = width * height
        tt = np.arctan2(np.minimum(width, height), np.maximum(width, height))
        if 8000 <= ss <= 30000 and np.pi * (1 / 4 - 1 / 12) <= tt <= np.pi * (1 / 4 + 1 / 12):
            valid = True
        else:
            valid = False
        return valid, int(top), int(bottom), int(left), int(right)

    roi = []
    contours = skimage.measure.find_contours(edge, level=0.8)
    for n, contour in enumerate(contours):
        if 5000 < contour_area(contour) < 20000:
            plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1)
            valid, top, bottom, left, right = check_bounding_rect(contour)
            if valid is True:
                roi.append([top, bottom, left, right])
    return roi


def check_piece(edge, ax):
    def check_bounding_rect(points):
        top, bottom = np.floor(np.min(points[:, 0])), np.ceil(np.max(points[:, 0]))
        left, right = np.floor(np.min(points[:, 1])), np.ceil(np.max(points[:, 1]))
        width, height = right - left, bottom - top
        ss = width * height
        tt = np.arctan2(np.minimum(width, height), np.maximum(width, height))
        if 1000 <= ss <= 20000 and 0.45 < tt < 0.55:
            valid = True
        else:
            valid = False
        return valid, int(top), int(bottom), int(left), int(right)

    roi = []
    contours = skimage.measure.find_contours(edge, level=0.8)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1)
        valid, top, bottom, left, right = check_bounding_rect(contour)
        if valid is True:
            roi.append([top, bottom, left, right])
    return roi


def check_center(roi_board, roi_chess):
    if len(roi_chess) != 1:
        return None

    top, bottom, left, right = roi_chess[0]
    cx, cy = (right + left) // 2, ((bottom + top) // 2) + (bottom - top) // 4
    (cx, cy), = project_2d_3d(points=[cx, cy])
    cx, cy = int(cx), int(cy)

    board_center = []
    for top, bottom, left, right in roi_board:
        bx, by = (right + left) // 2, (bottom + top) // 2
        if cy - by > 100 or cx - bx > 100:
            board_center.append([bx, by])

    if len(board_center) == 0:
        return None
    elif len(board_center) != 1:
        board_center = np.array(board_center)
        idx = np.argmin(board_center[:, 1])
        bx, by = board_center[idx]
    else:
        (bx, by), = board_center

    return cx, cy, bx, by


def main():
    flg, ax = plt.subplots(1, 3, figsize=(15, 5))
    while True:
        # img = get_frame()
        img = skimage.io.imread('9.png')
        img = img[420:1500, :, 0:3]
        gray = skimage.color.rgb2gray(img)
        gray = skimage.exposure.equalize_hist(gray)
        edge = skimage.feature.canny(gray, sigma=0.6)
        edge = skimage.morphology.dilation(edge, skimage.morphology.square(13))
        edge = skimage.morphology.erosion(edge, skimage.morphology.square(3))
        edge = project_3d_2d(img=edge)
        # plt.imshow(edge)
        # plt.show()

        # check chess
        result = np.logical_and(np.logical_and(img[:, :, 0] > 40, img[:, :, 0] < 80),
                                np.logical_and(img[:, :, 1] > 40, img[:, :, 1] < 80))
        result = np.logical_and(result, np.logical_and(img[:, :, 2] > 60, img[:, :, 2] < 120))
        result = skimage.morphology.dilation(result, skimage.morphology.square(3))
        ax[0].imshow(img)
        roi_chess = check_piece(result, ax[2])
        for top, bottom, left, right in roi_chess:
            rr, cc = skimage.draw.polygon_perimeter([top, top, bottom, bottom], [left, right, right, left])
            ax[0].plot(cc, rr, 'r-', linewidth=3)

        # check board
        ax[1].imshow(edge, cmap='gray')
        roi_board = check_board(edge)
        for top, bottom, left, right in roi_board:
            rr, cc = skimage.draw.polygon_perimeter([top, top, bottom, bottom], [left, right, right, left])
            ax[1].plot(cc, rr, 'g-', linewidth=3)

        # check
        d = 0
        apply = False
        ax[2].imshow(edge)
        checked = check_center(roi_board, roi_chess)
        if checked is not None:
            cx, cy, bx, by = checked
            ax[2].scatter(cx, cy, marker='x', s=100, c='r')
            ax[2].scatter(bx, by, marker='x', s=100, c='g')
            d = np.abs(cx - bx) + np.abs(cy - by)
            apply = True

        plt.show()
        plt.draw()
        plt.pause(1.0)
        if apply:
            apply_swipe(100, 100, 2.6 * d)
        plt.pause(3.0)


if __name__ == '__main__':
    main()
    # subprocess.run(['adb', 'shell', 'input', 'swipe', '100 100 100 100', '1000'])
