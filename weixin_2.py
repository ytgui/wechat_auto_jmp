import time
import random
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


def apply_swipe(x1, y1, x2, y2, delay_ms):
    swipe = '{} {} {} {}'.format(x1, y1, x2, y2)
    delay_ms = '{}'.format(int(delay_ms))
    subprocess.run(['adb', 'shell', 'input', 'swipe', swipe, delay_ms])


def contour_area(contour):
    a = 0
    prev = contour[-1]
    for pt in contour:
        a += prev[0] * pt[1] - prev[1] * pt[0]
        prev = pt
    a *= 0.5
    return a


def check_chess(edge, ax):
    def check_bounding_rect(points):
        top, bottom = np.floor(np.min(points[:, 0])), np.ceil(np.max(points[:, 0]))
        left, right = np.floor(np.min(points[:, 1])), np.ceil(np.max(points[:, 1]))
        width, height = right - left, bottom - top
        ss = width * height
        tt = np.arctan2(np.minimum(width, height), np.maximum(width, height))
        if 10000 <= ss <= 40000 and 0.45 < tt < 0.75:
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


def check_board(edge, cx):
    if cx < 540:  # chess on the left
        valid_edge = edge[:, 540:]
    else:
        valid_edge = edge[:, : 540]

    idx = 0
    vertical = np.sum(valid_edge, axis=1)
    for idx, top in enumerate(vertical):
        if top > 5:
            break
    by = idx
    #
    bxs = []
    horizontal = valid_edge[by + 1, :]
    for idx, left in enumerate(horizontal):
        if left > 0.1:
            bxs.append(idx)
    if cx < 540:  # chess on the left
        bx = 540 + np.average(bxs)
    else:
        bx = np.average(bxs)
    return bx, by


def play_again(img):
    return False


def main():
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    while True:
        img = get_frame()
        # img = skimage.io.imread('13.png')
        if play_again(img):
            # apply_swipe(1600, 500, 100)
            apply_swipe(500, 1600, 500, 1600, 100)
            time.sleep(1.0)
            continue
        img = img[420:1500, :, 0:3]
        hsv = skimage.color.rgb2hsv(img)
        #
        img_r, img_g, img_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        img_r = skimage.exposure.equalize_adapthist(img_r)
        img_g = skimage.exposure.equalize_adapthist(img_g)
        img_b = skimage.exposure.equalize_adapthist(img_b)
        gray = skimage.color.rgb2gray(img_r + img_g + img_b)
        #
        edge = skimage.feature.canny(gray, sigma=1.2)
        edge = skimage.filters.gaussian(edge)
        edge = skimage.morphology.dilation(edge, skimage.morphology.square(7))
        edge = skimage.morphology.erosion(edge, skimage.morphology.square(3))

        ax[0].clear()
        ax[0].imshow(img)
        ax[1].clear()
        ax[1].imshow(edge)
        # check chess
        result1 = np.logical_and(np.logical_and(hsv[:, :, 0] > 0.65, hsv[:, :, 0] < 0.75),
                                 np.logical_and(hsv[:, :, 1] > 0.20, hsv[:, :, 1] < 0.50))
        result1 = np.logical_and(result1, np.logical_and(hsv[:, :, 2] > 0.25, hsv[:, :, 2] < 0.5))
        result2 = np.logical_and(np.logical_and(img[:, :, 0] > 40, img[:, :, 0] < 80),
                                 np.logical_and(img[:, :, 1] > 40, img[:, :, 1] < 80))
        result2 = np.logical_and(result2, np.logical_and(img[:, :, 2] > 50, img[:, :, 2] < 120))
        result = np.logical_and(result1, result2)
        result = skimage.morphology.dilation(result, skimage.morphology.square(7))

        ax[2].clear()
        ax[2].imshow(result)

        roi_chess = check_chess(result, ax[2])
        for top, bottom, left, right in roi_chess:
            rr, cc = skimage.draw.polygon_perimeter([top, top, bottom, bottom], [left, right, right, left])
            ax[1].plot(cc, rr, 'r-', linewidth=3)

        d = None
        if len(roi_chess) == 1:
            #
            (top, bottom, left, right), = roi_chess
            cx, cy = (left + right) // 2, top
            ax[1].scatter(cx, cy, marker='x', s=100, c='g')
            # check board
            bx, by = check_board(edge, cx)
            ax[1].scatter(bx, by, marker='x', s=100, c='g')
            #
            d = np.abs(cx - bx) + np.abs(cy - by)

        # plt.show()
        plt.draw()
        plt.pause(1.0)
        if d is not None:
            print('distance', d)
            x1, y1 = random.randrange(600), random.randrange(600)
            if d < 400:
                apply_swipe(x1, y1, x1, y1, 1.15 * d)
            elif 400 <= d < 700:
                apply_swipe(x1, y1, x1, y1, 1.15 * d)
            elif 700 <= d < 1200:
                apply_swipe(x1, y1, x1, y1, 1.05 * d)
            else:
                apply_swipe(x1, y1, x1, y1, 0.9 * d)
        plt.pause(2.0)


if __name__ == '__main__':
    main()
