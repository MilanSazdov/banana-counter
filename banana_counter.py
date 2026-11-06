"""Count the total value of collectibles in Donkey Kong Country: Tropical Freeze screenshots.

Values: banana = 1, banana bunch = 5, banana coin = 10.

Usage:    python banana_counter.py <dataset_folder>
Expects:  <dataset_folder>/images/<files> and <dataset_folder>/count.csv
Prints:   the mean absolute error (MAE) against the totals in count.csv
"""

import sys
import os
import csv

import numpy as np
import cv2

# HSV color ranges (OpenCV: H is 0-179, S and V are 0-255)
LOWER_YELLOW = np.array([20, 170, 185])     # bananas, bunches and coins
UPPER_YELLOW = np.array([34, 255, 255])
LOWER_PINK = np.array([165, 115, 110])      # Dixie Kong's pink hat
UPPER_PINK = np.array([179, 205, 255])

# Structuring element for morphological closing
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Region filters. Pixel sizes assume a 1280x720 image.
MIN_AREA = 10        # minimum contour area
MAX_SIDE = 60        # maximum long side of the rotated bounding box
MIN_FILL = 0.30      # minimum share of the rotated bounding box covered by the contour
MIN_S = 200          # minimum mean saturation of the region's yellow pixels
MIN_V = 210          # minimum mean brightness of the region's yellow pixels
PAD = 10             # margin around the bounding box that counts as surroundings
MAX_OTHER = 0.20     # maximum other yellow pixels in the surroundings, relative to the region area
HAT_SIDE = 40        # a hat is a pink region with a long side below this

# Shape classes
COIN_FILL = 0.72     # a coin fills its box like an ellipse (pi / 4 = 0.79)
COIN_SIDE = 24       # minimum long side of a coin
COIN_RATIO = 0.30    # minimum short / long side ratio of a coin
BUNCH_RATIO = 0.56   # short / long >= 0.56 is a bunch, below is a banana
BUNCH_SIDE = 13      # minimum short side of a bunch


def load_image(path):
    """Read an image as RGB."""
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def hat_mask(img_hsv):
    """Mask of pink regions the size of Dixie Kong's hat."""
    mask_pink = cv2.inRange(img_hsv, LOWER_PINK, UPPER_PINK)
    contours, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(mask_pink.shape, np.uint8)
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        if cv2.contourArea(contour) >= MIN_AREA and max(width, height) < HAT_SIDE:
            cv2.drawContours(mask, [contour], -1, 255, -1)      # thickness -1 fills the contour
    return mask


def find_objects(img):
    """Find collectibles in a 1280x720 RGB image. Returns a list of (value, x, y, w, h)."""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv, LOWER_YELLOW, UPPER_YELLOW)

    # Closing joins the bananas of a bunch into one region
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    mask_hat = hat_mask(img_hsv)
    contours, _ = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w / 2, y + h / 2

        # HUD: the block in the top left corner and the banana counter below it
        if cx < 200 and (cy < 115 or 205 < cy < 275):
            continue

        # Size and shape come from the rotated bounding box, so the angle does not matter
        area = cv2.contourArea(contour)
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        long_side, short_side = max(width, height), min(width, height)
        if area < MIN_AREA or long_side > MAX_SIDE:
            continue

        ratio = short_side / long_side      # 1 = square, small = elongated
        fill = area / (width * height)      # share of the box covered by the contour
        if fill < MIN_FILL:
            continue

        # Color: the region's yellow pixels must be saturated and bright on average
        mean_h, mean_s, mean_v, _ = cv2.mean(img_hsv[y:y + h, x:x + w], mask=mask[y:y + h, x:x + w])
        if mean_s < MIN_S or mean_v < MIN_V:
            continue

        # Surroundings: the bounding box grown by PAD pixels
        x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
        around = mask_close[y0:y + h + PAD, x0:x + w + PAD]

        # Isolation: collectibles stand alone, while leaves, grass and ornaments come in clusters
        own = np.zeros(around.shape, np.uint8)
        cv2.drawContours(own, [contour], -1, 255, -1, offset=(-x0, -y0))
        other = np.sum(cv2.subtract(around, own) > 0)       # yellow pixels outside the region
        if other > MAX_OTHER * area:
            continue

        # Dixie Kong's blonde ponytail looks like a banana but is always next to her pink hat
        if np.sum(mask_hat[y0:y + h + PAD, x0:x + w + PAD] > 0) > 0:
            continue

        # Class by shape: a coin is a circle or an ellipse, a bunch is compact, a banana is elongated
        if fill >= COIN_FILL and long_side >= COIN_SIDE and ratio >= COIN_RATIO:
            objects.append((10, x, y, w, h))
        elif ratio >= BUNCH_RATIO:
            if short_side >= BUNCH_SIDE:
                objects.append((5, x, y, w, h))
        else:
            objects.append((1, x, y, w, h))
    return objects


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else 'data'
    with open(os.path.join(folder, 'count.csv')) as f:
        rows = list(csv.reader(f))

    errors = []
    for row in rows[1:]:                                # the first row is the header
        if len(row) < 2:
            continue
        name, true_value = row[0].strip(), float(row[1])
        try:
            img = load_image(os.path.join(folder, 'images', name))
            img = cv2.resize(img, (1280, 720))          # pixel thresholds assume this size
            predicted = sum(value for value, x, y, w, h in find_objects(img))
        except Exception:
            predicted = 0                               # an unreadable image must not stop the run
        errors.append(abs(predicted - true_value))

    print(sum(errors) / len(errors))


if __name__ == '__main__':
    main()
