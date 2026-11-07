"""Write step-by-step images of the banana_counter.py pipeline for every screenshot.

Usage:
    python visualize.py                          all images in data/ -> examples/
    python visualize.py data --out examples      explicit dataset and output folders
    python visualize.py --only image_9.jpg       selected images only

For each image a folder <out>/<image name>/ is written:
    0_overview.jpg         all steps on one sheet
    1_original.jpg
    2a_hsv_h.jpg, 2b_hsv_s.jpg, 2c_hsv_v.jpg
    3_yellow_mask.png      cv2.inRange
    4_closed_mask.png      after morphological closing
    5_contours.jpg         all contours
    6_hat_mask.png         pink mask and hat-sized regions
    7_surroundings.jpg     isolation rule and tail rule
    8_decisions.jpg        every contour with its decision
    9_detections.jpg       accepted objects and the total

The steps repeat banana_counter.find_objects with the same thresholds, and the accepted
objects are checked against it.
"""

import argparse
import csv
import os

import numpy as np
import cv2

import banana_counter as bc

# BGR colors of accepted objects: banana green, bunch blue, coin red
VALUE_COLOR = {1: (0, 255, 0), 5: (255, 160, 0), 10: (0, 0, 255)}
VALUE_NAME = {1: 'banana', 5: 'bunch', 10: 'coin'}
REJECT_ORDER = ['HUD', 'large', 'shape', 'color', 'crowded', 'tail', 'small']
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


def analyze(img):
    """Repeat bc.find_objects on a 1280x720 RGB image, keeping every intermediate result."""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv, bc.LOWER_YELLOW, bc.UPPER_YELLOW)
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bc.KERNEL)
    mask_pink = cv2.inRange(img_hsv, bc.LOWER_PINK, bc.UPPER_PINK)
    mask_hat = bc.hat_mask(img_hsv)
    contours, _ = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w / 2, y + h / 2
        area = cv2.contourArea(contour)
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        long_side, short_side = max(width, height), min(width, height)
        r = {'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'value': 0, 'box': None}

        if cx < 200 and (cy < 115 or 205 < cy < 275):
            r['decision'] = 'HUD'
        elif area < bc.MIN_AREA:
            r['decision'] = 'tiny'
        elif long_side > bc.MAX_SIDE:
            r['decision'] = 'large'
        else:
            ratio = short_side / long_side
            fill = area / (width * height)
            mean_h, mean_s, mean_v, _ = cv2.mean(img_hsv[y:y + h, x:x + w], mask=mask[y:y + h, x:x + w])
            if fill < bc.MIN_FILL:
                r['decision'] = 'shape'
            elif mean_s < bc.MIN_S or mean_v < bc.MIN_V:
                r['decision'] = 'color'
            else:
                x0, y0 = max(x - bc.PAD, 0), max(y - bc.PAD, 0)
                x1, y1 = min(x + w + bc.PAD, img.shape[1]), min(y + h + bc.PAD, img.shape[0])
                around = mask_close[y0:y1, x0:x1]
                own = np.zeros(around.shape, np.uint8)
                cv2.drawContours(own, [contour], -1, 255, -1, offset=(-x0, -y0))
                other_mask = cv2.subtract(around, own)
                other = int(np.sum(other_mask > 0))
                hat = int(np.sum(mask_hat[y0:y1, x0:x1] > 0))
                r.update(box=(x0, y0, x1, y1), own=own, other_mask=other_mask, other=other)
                if other > bc.MAX_OTHER * area:
                    r['decision'] = 'crowded'
                elif hat > 0:
                    r['decision'] = 'tail'
                elif fill >= bc.COIN_FILL and long_side >= bc.COIN_SIDE and ratio >= bc.COIN_RATIO:
                    r['decision'], r['value'] = 'coin', 10
                elif ratio >= bc.BUNCH_RATIO:
                    if short_side >= bc.BUNCH_SIDE:
                        r['decision'], r['value'] = 'bunch', 5
                    else:
                        r['decision'] = 'small'
                else:
                    r['decision'], r['value'] = 'banana', 1
        regions.append(r)

    # The images must show exactly what the script detects
    accepted = sorted((r['value'], r['x'], r['y'], r['w'], r['h']) for r in regions if r['value'] > 0)
    assert accepted == sorted(bc.find_objects(img)), 'visualization does not match find_objects'
    return {'hsv': img_hsv, 'mask': mask, 'mask_close': mask_close, 'mask_pink': mask_pink,
            'mask_hat': mask_hat, 'contours': contours, 'regions': regions}


def put(img, text, org, scale, color, thick):
    """Draw text with a black outline."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def with_header(img, lines, color):
    """Put the title in a black strip above the image, so it never covers an object."""
    header = np.zeros((10 + 26 * len(lines), img.shape[1], 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(header, line, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    return np.vstack([header, img])


def gray_to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_box(canvas, r):
    """Draw an accepted object with its value."""
    p0, p1 = (r['x'] - 3, r['y'] - 3), (r['x'] + r['w'] + 2, r['y'] + r['h'] + 2)
    color = VALUE_COLOR[r['value']]
    cv2.rectangle(canvas, p0, p1, color, 2)
    put(canvas, str(r['value']), (p0[0], max(p0[1] - 4, 12)), 0.5, color, 2)


def step_images(name, true_text, img):
    """Return a list of (file name, short title, image, header lines, header color) and the prediction."""
    a = analyze(img)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    regions = a['regions']
    predicted = sum(r['value'] for r in regions)
    steps = []

    steps.append(('1_original.jpg', '1. original', bgr.copy(),
                  ['1. original: %s, true value %s' % (name, true_text)], WHITE))

    # H is 0-179, stretched to 0-255 for display
    h_view = cv2.convertScaleAbs(a['hsv'][:, :, 0], alpha=255.0 / 179.0)
    steps.append(('2a_hsv_h.jpg', '2a. HSV: H', gray_to_bgr(h_view),
                  ['2a. HSV: H channel (hue); yellow is H from %d to %d' % (bc.LOWER_YELLOW[0], bc.UPPER_YELLOW[0])], YELLOW))
    steps.append(('2b_hsv_s.jpg', '2b. HSV: S', gray_to_bgr(a['hsv'][:, :, 1]),
                  ['2b. HSV: S channel (saturation); the mask needs S >= %d' % bc.LOWER_YELLOW[1]], YELLOW))
    steps.append(('2c_hsv_v.jpg', '2c. HSV: V', gray_to_bgr(a['hsv'][:, :, 2]),
                  ['2c. HSV: V channel (brightness); the mask needs V >= %d' % bc.LOWER_YELLOW[2]], YELLOW))

    share = 100.0 * np.mean(a['mask'] > 0)
    steps.append(('3_yellow_mask.png', '3. yellow mask', gray_to_bgr(a['mask']),
                  ['3. yellow mask (cv2.inRange): %.2f%% of the image pixels' % share], YELLOW))

    raw, _ = cv2.findContours(a['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    steps.append(('4_closed_mask.png', '4. after closing', gray_to_bgr(a['mask_close']),
                  ['4. after 5x5 closing: %d regions (%d before closing)' % (len(a['contours']), len(raw))], YELLOW))

    canvas = bgr.copy()
    cv2.drawContours(canvas, a['contours'], -1, (0, 0, 255), 2)
    steps.append(('5_contours.jpg', '5. contours', canvas,
                  ['5. contours (cv2.findContours): %d regions' % len(a['contours'])], WHITE))

    canvas = np.zeros_like(bgr)
    canvas[a['mask_pink'] > 0] = (150, 60, 150)
    canvas[a['mask_hat'] > 0] = (255, 255, 255)
    steps.append(('6_hat_mask.png', '6. hat mask', canvas,
                  ['6. pink mask (purple) and hat-sized regions (white):',
                   'area >= %d and long side < %d' % (bc.MIN_AREA, bc.HAT_SIDE)], YELLOW))

    # Surroundings: only regions that passed the HUD, size, shape and color filters
    canvas = cv2.convertScaleAbs(bgr, alpha=0.35)
    reached = [r for r in regions if r['box'] is not None]
    for r in reached:
        x0, y0, x1, y1 = r['box']
        part = canvas[y0:y1, x0:x1]
        part[r['own'] > 0] = bgr[y0:y1, x0:x1][r['own'] > 0]             # the region in its original colors
    for r in reached:
        x0, y0, x1, y1 = r['box']
        part = canvas[y0:y1, x0:x1]
        part[r['other_mask'] > 0] = (0, 0, 255)                          # other yellow pixels: red
        part[a['mask_hat'][y0:y1, x0:x1] > 0] = (255, 0, 255)            # hat: magenta
    for r in reached:
        x0, y0, x1, y1 = r['box']
        color = {'crowded': (0, 0, 255), 'tail': (255, 0, 255)}.get(r['decision'], (0, 255, 0))
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, 1)
        put(canvas, '%d%%' % round(100.0 * r['other'] / r['area']), (x0, max(y0 - 4, 12)), 0.4, color, 1)
    steps.append(('7_surroundings.jpg', '7. surroundings', canvas,
                  ['7. surroundings (%d px around the region): other yellow pixels (red), hat (magenta)' % bc.PAD,
                   'number = other yellow pixels / region area; rejected above %d%%' % round(100 * bc.MAX_OTHER)], WHITE))

    canvas = bgr.copy()
    counts = {}
    for r in regions:
        counts[r['decision']] = counts.get(r['decision'], 0) + 1
        if r['decision'] == 'tiny':
            continue
        if r['value'] > 0:
            draw_box(canvas, r)
        else:
            p0, p1 = (r['x'] - 3, r['y'] - 3), (r['x'] + r['w'] + 2, r['y'] + r['h'] + 2)
            cv2.rectangle(canvas, p0, p1, WHITE, 1)
            put(canvas, r['decision'], (p0[0], max(p0[1] - 4, 12)), 0.4, WHITE, 1)
    rejected = ', '.join('%s %d' % (k, counts[k]) for k in REJECT_ORDER if k in counts)
    accepted = ', '.join('%s %d' % (VALUE_NAME[v], counts[VALUE_NAME[v]]) for v in (1, 5, 10) if VALUE_NAME[v] in counts)
    steps.append(('8_decisions.jpg', '8. decisions', canvas,
                  ['8. decisions: rejected (white): %s' % (rejected or 'none'),
                   'tiny (not drawn): %d;  accepted: %s' % (counts.get('tiny', 0), accepted or 'none')], WHITE))

    canvas = bgr.copy()
    for r in regions:
        if r['value'] > 0:
            draw_box(canvas, r)
    steps.append(('9_detections.jpg', '9. detections', canvas,
                  ['9. detections: %s  true=%s  predicted=%d' % (name, true_text, predicted)], WHITE))
    return steps, predicted


def overview(steps):
    """All steps scaled down, four per row."""
    tiles = []
    for file_name, title, image, lines, color in steps:
        strip = np.zeros((24, 480, 3), np.uint8)
        cv2.putText(strip, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        tiles.append(np.vstack([strip, cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)]))
    while len(tiles) % 4:
        tiles.append(np.zeros((294, 480, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])


def save(path, image):
    if path.endswith('.png'):
        cv2.imwrite(path, image)
    else:
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    parser = argparse.ArgumentParser(description='Write step-by-step images of the banana counter pipeline')
    parser.add_argument('data', nargs='?', default='data', help='dataset folder with images/ and count.csv')
    parser.add_argument('--out', default='examples', help='output folder')
    parser.add_argument('--only', default='', help='comma-separated image names to process')
    args = parser.parse_args()
    only = set(n for n in args.only.split(',') if n)

    with open(os.path.join(args.data, 'count.csv')) as f:
        rows = [row for row in csv.reader(f) if len(row) >= 2][1:]

    for row in rows:
        name, true_text = row[0].strip(), str(int(float(row[1])))
        if only and name not in only:
            continue
        img = cv2.imread(os.path.join(args.data, 'images', name))
        if img is None:
            print('%-16s cannot be read' % name)
            continue
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (1280, 720))
        steps, predicted = step_images(name, true_text, img)

        folder = os.path.join(args.out, os.path.splitext(name)[0])
        os.makedirs(folder, exist_ok=True)
        for file_name, title, image, lines, color in steps:
            save(os.path.join(folder, file_name), with_header(image, lines, color))
        save(os.path.join(folder, '0_overview.jpg'), overview(steps))
        print('%-16s true %-5s predicted %d' % (name, true_text, predicted))
    print('written to: %s' % args.out)


if __name__ == '__main__':
    main()
