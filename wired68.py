#!/usr/bin/python3
"""
Test detection of face construction with 68 point predictor
Author: Mikołaj Machowski, 2019
License: MIT
"""

import sys
import dlib
from PIL import Image, ImageDraw

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Split marks into separate features as per facial_landmarks_68markup.jpg
# Tuple stores: start point, end point, color of line, type of line
# We are getting to coordinates through range() so we need to get +1
# to end point.
FEATS = {
        'right_brow': (17, 22, 'black', 'open'),
        'left_brow': (22, 27, 'black', 'open'),
        'right_eye': (36, 42, 'pink', 'close'),
        'left_eye': (42, 48, 'pink', 'close'),
        'face_oval': (0, 17, 'pink', 'open'),
        'nose_bridge': (27, 32, 'pink', 'open'),
        'nose_base': (30, 36, 'pink', 'close'),
        'mouth_in': (60, 68, 'red', 'close'),
        'mouth_out': (48, 60, 'red', 'close')
        }

def draw_shapes(canvas, shapes):
    """
    Draw wired mask created from 68 landmarks.
    """

    for feature in FEATS:
        coords = []
        for i in range(FEATS[feature][0], FEATS[feature][1]):
            coords.append((shapes.part(i).x, shapes.part(i).y))

        # We want to close some shapes and .polygon isn't good enough
        if FEATS[feature][3] == 'close':
            coords.append(coords[0])

        canvas.line(coords, fill=FEATS[feature][2], width=4, joint='curve')

    return canvas

def main():
    """
    Main function processing images
    """

    if len(sys.argv) < 2:
        raise SystemExit('Usage: Not enough arguments')


    for fname in sys.argv[1:]:
        img = dlib.load_rgb_image(fname)
        # We use source image only as source - whole drawing is done on blank
        # out_img
        in_img = Image.open(fname)
        out_img = Image.new('RGB', (in_img.width, in_img.height), 'white')
        # We will paste both images to final one to see them side by side
        final_img = Image.new('RGB', (in_img.width*2, in_img.height), 'white')

        # Prepare draw operations 
        draw = ImageDraw.Draw(out_img)

        # Ask the detector to find the bounding boxes of each face.
        dets = DETECTOR(img, 1)
        for k, d in enumerate(dets):
            # Get the landmarks/parts for the face in box d.
            shape = PREDICTOR(img, d)
            # Draw wired contour on blank 
            draw = draw_shapes(draw, shape)

        # Paste our original image and wire mask of 68 points onto background
        final_img.paste(in_img)
        final_img.paste(out_img, (in_img.width, 0))

        # Save output in separate directory
        fname = fname.replace('dane/', '')
        final_img.save(f'dane68/{fname}-68.jpg', 'JPEG')

if __name__ == '__main__':
    main()
