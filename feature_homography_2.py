#!/usr/bin/env python

'''

Baseline file feature_homography.py modified from opencv repository
Improvements and modifications include changing drawn rectangle into
circle and changing the colors of the tracker:
Selected rectangle outline: white rectangle to red circle
Tracker line: green to  blue
Tracker rectangle: green to magenta

Simultaneous video frame without tracker technology


References:
Original file: https://github.com/opencv/opencv/blob/master/samples/python/feature_homography.py
-----------------------------------------
ORIGINAL SOURCE DOCUMENTATION BELOW
-----------------------------------------
Feature homography
==================

Example of using features2d framework for interactive video homography matching.
ORB features and FLANN matcher are used. The actual tracking is implemented by
PlaneTracker class in plane_tracker.py

Inspired by http://www.youtube.com/watch?v=-ZNYoL8rzPY

video: http://www.youtube.com/watch?v=FirtmYcC0Vc

Usage
-----
feature_homography.py [<video source>]

Keys:
   SPACE  -  pause video

Select a textured planar object to track by drawing a box with a mouse.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import math
# local modules
import video
from video import presets
import common
from common import getsize, draw_keypoints
from plane_tracker import PlaneTracker


class App:
    def __init__(self, src):
        self.cap = video.create_capture(src, presets['book'])
        self.frame = None
        self.paused = False
        self.tracker = PlaneTracker()

        cv.namedWindow('plane')
        self.rect_sel = common.RectSelector('plane', self.on_rect)

    def on_rect(self, rect):
        self.tracker.clear()
        self.tracker.add_target(self.frame, rect)

    def run(self):
        while True:
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()

            blue = cv.cvtColor(frame, cv.COLOR_BGR2Luv)
            # Display the resulting frame
            cv.imshow('self.frame',blue)

            w, h = getsize(self.frame)
            vis = np.zeros((h, w*2, 3), np.uint8)
            vis[:h,:w] = self.frame
            if len(self.tracker.targets) > 0:
                target = self.tracker.targets[0]
                vis[:,w:] = target.image
                draw_keypoints(vis[:,w:], target.keypoints)
                x0, y0, x1, y1 = target.rect
                #cv.rectangle(vis, (x0+w, y0), (x1+w, y1), (255, 0, 0), 2)
                center_x = int((x0+x1+2*w)/2)
                center_y = int((y0+y1)/2)
                r_x = int(abs((x0 - x1)/2))
                r_y = int(abs((x0 - x1)/2))
                radius = int(math.sqrt((r_x *r_x)+ (r_y*r_y)))

                cv.circle(vis, (center_x, center_y),radius, (0, 0, 255),2)
            if playing:
                tracked = self.tracker.track(self.frame)
                if len(tracked) > 0:
                    tracked = tracked[0]
                    cv.polylines(vis, [np.int32(tracked.quad)], True, (255, 0, 255), 2)
                    for (x0, y0), (x1, y1) in zip(np.int32(tracked.p0), np.int32(tracked.p1)):
                        cv.line(vis, (x0+w, y0), (x1, y1), (255, 0, 0))
            draw_keypoints(vis, self.tracker.frame_points)

            self.rect_sel.draw(vis)
            cv.imshow('plane', vis)
            ch = cv.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == 27:
                break


if __name__ == '__main__':
    print(__doc__)

    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    App(video_src).run()
