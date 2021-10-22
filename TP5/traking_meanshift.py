import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class staticROI(object):
    def __init__(self, type):
        self.type = type
        self.capture = cv.VideoCapture(0)
        # Bounding box reference points and boolean if we are extracting coordinates
        self.image_coordinates = []
        self.extract = False
        self.selected_ROI = False
        self.cropped = False
        self.term_crit = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 1)
        self.update()

    def update(self):

        while True:
            if self.capture.isOpened():
                # Read frame
                (self.status, self.frame) = self.capture.read()
                cv.imshow('image', self.frame)
                key = cv.waitKey(60)

                # Crop image
                if key == ord('c'):
                    self.clone = self.frame.copy()
                    cv.namedWindow('image')
                    cv.setMouseCallback('image', self.extract_coordinates)
                    while True:
                        key = cv.waitKey(2)
                        cv.imshow('image', self.clone)

                        # Crop and display cropped image
                        if key == ord('c'):
                            self.crop_ROI()
                            roi = self.cropped_image
                            hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
                            mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                            roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                            cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
                            self.cropped = True
                        # Resume video
                        if key == ord('r'):
                            break

                if self.cropped:

                    hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
                    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                    if self.type == "meanshift":
                        ret, self.track_window = cv.meanShift(dst, self.track_window, self.term_crit)
                        x, y, w, h = self.track_window
                        dst = cv.rectangle(dst, (x, y), (x + w, y + h), 255, 2)
                        # cv.imshow('Seguimiento', img2)
                        cv.imshow('Seguimiento', dst)
                    else:
                        ret, self.track_window = cv.CamShift(dst, self.track_window, self.term_crit)
                        x, y, w, h = self.track_window
                        dst = cv.rectangle(dst, (x, y), (x + w, y + h), 255, 2)
                        cv.imshow('Seguimiento', dst)

                # Close program with keyboard 'q'
                if key == ord('q'):
                    cv.destroyAllWindows()
                    exit(1)
            else:
                pass

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x, y)]
            self.extract = True

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv.EVENT_LBUTTONUP:
            self.image_coordinates.append((x, y))
            self.extract = False

            self.selected_ROI = True

            # Draw rectangle around ROI
            cv.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0, 255, 0), 2)

        # Clear drawing boxes on right mouse button click
        elif event == cv.EVENT_RBUTTONDOWN:
            self.clone = self.frame.copy()
            self.selected_ROI = False

    def crop_ROI(self):
        if self.selected_ROI:
            self.cropped_image = self.frame.copy()

            x1 = self.image_coordinates[0][0]
            y1 = self.image_coordinates[0][1]
            x2 = self.image_coordinates[1][0]
            y2 = self.image_coordinates[1][1]
            self.cropped_image = self.cropped_image[y1:y2, x1:x2]
            self.track_window = (x1, y1, x2-x1, y2-y1)

            print('Cropped image: {} {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
        else:
            print('Select ROI to crop before cropping')

    def show_cropped_ROI(self):
        cv.imshow('cropped image', self.cropped_image)

static_ROI = staticROI("meanshift")