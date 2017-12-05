#!/usr/bin/env python
# Import necessary packages
import cv2
import glob
import os
import pickle
import itertools
import argparse
import numpy as np
from os.path import basename, splitext

"""
Script created to mark images in KITTI style for Digits

Only type, truncated and bbox are used in Digits, all other fields are default to 0
"""


class ImageMarker:

    def __init__(self, input_folder, output_folder, labels, mode, label_format):
        self.input_folder = input_folder
        self.input_files = self.read_input_files(input_folder)
        self.output_folder = self.setup_output_folder(output_folder)
        self.labels = self.read_labels_file(labels)
        self.mode = mode
        self.label_format = label_format
        self.draw_points = []
        self.marks_dict = {}
        self.current_label = None
        self.current_label_index = 0
        self.current_image = None
        self.current_index = 0

    def read_input_files(self, input_folder):
        """
        Read all images inside a folder, and return a list of all suported files
        
        Arguments:
            input_folder -- path to folder with images

        Returns:
            files -- list of supported files found in input_folder
        """
        supported_file_types = ('*.jpg', '*.jpeg', '*.png')
        files = [glob.glob('%s/%s' % (input_folder, filetype)) for filetype in supported_file_types]
        files = list(itertools.chain.from_iterable(files))
        return files

    def setup_output_folder(self, output_folder):
        """
        Check if output folder exists, if not create one.

        Arguments:
            output_folder -- path where the KITTI format files for each image will be saved
        """
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return output_folder

    def load_marks_dict(self, input_folder, input_list, labels):
        """
        Loads a 'marks.pickle' file with previous saved marks if it exists,
        otherwise it creates an empty marks dictionary.
        
        Arguments:
            input_folder -- path where all images are stored
            input_list -- list of all images in input_folder
            labels -- list of labels

        Returns:
            marks_dict -- dictionary with all files and labels created for each one of them
        """
        marks_file = '%s%s' % (input_folder, '%s_marks.pickle' % self.mode)
        if os.path.isfile(marks_file):
            try:
                load_dict = open(marks_file, 'rb')
                tmp_dict = pickle.load(load_dict)
                load_dict.close()
                return tmp_dict
            except IOError:
                return {key: [] for key in input_list}
        else:
            return {input_path: {label: [] for label in labels} for input_path in input_list}

    def save_marks(self, input_folder, marks_dict):
        """
        Save marks on images to a pickle file.

        Arguments:
            input_folder -- input folder where the pickle file will be saved
            marks_dict -- marks file 
        """
        try:
            marks_file = '%s%s' % (input_folder, '%s_marks.pickle' % self.mode)
            with open(marks_file, 'wb') as mf:
                pickle.dump(marks_dict, mf)
        except IOError:
            print('Could not save marks directory')

    def load_current_image(self, current_index, marks_dict, current_label, mode):
        """
        Loads an image and all marks found in the marks dict for it

        Arguments:
            input_folder -- path to folder with all images
            image_path -- path to the current image to be displayed

        Returns:
            current_image -- numpy matrix for the current image loaded
            current_index -- index of the current image in the full list of images
        """

        keys = list(marks_dict.keys())
        current_index = 0 if current_index >= len(keys) else current_index
        # Load current image
        image_path = keys[current_index]
        current_image = cv2.imread('%s' % image_path)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(current_image, "Label: %s" % current_label, (10, 20), font, 0.6, (0, 0, 0), 2)
        # Draw labels for current image
        if image_path in marks_dict:
            for mark in marks_dict[image_path][current_label]:
                if mode == 'detection':
                    # Draw marks
                    x1, y1, x2, y2 = mark
                    self.draw_selection(current_image, x1, y1, x2, y2)
                elif mode == 'segmentation':
                    self.draw_polygon(current_image, mark)
        cv2.imshow("Current Image", current_image)
        return current_image, current_index

    def print_size_text(self, options, img, x1, y1, x2, y2):
        """
        Prints text on the image, next to the label created,
        if the label is smaller than 50x50 pixels, it's shown in Red.
        
        Arguments:
            options -- values to show or not the text
            img -- current image being drawn
            x1, y1, x2, y2 -- coordinates of the label created
        
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        text = "[%d,%d]" % (abs(x2 - x1), abs(y2 - y1))
        if options['size_warn'] and ((w >= 50) and (h >= 50)) and ((w <= 400) and (h <= 400)):
            cv2.putText(img, text, (x2, y2), font, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(img, text, (x2, y2), font, 0.6, (0, 0, 255), 2)

    def draw_selection(self, img, x1, y1, x2, y2):
        """
        Draws a label in an image
        
        Arguments:
            img -- current image being drawn
            x1, y1, x2, y2 -- coordinates of the label created
        """
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x2, y1), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    def draw_polygon(self, current_image, draw_points, close=False):
        """
        Draws all points in draw_points list in order.

            Arguments:
                current_image -- Current image shown on display
                draw_points -- list of pairs for each point in segmentation mode
        """
        for x1, y1 in draw_points:
            cv2.rectangle(current_image, (x1-2, y1-2), (x1+2, y1+2), (255, 0, 0), 2, cv2.LINE_AA)

        if len(draw_points) >= 2:
            if not close:
                cv2.polylines(current_image, np.int32([draw_points]), False, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.polylines(current_image, np.int32([draw_points]), True, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.fillPoly(current_image, np.int32([draw_points]), (0, 128, 0), cv2.LINE_AA)

    def mouse_segmentation(self, event, x, y, flags, param):
        """
        Drawing event, used to mark objects for segmenation in an image.
        Click each point in the polygon to label an object.

        Arguments:
            event -- which type of event is recorded
            x -- x coordinate of the click in an image
            y -- y coordinate of the click in an image

        """
        # if the left mouse is clicked, record the starting (x,y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw_points.append([x, y])
        # check to see if the mouse was released
        elif event == cv2.EVENT_LBUTTONUP:
            self.draw_polygon(self.current_image, self.draw_points)
            cv2.imshow("Current Image", self.current_image)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.draw_points) >= 3:
                self.draw_points.append(self.draw_points[0])
                self.draw_polygon(self.current_image, self.draw_points, close=True)
                cv2.imshow("Current Image", self.current_image)
                key = list(self.marks_dict)[self.current_index]
                self.marks_dict[key][self.current_label].append(self.draw_points)

    def mouse_detection(self, event, x, y, flags, param):
        """
        Drawing event, used to mark all objects in an image.
        Click, drag and release to mark an object.

        Arguments:
            event -- which type of event is recorded lbuttonddown, lbuttonup or mousemove
            x -- x coordinate of the click in an image
            y -- y coordinate of the click in an image
        """
        # if the left mouse was clicked, record the starting (x,y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw_points = [x, y]
        # check to see if the mouse was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x,y) coordinates
            self.draw_points.extend([x, y])
            # draw a rectangle around the region of interest
            x1, y1, x2, y2 = self.draw_points
            self.draw_selection(self.current_image, x1, y1, x2, y2)
            cv2.imshow("Current Image", self.current_image)
            # save mark in marks directory
            key = list(self.marks_dict)[self.current_index]
            self.marks_dict[key][self.current_label].append(self.draw_points)
            self.draw_points = []

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw_points:
                tmp_points = self.draw_points + [x, y]
                x1, y1, x2, y2 = tmp_points
                tmp_image = self.current_image.copy()
                self.draw_selection(tmp_image, x1, y1, x2, y2)
                cv2.imshow("Current Image", tmp_image)

    def generate_KITTI_labels(self, output_folder, marks_dict):
        """
        Iterate all files and creates a KITTI format label files
        in a way supported by Digits for object detection.

        Arguments:
            output_folder -- folder where all KITTI files will be saved
            marks_dict -- dictionary with all images marks
        """
        print("Generating KITTI format marks for detection...")
        for filename, labels in marks_dict.items():
            input_name = basename(filename)
            output_name = splitext(input_name)[0] + ".txt"
            with open(output_folder + "/" + output_name, "w") as text_file:
                for key, values in labels.items():
                    if values:
                        for x1, y1, x2, y2 in values:
                            text_file.write("%s 0 0 0 %s %s %s %s 0 0 0 0 0 0 0\n" % (key, x1, y1, x2, y2))
        print("Done!")

    def generate_VOC_labels(self, output_folder, marks_dict):
        pass

    def generate_IMAGE_labels(self, output_folder, marks_dict):
        """
        Iterate all files and creates IMAGE segmentation labels.

            Arguments:
                output_folder -- folder where all segmentation images will be saved
                marks_dict -- dictionary with all image marks
        """
        print("Generating IMAGE labels for segmentation...")
        for filename, labels in marks_dict.items():
            input_name = basename(filename)
            output_name = input_name
            output_image = np.zeros((640, 640))
            for key, values in labels.items():
                for points in values:
                    cv2.fillPoly(output_image, np.int32([points]), (255, 255, 255), cv2.LINE_AA)
                cv2.imwrite('%s/%s' % (output_folder, output_name), output_image)
        print("Done!")

    def find_next_image_without_marks(self, current_index, marks_dict, current_label, mode):
        """
        Finds the next image without any marks in the marks dictionary

        Arguments:
            current_index -- position of the current image shown
            marks_dict -- marks dictionary with filename and labels 
        """
        for i, (filename, labels) in enumerate(marks_dict.items()):
            if not labels[current_label]:
                current_index = i
                current_image, current_index = self.load_current_image(current_index, marks_dict, current_label, mode)
                return current_index, current_image

        # if all images have labels, do nothing
        print("All images have been labeled.")
        current_image, current_index = self.load_current_image(current_index, marks_dict, current_label, mode)
        return current_index, current_image

    def remove_last_mark_created(self, marks_dict, current_index, current_label):
        """
        Removes the last mark created in the current image being shown
        
        Arguments:
            marks_dict -- dictionary with all image paths and labels created
            current_index -- index of the current image shown
        """
        try:
            marks_dict[list(marks_dict.keys())[current_index]][current_label].pop()
            return marks_dict
        except IndexError:
            print("There are no marks in this image")
            return marks_dict

    def read_labels_file(self, labels_file_path):
        """
        Reads a file of labels to use,
        labels must be one line each.
        
        Arguments:
            labels_file_path -- path to a text file
        """
        with open(labels_file_path) as f:
            lines = f.read().splitlines()
            return lines

    def generate_labels(self, mode, label_format, output_folder, marks_dict):
        if mode == 'detection':
            if label_format == 'kitti':
                self.generate_KITTI_labels(output_folder, marks_dict)
            elif label_format == 'voc':
                self.generate_VOC_labels(output_folder, marks_dict)
            else:
                print("Not implemented: %s format in mode %s." % (label_format, mode))

        elif mode == 'segmentation':
            self.generate_IMAGE_labels(output_folder, marks_dict)

    def run(self):
        """
        All images are displayed one by one,
        the user manually labels them.
        """
        self.current_label = self.labels[self.current_label_index]

        self.marks_dict = self.load_marks_dict(self.input_folder, self.input_files, self.labels)

        cv2.namedWindow('Current Image')

        if self.mode == 'detection':
            cv2.setMouseCallback('Current Image', self.mouse_detection)
        elif self.mode == 'segmentation':
            cv2.setMouseCallback('Current Image', self.mouse_segmentation)

        key = ''
        while key != ord('q'):

            # Display current image
            self.current_image, self.current_index = self.load_current_image(self.current_index,
                                                                             self.marks_dict,
                                                                             self.current_label,
                                                                             self.mode)
            # Wait for user input
            key = cv2.waitKey() & 0xFF

            # If 'a' is pressed, go left on the images list.
            if key == ord("a"):
                self.current_index -= 1
                if self.mode == 'segmentation':
                    self.draw_points = []

            # If 'd' is pressed, go right on the images list.
            elif key == ord("d"):
                self.current_index += 1
                if self.mode == 'segmentation':
                    self.draw_points = []

            # If 'g' is pressed, generate labels in format specified.
            elif key == ord("g"):
                self.generate_labels(self.mode, self.label_format, self.output_folder, self.marks_dict)

            # If 's' is pressed, find the next image in the list without marks.
            elif key == ord("s"):
                self.current_index, self.current_image = self.find_next_image_without_marks(self.current_index,
                                                                                            self.marks_dict,
                                                                                            self.current_label,
                                                                                            self.mode)
            # If 'r' is pressed, delete last mark created on the current image.
            elif key == ord("r"):
                self.marks_dict = self.remove_last_mark_created(self.marks_dict,
                                                                self.current_index,
                                                                self.current_label)

            # If '1' is pressed, cycle among all labels in the file.
            elif key == ord("1"):
                self.current_label_index += 1
                if self.current_label_index >= len(self.labels):
                    self.current_label_index = 0
                self.current_label = self.labels[self.current_label_index]

            # If "q" is pressed, close the script
            elif key == ord("q"):
                self.save_marks(self.input_folder, self.marks_dict)

            # always save the marks in file
            self.save_marks(self.input_folder, self.marks_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image labeling tool.')
    parser.add_argument('--input', '-in', dest='input_folder', required=True,
                        help='Input folder with images.')
    parser.add_argument('--output', '-out', dest='output_folder', default='output/',
                        help='Output folder where labels will be saved.')
    parser.add_argument('--labels', '-l', required=True,
                        help='Labels file with a list of all objects to label.')
    parser.add_argument('--mode', '-m', default='detection',
                        choices=['detection', 'segmentation'],
                        help='Labeling mode to use.')
    parser.add_argument('--label_format', '-lf', default='kitti',
                        choices=['kitti', 'voc'],
                        help='Label format for output files.')

    args = parser.parse_args()

    tool = ImageMarker(args.input_folder,
                       args.output_folder,
                       args.labels,
                       args.mode,
                       args.label_format)
    tool.run()
