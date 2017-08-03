# Import necessary packages
import cv2
import glob
import sys
import os
import pickle
import itertools
from os.path import basename, splitext

"""
Script created to mark images in KITTI style for Digits

Only type, truncated and bbox are used in Digits, all other fields are default to 0
"""

class ImageMarker():


    def __init__(self):
        self.marks_dict = {}
        self.current_image = None
        self.current_index = 0
        self.input_list = []
        self.draw_points = []
        self.options = { 'show_size': True, 'size_warn': True }


    def read_input_folder(self, input_folder):
        """
        Read all images inside a folder, and return a list of all suported files
        
        Arguments:
            input_folder -- path to folder with images
        """
        supported_file_types = ('*.jpg','*.jpeg','*.png')
        files = [ glob.glob('%s/%s' % ( input_folder, filetype )) for filetype in supported_file_types ]
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


    def load_saved_marks(self, input_folder, input_list):
        """
        Loads a 'marks.pickle' file with all saved marks from previous works

        Arguments:
            input_folder -- folder where the images and 'marks.pickle' are stored
        """
        marks_file = '%s%s' % (input_folder, 'marks.pickle')
        try:
            load_dict = open(marks_file, 'rb')
            tmp_dict = pickle.load(load_dict)
            load_dict.close()
            return tmp_dict
        except IOError:
            return { key:[] for key in input_list}


    def save_marks(self, input_folder, marks_dict):
        """
        Save marks on images to a pickle file.

        Arguments:
            input_folder -- input folder where the pickle file will be saved
            marks_dict -- marks file 
        """
        marks_file = '%s%s' % (input_folder, 'marks.pickle')
        save_dict = open(marks_file, 'wb')
        pickle.dump(marks_dict, save_dict)
        save_dict.close()


    def load_current_image(self, current_index, marks_dict):
        """
        Loads an image and all marks found in the marks dict for it

        Arguments:
            input_folder -- path to folder with all images
            image_path -- path to the current image to be displayed
        """
        
        keys = list(marks_dict.keys())
        current_index = 0 if current_index >= len(keys) else current_index
        # Load current image
        image_path = keys[current_index]
        current_image = cv2.imread('%s' % image_path)
        # Draw labels for current image
        if image_path in marks_dict:
            for mark in marks_dict[image_path]:
                # Draw marks
                x1,y1,x2,y2 = mark
                self.draw_selection(current_image, x1, y1, x2, y2)
                if self.options['show_size']:
                    self.print_size_text(self.options, current_image, x1, y1, x2, y2)
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
        w = abs(x2-x1)
        h = abs(y2-y1)
        text = "[%d,%d]" % (abs(x2-x1), abs(y2-y1))
        if options['size_warn'] and ((w >= 50) and (h >= 50)) and ((w <= 400) and (h <= 400)):
            cv2.putText(img, text, (x2, y2), font, 0.6, (0,255,0), 2)
        else:
            cv2.putText(img, text, (x2, y2), font, 0.6, (0,0,255), 2)

    def draw_selection(self, img, x1, y1, x2, y2):
        """
        Draws a label in an image
        
        Arguments:
            img -- current image being drawn
            x1, y1, x2, y2 -- coordinates of the label created
        """
        cv2.line(img, (x1,y1), (x2,y2), (0,0,0), 2, cv2.LINE_AA)
        cv2.line(img, (x1,y2), (x2,y1), (0,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2, cv2.LINE_AA)
    

    def mouse_actions(self, event, x, y, flags, param):
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
            self.draw_points = [x,y]
        # check to see if the mouse was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x,y) coordinates
            self.draw_points.extend([x,y])
            # draw a rectangle around the region of interest
            x1,y1,x2,y2 = self.draw_points
            self.draw_selection(self.current_image, x1, y1, x2, y2)
            if self.options['show_size']:
                self.print_size_text(self.options, self.current_image, x1, y1, x2, y2)              
            cv2.imshow("Current Image", self.current_image)
            # save mark in marks directory
            self.marks_dict[ list(self.marks_dict.keys())[ self.current_index ] ].append(self.draw_points)
            self.draw_points = []

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw_points:
                tmp_points = self.draw_points + [x,y]
                x1,y1,x2,y2 = tmp_points
                tmp_image = self.current_image.copy()
                self.draw_selection(tmp_image, x1, y1, x2, y2)
                if self.options['show_size']:
                    self.print_size_text(self.options, tmp_image, x1, y1, x2, y2) 
                cv2.imshow("Current Image", tmp_image)



    def generate_KITTI_labels(self, input_list, output_folder, marks_dict):
        """
        Iterate all files and creates a KITTI format label files
        in a way supported by Digits for object detection.

        Arguments:
            input_list -- list of all image files
            output_folder -- folder where all KITTI files will be saved
        """        
        print ("Generating KITTI format marks...")
        for filename, list_marks in marks_dict.items():
            input_name = basename(filename)
            output_name = splitext(input_name)[0]+".txt"
            with open(output_folder + "/" + output_name, "w") as text_file:
                for label in list_marks:
                    x1,y1,x2,y2 = label
                    text_file.write("aislador 0 0 0 %s %s %s %s 0 0 0 0 0 0 0\n" % (x1, y1, x2, y2)) 
        print ("Done!")


    def find_next_image_without_marks(self, current_index, marks_dict):
        """
        Finds the next image without any marks in the marks dictionary

        Arguments:
            current_index -- position of the current image shown
            marks_dict -- marks dictionary with filename and labels 
        """    
        for i, (filename, list_labels) in enumerate(marks_dict.items()):
            if not list_labels:
                current_index = i
                current_image, current_index = self.load_current_image(current_index, marks_dict)
                return current_index, current_image
        
        # if all images have labels, do nothing
        print ("All images have been labeled.")
        current_image, current_index = self.load_current_image(current_index, marks_dict)
        return current_index, current_image


    def remove_last_mark_created(self, marks_dict, current_index):
        """
        Removes the last mark created in the current image being shown
        
        Arguments:
            marks_dict -- dictionary with all image paths and labels created
            current_index -- index of the current image shown
        """
        try:
            marks_dict[list(marks_dict.keys())[current_index]].pop()
            return marks_dict
        except IndexError:
            print ("There are no marks in this image" )
            return marks_dict


    def main(self, input_folder, output_folder):
        # Load input folder
        self.input_list = self.read_input_folder(input_folder)

        # Setup output folder
        self.setup_output_folder(output_folder)

        # Load image marks from input directory
        self.marks_dict = self.load_saved_marks(input_folder, self.input_list)

        cv2.namedWindow("Current Image")
        cv2.setMouseCallback("Current Image", self.mouse_actions)

        key = ''
        while key != ord("q"):
            # display current image
            self.current_image, self.current_index = self.load_current_image(self.current_index, self.marks_dict)

            # wait and get a keypress
            key = cv2.waitKey() & 0xFF

            # if "a" is pressed, move left on images list
            if key == ord("a"):
                self.current_index -= 1
            
            # if "d" is pressed, move right on images list
            elif key == ord("d"):
                self.current_index += 1

            elif key == ord("g"):
                self.generate_KITTI_labels(self.input_list, output_folder, self.marks_dict)

            elif key == ord("s"):
                self.current_index, self.current_image = self.find_next_image_without_marks(self.current_index, self.marks_dict)

            elif key == ord("r"):
                self.marks_dict = self.remove_last_mark_created(self.marks_dict, self.current_index)
            # if "q" is pressed, close the script         
            elif key == ord("q"):      
                self.save_marks(input_folder, self.marks_dict)
            elif key == ord("1"):
                self.options['show_size'] = not self.options['show_size']

            # always save the marks in file
            self.save_marks(input_folder, self.marks_dict)


if __name__ == '__main__':
    x = ImageMarker()
    x.main(sys.argv[1], sys.argv[2])