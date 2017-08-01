# Import necessary packages
import cv2
import numpy as np
import glob
import sys
import os
import csv
import pickle
import itertools

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


	def load_current_image(self, image_path):
		"""
		Loads an image and all marks found in the marks dict for it

		Arguments:
		input_folder -- path to folder with all images
		image_path -- path to the current image to be displayed
		"""
		try:
			# Load current image
			self.current_image = cv2.imread('%s' % image_path)
			# Draw labels for current image
			if image_path in self.marks_dict:
				for mark in self.marks_dict[image_path]:
					# Draw marks
					x1,y1,x2,y2 = mark
					cv2.rectangle(self.current_image, (x1,y1), (x2,y2), (0,255,0), 1)
			
			# Show current image
			cv2.imshow("Current Image", self.current_image)
		except:
			print "error: " + image_path



	def draw_region(self, event, x, y, flags, param):
		"""
		Drawing event, used to mark all objects in an image.
		Click, drag and release to mark an object.

		Arguments:
		event -- which type of event is recorded, lbuttonddown or lbuttonup
		x -- x coordinate of the click in an image
		y -- y coordinate of the click in an image
		"""
		# if the left mouse was clicked, record the starting (x,y) coordinates
		if event == cv2.EVENT_LBUTTONDOWN:
			self.draw_points = [x,y]
			print self.draw_points
		# check to see if the mouse was released
		elif event == cv2.EVENT_LBUTTONUP:
			# record the ending (x,y) coordinates
			self.draw_points.extend([x,y])
			print self.draw_points
			# draw a rectangle around the region of interest
			x1,y1,x2,y2 = self.draw_points

			cv2.rectangle(self.current_image, (x1,y1), (x2,y2), (0,255,0), 1)
			cv2.imshow("Current Image", self.current_image)
			# save mark in marks directory
			print self.current_index
			print self.marks_dict.keys()[ self.current_index ]
			print self.marks_dict[ self.marks_dict.keys()[ self.current_index ] ]
			self.marks_dict[ self.marks_dict.keys()[ self.current_index ] ].append(self.draw_points)


	def generate_KITTI_labels(self, input_list, output_folder):
		"""
		Iterate all files and creates a KITTI format label files
		in a way supported by Digits for object detection.

		Arguments:
		input_list -- list of all image files
		output_folder -- folder where all KITTI files will be saved
		"""		
		print "Generating KITTI format marks..."
		for filename, list_marks in self.marks_dict.iteritems():
			output_file = os.path.splitext(filename)[0]+".txt"
			with open(output_folder + "/" + output_file, "w") as text_file:
				for label in list_marks:
					x1,y1,x2,y2 = label
					text_file.write("aislador 0 0 0 %s %s %s %s 0 0 0 0 0 0 0\n" % (x1, y1, x2, y2)) 
		print "Done!"


	def find_next_image_without_marks(self, input_folder):
		"""
		Finds the next image without any marks in the marks dictionary

		Arguments:
		input_folder -- path to folder where all images are stored
		"""		
		for i, (filename, list_labels) in enumerate(self.marks_dict.iteritems()):
			if not list_labels:
				self.current_index = i
				self.load_current_image(filename)
				break


	def main(self, input_folder, output_folder):
		# Load input folder
		self.input_list = self.read_input_folder(input_folder)

		# Setup output folder
		self.setup_output_folder(output_folder)

		# Load image marks from input directory
		self.marks_dict = self.load_saved_marks(input_folder, self.input_list)

		cv2.namedWindow("Current Image")
		cv2.setMouseCallback("Current Image", self.draw_region)


		while True:
			# display current image
			self.load_current_image(self.marks_dict.keys()[self.current_index])

			# wait and get a keypress
			key = cv2.waitKey(1) & 0xFF

			# if "a" is pressed, move left on images list
			if key == ord("a"):
				self.current_index -= 1
            
			# if "d" is pressed, move right on images list
			elif key == ord("d"):
				self.current_index += 1

			elif key == ord("g"):
				self.generate_KITTI_labels(self.input_list, output_folder)

			elif key == ord("s"):
				self.find_next_image_without_marks(input_folder)

			elif key == ord("r"):
				try:
					self.marks_dict[self.marks_dict.keys()[self.current_index]].pop()
				except IndexError:
					print "There are no marks in this image" 
			# if "q" is pressed, close the script         
			elif key == ord("q"):  	
				self.save_marks(input_folder, self.marks_dict)
				break

			# always save the marks in file
			self.save_marks(input_folder, self.marks_dict)


if __name__ == '__main__':
	x = ImageMarker()
	x.main(sys.argv[1], sys.argv[2])