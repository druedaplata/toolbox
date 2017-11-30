import os
import cv2
import sys

# Get input from user
video_source = sys.argv[1]
output_folder = sys.argv[2]
skip_frame = int(sys.argv[3])


# Get video source
video_capture = cv2.VideoCapture(video_source)
video_capture.set(cv2.CAP_PROP_FPS, 5)

# Create output folder
if not os.path.exists(sys.argv[2]):
	os.mkdir(output_folder)

video_path = os.path.join(sys.argv[2], sys.argv[1])
if not os.path.exists(video_path):
	os.mkdir(video_path)

count = 0

# Lets read the video
while(video_capture.isOpened()):
	ret, frame = video_capture.read()
	
	# increment counter
	count += 1

	# if frame exists
	if ret == True:
		# save current frame
		if count % skip_frame == 0:
			cv2.imwrite("%s/%s.jpg" % (video_path, count), frame)
	else:
		break

	# ESC to stop
	key = cv2.waitKey(7) % 0x100
	if key == 27:
		break

cv2.destroyAllWindows()

