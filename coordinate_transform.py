import pickle
import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio

import argparse

parser = argparse.ArgumentParser(description='Arguments for coordinate transform.')
parser.add_argument('--scene',type=int, default = 1, help='data scene number')
parser.add_argument('--direction',type=str, default ='front', help='front data for True and rear data for False')
args = parser.parse_args()


def conversion_x_axis(centre, width, stereo = False):
	if stereo:
		x_min = 0
		x_max = 1280

		cart_x_min = -33 #200 #0 rear
		cart_x_max = 33 #300 #180 rear 
		#centre = x_max - centre #scaled to account for azimuth in polar
	else:
		x_min = 0
		x_max = 1024

		cart_x_min = 0 #rear
		cart_x_max = 180 #rear 

	x_point = (centre/(x_max - x_min))* (cart_x_max - cart_x_min) + cart_x_min

	# if not stereo:
	# 	x_point = cart_x_max - (centre/(x_max - x_min))* (cart_x_max - cart_x_min)

	return x_point

	#depending on the object decide the range for x_point

def conversion_y_axis(centre, height, stereo = False):
	if stereo:
		y_min = 200
		y_max = 600
		cart_y_min = 0
		cart_y_max = 250
	else:
		y_min = 300
		y_max = 600
		cart_y_min = 250
		cart_y_max = 400

	if stereo:
		centre -= y_min #adjusting based on y_min
		y_point = (centre/(y_max - y_min))*(cart_y_max - cart_y_min) + cart_y_min

	else:
		#centre -= y_min
		#centre -=  y_max -  centre
		centre = y_max - centre
		y_point = (centre/(y_max - y_min))*(cart_y_max - cart_y_min) + cart_y_min - 50
	
	#print(centre, (centre/(y_max - y_min))*(cart_y_max - cart_y_min), y_point)

	return y_point

def object_area(height, width):
	return height*width

def draw_on_radar(x_points, y_points, radar):
	#radar = np.reshape(radar, (radar.shape[0], radar.shape[1], 1))
	overlay = radar.copy()
	for x_point, y_point in zip(x_points, y_points):
		#print(x_point, y_point, overlay.shape)

		x,y,w,h = int(x_point), int(y_point), 5, 5
		cv2.rectangle(overlay, (x-w, y-h), (x+w, y+h), (255, 255, 255), -1)  # A filled rectangle
		alpha = 0.2
		cv2.addWeighted(overlay, alpha, radar, 1 - alpha, 0, radar)
		#break
	# cv2.imshow("image_new", radar)
	# cv2.waitKey(0)
		#return None
	return radar


def closest_frame_time(lst, K): 
      
     lst = np.asarray(lst) 
     lst = lst[lst < K]
     idx = (np.abs(lst - K)).argmin() 
     return lst[idx] 

def previous_frame_time(lst, K):
	lst = np.asarray(lst)
	K_req = K - 180000 #0.07 seconds before this frame
	lst = lst[lst < K_req]
	idx = (np.abs(lst - K_req)).argmin() 
	return lst[idx] 


home_dir='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/scene'+str(args.scene)+'/'
data_dir='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/'
radar_dir=home_dir+'radar/radar-cart-img'
camera_dir=home_dir+args.direction+'/'
object_detection_result=data_dir+'object_detection/'+'predictions-scene-'+str(args.scene)+'-'+args.direction+'.pickle'

if args.direction=='stereo':
    stereo=True
    saveDir=home_dir+'radar-x-points-centre-left18Close/'
else:
    stereo=False
    saveDir=home_dir+'radar-x-points-rear-left18Close/'

if not os.path.isdir(saveDir):
    os.mkdir(saveDir)




#load the prediction pickle file 
with open(object_detection_result, 'rb') as f:
  predictions = pickle.load(f) 

pred_classes = predictions[0]
pred_boxes = predictions[1]
image_name = predictions[2]

#print(pred_classes, pred_boxes, image_name)

#radar_dir = '/Users/ms75986/Desktop/ResearchSpring2020/Qualcomm/Oxford-v2/scene1/radar/radar-cart-img/'
#save_dir = '/Users/ms75986/Desktop/ResearchSpring2020/Qualcomm/Oxford-v2/scene1/'
#camera_dir = '/Users/ms75986/Desktop/ResearchSpring2020/Qualcomm/Oxford-v2/scene1/mono_rear/'
camera_images = []
for name in image_name:
	camera_images.append(int(name[-20:-4]))
	camera_images.sort()



# match camera images to radar images
radar_path = os.path.join(radar_dir,'*png')
files = sorted(glob.glob(radar_path))
#saveDir = save_dir + 'radar-x-points-centre-left18Close/'
#os.mkdir(saveDir)
#stereo = False
for num, images in enumerate(files):
	if num < 1:
		continue
	radar_time = images[-20:-4]
	camera_time = previous_frame_time(camera_images, int(radar_time))
	print(int(radar_time)%10000000, camera_time%10000000)
	print("centre", radar_time, camera_time)
	#continue


	indices = [i for i, s in enumerate(image_name) if str(camera_time) in s]
	#print(image_name[indices[0]])
	# for i,s in enumerate(image_name):
	# 	if str(camera_time)  == s[-20:-4]:
	# 		print(i)
	# 		print(s)
	# print(indices)

	boxes = pred_boxes[indices[0]]
	#print(boxes)
	pred_class = pred_classes[indices[0]]
	x_points = []
	y_points = []
	object_type = []
	objects = [0,1,2,5,7]  #filter out person, car, truck class #1 bicycle included in scene 

	x1 = [5,7] #bus, truck
	x2 = [2] #car
	x3 = [0,1] #bicycle, pedestrian
	for num, box in enumerate(boxes):
		if pred_class[num] in objects:
			x_points.append(conversion_x_axis(box[0], box[2], stereo))
			y_points.append(conversion_y_axis(box[1], box[3], stereo))
			if pred_class[num] in x1:
				object_type.append(1)
			elif pred_class[num] in x2:
				object_type.append(2)
			else:
				object_type.append(3)

			print(box[0], conversion_x_axis(box[0], box[2], stereo), pred_class[num])
			#print("box[0], xpoint, and box[1]. ypoint:", box[0], conversion_x_axis(box[0], box[2], stereo),\
			 #box[1], conversion_y_axis(box[1], box[3], stereo), camera_images[indices[0]])
			#print("pred_class, area:", pred_class[num], object_area(box[2] - box[0], box[3] - box[1]))




	im = cv2.imread(images, cv2.IMREAD_GRAYSCALE)

	radar = draw_on_radar(x_points, y_points, im)

	print(camera_dir+str(camera_time)+'.png', radar_time)
	'''cam = cv2.imread(camera_dir+str(camera_time)+'.png', cv2.IMREAD_GRAYSCALE)
	fig, axs = plt.subplots(2)
	axs[0].imshow(radar, cmap='gray', vmin=0, vmax=255)
	axs[1].imshow(cam, cmap='gray', vmin=0, vmax=255)
	plt.show()'''
	print(x_points, object_type)
	saveName = saveDir + radar_time + '.png'
	#plt.savefig(saveName)

	#save x-point list for further azimuth processing
	mat_name = saveDir + radar_time + '.mat'
	sio.savemat(mat_name, {'x_point_centre': x_points, 'object_type': object_type})



	#break

