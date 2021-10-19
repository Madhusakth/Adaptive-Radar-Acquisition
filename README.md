# Adaptive-Radar-Acquisition
Adaptive automotive radar data acquisition 


This code runs object detection on images and stores classes and bounding boxes as .pickle file. 
To run object detection code, activate virtual environment: source detectron2-venv/bin/activate
"python3 object_detection.py --scene=1 --direction=front" 
In the current setup, we have 3 scenes with images from front and rear. 

For coordinate transform from image to radar, 
"python3 coordinate_transform.py --scene=1 --direction=front" 
 
