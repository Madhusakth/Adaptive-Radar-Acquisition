# Adaptive-Radar-Acquisition
Adaptive automotive radar data acqusition using object detection

Please check the paper for detailed understanding of the algorithm: https://arxiv.org/pdf/2010.02367.pdf


This code runs object detection on images and stores classes and bounding boxes as .pickle file. 
To run object detection code, activate virtual environment: source detectron2-venv/bin/activate
"python3 object_detection.py --scene=1 --direction=front" 
In the current setup, we have 3 scenes with images from front and rear. 

For coordinate transform from image to radar, 
"python3 coordinate_transform.py --scene=1 --direction=front"

The src_cs folder consists of the matlab code to run compressed sensing 
Run the following command for baseline uniform sampling reconstruction using CS

"matlab -nodesktop -nosplash -r "meas='BPD';run compressive_sensing_radar_v1(2,meas)""  
 
