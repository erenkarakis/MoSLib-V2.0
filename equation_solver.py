import numpy as np 
import cv2
import yaml

def mouse_on_click(event,x,y,flags,param):
  global on_click_x, on_click_y, distance_in_cm, disparity_frame
  if event == cv2.EVENT_LBUTTONDOWN:
    on_click_x, on_click_y = x, y

    distance_in_cm = input("Enter the distance in cm: ")

    # Get the disparity at clicked location
    disparity_at_location = disparity_frame[on_click_x, on_click_y]
    print(on_click_x, on_click_y)
    print(disparity_at_location)

# Check for left and right camera IDs
# These values can change depending on the system
CamL_id = 4 # Camera ID for left camera
CamR_id = 2 # Camera ID for right camera
 
CamL= cv2.VideoCapture(CamL_id)
CamR= cv2.VideoCapture(CamR_id)

measurements = np.eye(10, 2)
 
# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("calibrated_params.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

stereo = cv2.StereoBM_create()
 
while True:
 
  # Capturing and storing left and right camera images
  retL, imgL= CamL.read()
  retR, imgR= CamR.read()
   
  # Proceed only if the frames have been captured
  if retL and retR:
    imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
 
    # Applying stereo image rectification on the left image
    Left_nice= cv2.remap(imgL_gray,
              Left_Stereo_Map_x,
              Left_Stereo_Map_y,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)
     
    # Applying stereo image rectification on the right image
    Right_nice= cv2.remap(imgR_gray,
              Right_Stereo_Map_x,
              Right_Stereo_Map_y,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)
    
    # Getting all the calibration values from .yaml file
    with open('block_matching_calibration.yaml', 'r') as f:
      block_matching_calibration = yaml.safe_load(f)

    numDisparities = block_matching_calibration['numDisparities'] * 16
    blockSize = (block_matching_calibration['blockSize'] * 2) + 5
    preFilterType = block_matching_calibration['preFilterType']
    preFilterSize = (block_matching_calibration['preFilterSize'] * 2) + 5
    preFilterCap = block_matching_calibration['preFilterCap']
    textureThreshold = block_matching_calibration['textureThreshold']
    uniquenessRatio = block_matching_calibration['uniquenessRatio']
    speckleRange = block_matching_calibration['speckleRange']
    speckleWindowSize = block_matching_calibration['speckleWindowSize'] * 2
    disp12MaxDiff = block_matching_calibration['disp12MaxDiff']
    minDisparity = block_matching_calibration['minDisparity']
    
    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # Calculating disparity using the StereoBM algorithm
    disparity_frame = stereo.compute(imgL_gray,imgR_gray)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.

    # Converting to float32 
    disparity_frame = disparity_frame.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity_frame = (disparity_frame/16.0 - minDisparity)/numDisparities
 
    # Displaying the disparity map
    cv2.imshow("disp",disparity_frame)

    # Get the clicked coordinate for getting disparity at that location
    cv2.setMouseCallback("disp", mouse_on_click)


    # Close window using q key
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break