'''Feature tracking and autofocus for ExM'''
''' 
07/29/2025 - Add functions to collect multiple FOV after each tracking step
12/10/2025 - updated with different z-scan range for each 0.5x expansion
Updated on 01/08/2026 - v11:
[done] 1. Add Ello beamblock function and save image with correct file name for each laser line. Laser will be switch at each 0.5x is met
[Not in use] 2. Add threshold for matching region - if <95% match for area and shape, scan 4 nearby location (identify move of xy in um) to find the match object 
[done] 3. Get user input instead of manually find and edit the code
Updated on 01/23/2026 - v12:
[done] 4. Change to adaptive binary threshold -  take average of image intensity 
[done] 5. Change object's area cutoff to 1000 for cleaner annotation
[done] 6. Update matching function for identifying same feature - Add normalization, change area to area_filled, and two more weighted factors (euler number, perimeter) 
Updated on 02/06/2026 -v13 final
1. Larger kernel size (ksize = 7) for Tenengrad to improve score '''

''' "Install OpenCV: Main modules package, Make sure that your pip version is up-to-date (19.3 is the minimum supported version): pip install --upgrade pip. Check version with pip -V."
# pip install opencv-python

# "Install pycromanager"
# pip install pycromanager

# install other library such as numpy, pandas


'''
import math
from typing import Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from time import sleep
from pycromanager import Core
import os
from time import time
from time import strftime
from time import localtime
import sys
import subprocess
import pandas as pd
from datetime import datetime
from skimage.measure import label, regionprops
import serial
import clr
# path for saving images
desktop_path = os.path.join(os.path.expanduser("~"), r"Desktop\TEMP_EXM\20260210")

'''Initiate Micromanager connection'''
'''Open Micro-Manager, select tools-options, and check the box that says Run server on port 4827 (you only need to do this once)
Check if everything is working by running below code, the output should be something like: <pycromanager.core.mmcorej_CMMCore object at 0x7fe32824a208>'''

core = Core()
print(core)

'''Ello beam block initiation'''

# clr.AddReference('C:\Program Files\Thorlabs\Elliptec\Thorlabs.Elliptec.ELLO_DLL.dll')

# from Thorlabs.Elliptec.ELLO_DLL import *

# #user check and enter com port for ello beam block (pass to function as string)
# ello_com_port = input("Enter COM port for Ello beam block connection (ex. COM5): ")  

# # Connect to device,check Windows Device Manager to find out which COM port is used.
# ELLDevicePort.Connect(ello_com_port)  #change to the correct port number

# # Define byte address. 
# min_address="0"
# max_address="F"

# # Build device list.
# ellDevices=ELLDevices()
# devices=ellDevices.ScanAddresses(min_address, max_address)
# filter_position = {"0":488, "31":561, "62":633}

# def ello_beam_control(laser):
#     # Initialize device. 
#     for device in devices:
#         if ellDevices.Configure(device):
            
#             addressedDevice=ellDevices.AddressedDevice(device[0])

#             print("Start changing filter")
#             # Call move methods. 
#             # addressedDevice.Home(ELLBaseDevice.DeviceDirection.Linear)
#             # time.sleep(1)

#             addressedDevice.MoveToPosition(laser)
#             pos = addressedDevice.Position
#             print("On filter:", filter_position[str(pos)]) 
            
'''Autofocus functions'''
def set_exposure(exp):

    # current_exposure = core.get_exposure()
    new_exposure = core.set_exposure(exp) #milliseconds
    print(new_exposure)

def move_stage_to_origin():

    """
    Moves the microscope stage to the origin (starting position when microscope start up).
    """

    # Get current XYZ positions
    x_origin = core.get_x_position()
    y_origin = core.get_y_position()
    z_origin = core.get_position()

    # print(f"Current Position -> X: {x}, Y: {y}, Z: {z}")

    # Move to origin
    core.set_xy_position(x_origin, y_origin)
    core.wait_for_device(core.get_xy_stage_device())
    # Add break time if needed
    #core.sleep(30000)  #sleep time in milliseconds (30sec)

    core.set_position(z_origin)
    core.wait_for_device(core.get_focus_device())
    # Add break time if needed
    #core.sleep(30000)  #sleep time in milliseconds (30sec)

    print("Stage moved to origin.")

def get_image() -> Any:
    """
    Capture an image and return it.

    """

    core.snap_image()     #Acquires a single image with current settings.
    tagged_image = core.get_tagged_image()

    # Pixel as default come out as 1D array - Convert to 2D NumPy array (in format that OpenCV can use)
    image = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])

    # Print the shape of the image
    #print(image.shape)

    return image

def get_metadata(filename, laser, blur, step, iris_position, iris_speed, iris_stretch_step, selected_index,desktop_path ):
    """
    Collects metadata from the microscope and appends it to a text file.
    
    Metadata includes:
    - camera device
    - X, Y, Z stage positions
    - Camera exposure time
    - Timestamp
    """
    # Get current metadata from the core
    cam = core.get_camera_device()
    x = core.get_x_position()
    y = core.get_y_position()
    z = core.get_position()
    exposure = core.get_exposure()
    timestamp = strftime("%Y-%m-%d %H:%M:%S", localtime())

    # Prepare metadata string
    metadata = (
        f"Camera: {cam},"
        f"Time: {timestamp}, "
        f"Iris position: {iris_position} x, "
        f"Iris speed: {iris_speed} mm/sec, "
        f"Iris stretch step: {iris_stretch_step} mm, "
        f"X: {x:.3f} um, Y: {y:.3f} um, Z: {z:.3f} um, "
        f"Exposure: {exposure} ms, "
        f"Laser wavelength: {laser} nm, "
        f"Blur: {blur}, "
        f"Step size: {step} um, "
        f"Selected region: {selected_index + 1} \n"
    )
    metadata_file_path = os.path.join(desktop_path, filename)

    # Append to the text file
    with open(metadata_file_path, "a") as f:
        f.write(metadata)

    # print(f"Metadata saved to: {file_path}")

def safety_cutoff_intensity(image):

    # Compute histogram for 16-bit image
    hist = cv2.calcHist([image], [0], None, [65535], [0, 65535])

    # Check intensity condition - if count of max intensity > 0 then lower Z stage and terminate program 
    if hist[60000] > 0:
        print("Too high intensity")
        #lower Z-position to original avoid camera damage from high intensity
        move_stage_to_origin()
        print("Z-stage was lowered to origin")

        # Plot and save histogram as image on Desktop
        plt.figure()
        plt.plot(hist)
        plt.title('Grayscale Image Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        histogram_save_path = os.path.join(os.path.expanduser("~"), "Desktop", "histogram_high_intensity.png")
        plt.savefig(histogram_save_path)
        plt.close()
        print(f"Histogram saved to {histogram_save_path}")

        #terminate the program - you will need to rerun the whole code again
        sys.exit("Program terminated due to high intensity.")

def figure_file_name(position: float) -> str:
    """
    Return a filename for a plot of data at a specific position.

    :param position: The position in um.
    """
    position_um = int(position // 1)
    position_frac = round((position - position_um) * 1000)
    return f"at_{position_um}_{position_frac}.png"

def tenengrad(image: Any) -> float:
    """
    Tenegrad algorithm - edge detection based on gradient 
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 7)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 7)
    tenengrad = np.sqrt(sobel_x**2 + sobel_y**2)
    focus_score = np.mean(tenengrad)
    return focus_score

   
def find_best_focus(start_um: float, end_um: float, step_size_um: float, blur: int, iris_position, previous_best_score, desktop_path) -> None:
    """
    Find best focus by changing the focal distance and taking images with the camera.

    :param start_um: The position, in um, to start at.
    :param end_um: The position, in um, to end at.
    :param step_size_um: The distance, in um, to move between taking images with the camera.
    :param blur: The blur to apply to images during processing.
    """
    previous_best_score = previous_best_score
    best_focus_score = 0.0
    best_focus_position = 0.0
    focus_scores: list[float] = []
    focus_positions: list[float] = []
        
    # How many steps to take to achieve the desired step size, +1 to check end_um
    steps = math.ceil((end_um - start_um) / step_size_um) + 1   #Total number of steps
    for step in range(0, steps):
        position = min(start_um + step * step_size_um, end_um)
        # Move Z axis to specific position
        core.set_position(position)
        core.wait_for_device(core.get_focus_device())

        # Add break time if needed
        #core.sleep(10000)  #sleep time in milliseconds (10sec)

        image = get_image()
        #check if image has too high intensity 
        safety_cutoff_intensity(image) #if reach intensity threshold, program will terminate 

        height, width = image.shape[:2]

        # Define the center region (e.g., the middle 50%)
        pcent = 50
        start_x = int(width * (0.5 - pcent/200))
        end_x = int(width * (0.5 + pcent/200))
        start_y = int(height * (0.5 - pcent/200))
        end_y = int(height * (0.5 + pcent/200))

        # Create the ROI
        roi = image[start_y:end_y, start_x:end_x]


        focus_score = tenengrad(roi)

        #append each position and its score to lists
        focus_positions.append(position)
        focus_scores.append(focus_score)

        # if focus_score > best_focus_score and focus_score <= previous_best_score - 20:  #since we know that approx. focus score from 1st round, so we can cut off some scan that is very far from the previous peak score.
        #     best_focus_position = position
        #     best_focus_score = focus_score
        #     step += 3  # skip to + 15um z-position 
        # else:
        if focus_score > best_focus_score:
            best_focus_position = position
            best_focus_score = focus_score

    # Move Z axis to best focus position
    core.set_position(best_focus_position)
    core.wait_for_device(core.get_focus_device())
    # Add break time if needed
    #core.sleep(10000)  #sleep time in milliseconds (10sec)

    #Capture image with best focus
    best_image = get_image()

    #preview in-focus image
    # fig = plt.figure()
    # plt.imshow(best_image,vmin=150,vmax=350,cmap = 'gray')
    # plt.show()

    # Save raw image to desktop (.tiff)
    file_path = os.path.join(desktop_path, f"best_focus_image_20x_{start_um}_{end_um}_{step_size_um}_Pos{iris_position}.tiff")
    cv2.imwrite(file_path, best_image)
    print(f"Image saved to {file_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return best_focus_score, best_focus_position, focus_positions, focus_scores

def variance_plot(desktop_path, all_positions_and_scores, input_name=""):
    """
    Plot focus score vs Z position across all iris positions.
    """
    plt.figure(figsize=(10, 6))
    for positions, scores in all_positions_and_scores:
        plt.plot(positions, scores, 'o-', alpha=0.7)

    plt.xlabel('Z Position (μm)')
    plt.ylabel('Focus Score')
    plt.title('Focus Scores vs Z Position')
    plt.grid(True)
    
    #Save plot to desktop path
    save_name = input_name.replace(" ", "_") if input_name else "plot"
    filename = f"variance_scores_{save_name}.png"
    variance_plot_filename = os.path.join(desktop_path, filename)
    plt.savefig(variance_plot_filename, dpi=300)
    print("Variance score plot was saved")
    #print(f"Saved variance score plot to: {variance_plot_filename}")

    #plt.show()
    plt.close()

def save_positions_scores_raw_data_to_excel(desktop_path, all_positions_and_scores, input_name="" ):
    """
    Save raw focus score and Z position data to an Excel file.
    """
    data = []
    for set_index, (positions, scores) in enumerate(all_positions_and_scores):
        for z, score in zip(positions, scores):
            data.append({
                "Set": set_index + 1,   #set#0 = iris position 0.0, set#1 = iris position 0.5
                "Z Position (μm)": z,
                "Focus Score": score
            })

    df = pd.DataFrame(data)

    # Generate filename
    save_name = input_name.replace(" ", "_") if input_name else "excel"
    filename = f"scores_positions_raw_data_{save_name}.xlsx"
    file_path = os.path.join(desktop_path, filename)

    # Save to Excel
    df.to_excel(file_path, index=False)
    print(f"Scores and positions raw data saved to Excel at: {file_path}")

'''Feature tracking functions'''

def segment_and_annotate(image):
    'Binarization'
    img_avg_intensity = np.average(image)
    img_sd_intensity = 1* np.std(image)
    adaptive_binary_intensity = img_avg_intensity + img_sd_intensity
    bin_image = (image > adaptive_binary_intensity).astype(np.uint8)
    # bin_image = (image > 120).astype(np.uint8)
    
    # use imageJ IsoData binarization method
    # bin_image = binarize_imagej_default(image)
    
    'Show Binary image'
    plt.imshow(bin_image, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    plt.show(block = False)
    plt.pause(2)
    plt.close()

    labeled_img = label(bin_image)
    regions = regionprops(labeled_img)
    # print(len(regions)) #This has all the labeled object 
    'Filter object by size'
    regions = [region for region in regions if 1000 <= region.area ] 
    # print(regions)


    rgb_image = cv2.cvtColor((bin_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for i, p in enumerate(regions):
        y, x = p['centroid']
        area = p['area']
        shape = p['moments_hu']
        # print('centroid:', y,x)
        # print('areas:', area)
        # print('shape:', shape)
        cv2.circle(rgb_image, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.putText(rgb_image, str(i+1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title("Regions with Centroids")
    plt.axis('off')
    plt.show(block = False)
    plt.pause(2)
    plt.close()

    return rgb_image, regions  # Return the annotated RGB image

def move_stage_to_center(centroid, CENTER, PIXEL_SIZE_UM ):
    delta_px = np.array(centroid) - CENTER   #calculate x, y offset from the center in pixels
    print(delta_px)

    delta_um = delta_px * PIXEL_SIZE_UM #Convert pixel displacement to micrometers
    current_x = core.get_x_position() #current microscope stage position in micrometers
    current_y = core.get_y_position()
    print(current_x, current_y)
    core.set_xy_position(current_x + delta_um[1], current_y - delta_um[0]) #Make sure this is correct?
    
    core.wait_for_device(core.get_xy_stage_device())
    new_x = core.get_x_position()
    new_y = core.get_y_position()

    print(new_x,new_y)

def track_centroid_shift( new_centroid, CENTER, PIXEL_SIZE_UM):
    delta_px = np.array(new_centroid) - CENTER
    delta_um = delta_px * PIXEL_SIZE_UM
    return delta_um

def display_save_image(image, title, filename, desktop_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('on')
    plt.title(title)
    plt.show(block = False)
    plt.pause(2)
    plt.close()
    save_path = os.path.join(desktop_path, filename)
    cv2.imwrite(save_path, image)
    print(f"Image saved to: {save_path}")


# def match_region_nearbyfield(target, candidates, binary_intensity, iris_position):
#     """
#     Finds the best matching region. 
#     1. Checks if candidates meet 90% similarity (Area & Shape) to target.
#     2. If exactly 1 match found -> Returns it immediately.
#     3. If >1 match found -> Sorts to find top 3, then runs detailed scoring.
#     4. If 0 matches found -> Triggers 4-direction search (Up/Down/Left/Right).
    
#     Requires:
#     - core: Microscope controller object.
#     - current_x, current_y: Current stage coordinates.
#     - Global functions: get_image(), segment_and_annotate()
#     """
#     current_x = core.get_x_position()
#     current_y = core.get_y_position()
#     # --- HELPER 1: Consistency Check ---
#     def is_candidate_valid(t, c, threshold=0.2):
#         """
#         Checks if candidate is within 90% similarity (10% tolerance) 
#         of target's area and shape.
#         """
#         d_area_pct = abs(c['area'] - t['area']) / t['area']
#         d_shape = np.linalg.norm(c['moments_hu'] - t['moments_hu'])
#         return d_area_pct <= threshold and d_shape <= threshold

#     # --- HELPER 2: Scoring Logic ---
#     def calculate_best_match(t, candidate_list):
#         best_s = float('inf')
#         best_m = None
        
#         for c in candidate_list:
#             d_centroid = np.linalg.norm(np.array(c['centroid']) - np.array(t['centroid']))
#             d_area = abs(c['area'] - t['area']) / t['area']
#             d_shape = np.linalg.norm(c['moments_hu'] - t['moments_hu'])

#             # Weighted scoring
#             score = (0.2 * d_centroid) + (0.4 * d_area * 100) + (0.4 * d_shape * 10)
            
#             if score < best_s:
#                 best_s = score
#                 best_m = c
#         return best_m

#     # ---------------------------------------------------------
#     # STEP 1: Filter Initial Candidates
#     # ---------------------------------------------------------

#     valid_candidates = [c for c in candidates if is_candidate_valid(target, c)]

#     # OPTIMIZATION: Handle 1 candidate vs multiple candidates
#     if len(valid_candidates) == 1:
#         print("Found exactly one valid match. Returning immediately.")
#         return valid_candidates[0]

#     elif len(valid_candidates) > 1:
#         # Sort by raw similarity to find the "Top 3" contenders
#         valid_candidates.sort(key=lambda x: (abs(x['area']-target['area'])/target['area']) + np.linalg.norm(x['moments_hu']-target['moments_hu']))
        
#         top_3_candidates = valid_candidates[:3]
#         print(f"Found {len(valid_candidates)} valid matches. Scoring Top {len(top_3_candidates)}.")
#         return calculate_best_match(target, top_3_candidates)

#     # ---------------------------------------------------------
#     # STEP 2: Search Mode (If 0 valid candidates found)
#     # ---------------------------------------------------------
#     print("No matches within 90% similarity. Initiating 4-direction search...")

#     search_offsets = [(200, 0), (-200, 0), (0, 200), (0, -200)] #um

#     for dx, dy in search_offsets:
#         new_x = current_x + dx
#         new_y = current_y + dy 

#         print(f"Searching field at offset: ({dx}, {dy})...")
        
#         # 1. Move Stage
#         core.set_xy_position(new_x, new_y)

#         '''Apply Autofocus'''
#         current_z = core.get_position()  
#         start = current_z -100 
#         end = current_z + 100
#         iris_position = iris_position
#         big_step= 5 #um
#         previous_best_score = 0
#         blur = 5
#         #low resolution autofocus (big step size)
#         best_focus_score_lowRes, best_focus_position_lowRes, focus_positions_lowRes, focus_scores_lowRes = find_best_focus(start, end, big_step, blur, iris_position, previous_best_score, desktop_path)
#         #keep update the best focus score for find_best_score to compare
#         best_focus_score_lowRes = previous_best_score
#         # 2. Capture and Segment
#         new_img = get_image() 
#         new_candidates = segment_and_annotate(new_img, binary_intensity) 

#         # 3. Check for valid candidates in new FOV
#         valid_new_candidates = [c for c in new_candidates if is_candidate_valid(target, c)]

#         if len(valid_new_candidates) == 1:
#             print("Found exactly one match in new field.")
#             return valid_new_candidates[0]
            
#         elif len(valid_new_candidates) > 1:
#             print(f"Found {len(valid_new_candidates)} matches in new field.")
#             valid_new_candidates.sort(key=lambda x: (abs(x['area']-target['area'])/target['area']) + np.linalg.norm(x['moments_hu']-target['moments_hu']))
#             return calculate_best_match(target, valid_new_candidates[:3])

#     print("Search complete. No matching object found in nearby fields.")
#     return None

def match_region(target, candidates):
    '''find the best matching region from a list of 
    candidates that resembles a given target region,
    using a combination of centroid distance, area difference, 
    and shape similarity (Hu moments).
    https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.regionprops'''

    best_score = float('inf') ## Start with an infinitely bad score
    best_match = None

    for c in candidates:
        d_centroid = np.linalg.norm(np.array(c['centroid']) - np.array(target['centroid']))  #Euclidean distance between the centroids of the target and a candidate. Measures spatial closeness.
        d_area = abs(c['area'] - target['area']) / target['area']  #Relative difference in region size. Normalized by target area 
        d_shape = np.linalg.norm(c['moments_hu'] - target['moments_hu'])  #Euclidean distance between the log-scaled Hu moments. Measures shape similarity.

        '''To do: check score max/min + area + shape
         Try adding edge detection then add as one of factor to weight - reduce centroid diff  '''
        # Weighted scoring - less score means closer match 
        score = (0.3 * d_centroid) + (0.2 * d_area * 100) + (0.5 * d_shape * 10)
        if score < best_score:
            best_score = score
            best_match = c

    return best_match

def match_region_normalize_scaling(target, candidates):
    if not candidates:
        return None, 0
    # 1. Pre-process Target Features (Log Transform Hu Moments)
    # target_hu_log = get_log_hu(target['moments_hu'])

    # 1. Pre-calculate raw differences for ALL candidates
    diffs = []
    for i, c in enumerate(candidates, start = 1):
        d_centroid = np.linalg.norm(np.array(c['centroid']) - np.array(target['centroid'])) #Centroid coordinate tuple (row, col)
        d_area_filled = abs(c['area_filled'] - target['area_filled']) #Area of the region with all the holes filled in.
        d_shape = np.linalg.norm(c['moments_hu'] - target['moments_hu'])    # Hu moments (translation, scale and rotation invariant). 
        d_euler = abs(c['euler_number'] - target['euler_number']) #Euler characteristic of the set of non-zero pixels. Computed as number of connected components subtracted by number of holes
        d_perimeter = abs(c['perimeter'] - target['perimeter']) #Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
        # print(i)
        # print('d_centroid:', d_centroid)
        # print('d_area:', d_area_filled)
        # print('d_shape:', d_shape)
        # print('d_euler:', d_euler)
        # print('d_perimeter:', d_perimeter)
        diffs.append([d_centroid, d_area_filled, d_shape, d_euler, d_perimeter])
        
    diffs = np.array(diffs) # Shape: (num_candidates, 4)

    # 2. Min-Max Scaling: Bring every factor into a 0.0 to 1.0 range
    # Formula: (val - min) / (max - min)
    mins = diffs.min(axis=0)
    maxs = diffs.max(axis=0)
    
    # Avoid division by zero if all candidates have the same value for a factor
    ranges = maxs - mins
    ranges[ranges == 0] = 1 
    
    normalized_diffs = (diffs - mins) / ranges
    print('normalized_diffs:', normalized_diffs)

    # 3. Apply Weights
    # Now that everything is 0-1, weights truly represent importance
    weights = np.array([
        0.20, # Centroid
        0.20, # d_area_filled 
        0.20, # Shape (Hu Moments)
        0.20, # Euler Number (Holes)
        0.20  #perimeter
    ])

    final_scores = np.dot(normalized_diffs, weights)
    # print('final_scores:', final_scores)
    # 4. Find the best match
    best_idx_in_list = np.argmin(final_scores)
    best_match = candidates[best_idx_in_list]
    # Return best match and the human-readable indePost-stretch not center image_pos1x (starting at 1)
    return best_match

def iris_control(iris_com_port, input, iris_stretch_step, initial_val):
    input = input
    # Connect to a device (adjust COM port as needed)

    dev = serial.Serial(iris_com_port, baudrate=115200, timeout=1)  # COM5 is for IX83 PC, COM3 for ramita's pc, check in device manager to see what the port is (can change if the controller or other electronic components are changed)
    print("Serial connection established.")
    dev.write(b'Xspeed 0.1 \n')  # Include newline if your device expects it
    # Send a command
    # Compute the position
    position = (input * iris_stretch_step) + initial_val    #Add offset for iris starting point here 

    # Format and encode the command string
    command = f"Xgoto {position}\n".encode()

    # Send the command
    dev.write(command)
    
    dev.close()

def multiple_FOV_acq(x_position, y_position, iris_position,overlap,objective_mag,gridsize_x,gridsize_y,blur, desktop_path):
    '''input: x, y of the centroid of the selected features
    Do 9x9 grid around the features'''
    total_time = time() 
    print("starting grid imaging")
    fov_offsets = ((100/objective_mag)*0.102)*0.01*(100-overlap)*1412  #pixelsize for chosen objective * overlap percent * chip size
    print("overalp in um:",fov_offsets)
    
    tempx = np.linspace( (x_position - (fov_offsets*gridsize_x)/2), (x_position + (fov_offsets*gridsize_x)/2 ),gridsize_x )
    tempy = np.linspace( (y_position - (fov_offsets*gridsize_y)/2), (y_position + (fov_offsets*gridsize_y)/2 ),gridsize_y )
    X,Y = np.meshgrid(tempx,tempy)
    grid = np.stack((X.ravel(), Y.ravel()), axis=1)
    print(grid)
    for i, (dx, dy) in enumerate(grid):
        move_time = time()
        core.set_xy_position(dx,dy)
        core.wait_for_device(core.get_xy_stage_device())
        print("move time:",time() - move_time)
        '''Apply Autofocus'''
        current_z = core.get_position()  
        start = current_z - 10
        end = current_z + 10
        iris_position = iris_position
        big_step= 5 #um
        previous_best_score = 0

        #low resolution autofocus (big step size)
        autofocus_time = time() 
        best_focus_score_lowRes, best_focus_position_lowRes, focus_positions_lowRes, focus_scores_lowRes = find_best_focus(start, end, big_step, blur, iris_position, previous_best_score, desktop_path)
        #keep update the best focus score for find_best_score to compare
        best_focus_score_lowRes = previous_best_score
        print("autofocus time:",time() - autofocus_time)
        img = get_image()
        # display_save_image(img, f"Tracked_FOV_{i}_gridsz_{gridsize_x}X{gridsize_y}", f"Tracked_features_FOV{i}_{iris_position}x.tiff")
        
        
        
        folder_path1 = f"MFOV_grid_{gridsize_x}X{gridsize_y}_irispos_{iris_position}"
        # folder_path2 = f"stack{i}"
        
        
        os.makedirs(os.path.join(desktop_path,folder_path1), exist_ok=True)

        # os.makedirs(folder_path2, exist_ok=True)

        save_path = os.path.join(desktop_path,folder_path1, f"Tracked_features_FOV{i}_{iris_position}expansion.tiff")
            

        cv2.imwrite(save_path, img)
        print(f"Image saved to: {save_path}")
    print("total time:",time() - total_time)
    
    return "Done capture multiple FOV"

def main():  
    
    #set exposure 
    selected_exp = int(input("Enter the desired exposure (whole number only): ")) #milliseconds
    set_exposure(selected_exp)

    #enter initial stretching expansion condition
    initial_val = float(input("Enter the iris intital stretching value (number only), default speed = 0.1, step = 0.05: "))

    #user enter laser line used
    laser = int(input("Enter the laser line (number only): ")) #um
     
    #user enter objective used
    objective_mag = int(input("Enter the objective (number only): "))

    #user enter pixel size 
    PIXEL_SIZE_UM = float(input("Enter pixel size (0.51 for 20x, 0.204 for 50x): "))
    
    #user enter intensity value for binary mask
    # binary_intensity = int(input("Enter intensity for binary mask: "))

    #user check and enter com port for iris (pass to function as string)
    iris_com_port = input("Enter COM port for Iris connection (ex. COM11): ")

    '''Set laser at 633nm for nanoscribe tracking'''
    # ello_beam_control(2)

    '''Constants'''
    IMAGE_SIZE = 1412
    CENTER = np.array([IMAGE_SIZE // 2, IMAGE_SIZE // 2])
    # z start position for first autofocus run
    start= core.get_position() #um 
    end = core.get_position()  #um
    big_step= 3 #um
    previous_best_score = 0
    blur = 5
    core.set_xy_position(core.get_x_position(), core.get_y_position())  
    core.wait_for_device(core.get_xy_stage_device())

    # 0.5 steps, max at 3.8, total number of loop (N = 13) Number of loop depend on step increment!
    iris_stretch_step = 0.04 # can be changed
    max_value = 3.9 # can be changed
    N = int((max_value -initial_val) / iris_stretch_step)  
    iris_position = initial_val

    #create empty lists to store data
    all_positions_and_scores_lowRes = []
    best_focus_scores_lowRes = []
    best_focus_positions_lowRes = []

    timestamps = []
    durations = []

    #record starting time
    t0 = time()
    print(f"Starting tracking and autofocus: {t0}")
    
    #loop through each iris movement step - assume first image at position 0 is already in-focus
    for n in range(N): 
        
        start_time = time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        '''Apply tracking algorithm'''
        'Center and select the interested features to track (only do this once at position 0)'
        if n == 0:  #image should be in-focus already!
            # Step 1: capture image
            print("Capture image")
            first_image = get_image()
            # save first image
            file_path = os.path.join(desktop_path, f"best_focus_image_20x_{start}_{end}_{big_step}_Pos{iris_position}.tiff")
            cv2.imwrite(file_path, first_image)
            print(f"Image saved to {file_path}")

            # Step 2-3: Segment, label, and get centroids
            annotated_image, props = segment_and_annotate(first_image)

            display_save_image(annotated_image, "Segment and Annotated image", f"annotated_image_pos{iris_position}.tiff", desktop_path)

            # Step 4: Select region and obtain its centroid
            selected_index = int(input(f"Enter region number (1 to {len(props)}): ")) - 1
            selected_region = props[selected_index]
            selected_centroid = selected_region['centroid']
            print(selected_centroid)

            # Step 5: Move stage 
            print("Moving stage to center selected region...")
            move_stage_to_center(selected_centroid, CENTER, PIXEL_SIZE_UM)
            center_image = get_image()
            display_save_image(center_image, "Move selected region to center", f"move_to_center_pos{iris_position}.tiff", desktop_path)

            #Step 6: reselect the center features - edit this to don't need user selection again
            annotated_image, props = segment_and_annotate(center_image) 
            display_save_image(annotated_image, "Annotated image after centering", f"Annotated_image_after_centering{iris_position}.tiff", desktop_path)

            # Select region and obtain its centroid
            selected_index = int(input(f"Enter region number (1 to {len(props)}): ")) - 1
            selected_region = props[selected_index]
            selected_centroid = selected_region['centroid']
            
            # # switch laser to 488 and 561 to capture cell images
            # ello_beam_control(0)
            # image = get_image()
            # display_save_image(image, "Tracked image on 488nm", f"Tracked_Image_488nm_pos{iris_position}.tiff", desktop_path)

            # ello_beam_control(1)
            # image = get_image()
            # display_save_image(image, "Tracked image on 561nm", f"Tracked_Image_561nm_pos{iris_position}.tiff", desktop_path)
            # #Bring laser back to 633nm for nanoscribe tracking 
            # ello_beam_control(2)

            'Other cases when n != 0 and we want to compare between previous and next frame to track object'
        else:
            # Step 7: Re-capture and track movement compare with first centered image 
            new_image = get_image()
            display_save_image(new_image, "Post-stretch not center image", f"Post-stretch not center image_pos{iris_position}.tiff", desktop_path)
            _, new_props = segment_and_annotate(new_image)
            
            # matched_region = match_region(selected_region, new_props, binary_intensity, iris_position)
            matched_region = match_region_normalize_scaling(selected_region, new_props)

            new_centroid = matched_region['centroid']

            delta_um = track_centroid_shift(new_centroid, CENTER, PIXEL_SIZE_UM)
            # print(f"Centroid moved by {delta_px} pixels or {delta_um} µm")
            print(delta_um)
            # step 8: move stage according to delta_um
            print("stage move to track object")
            current_x = core.get_x_position() #current microscope stage position in micrometers
            current_y = core.get_y_position()
            core.set_xy_position(current_x + delta_um[1], current_y - delta_um[0]) #Make sure this is correct?
            core.wait_for_device(core.get_xy_stage_device())
            image_track = get_image()
            display_save_image(image_track, "track image", f"track_image_pos{iris_position}.tiff", desktop_path)
            # Step 9: Update the reference frame and region for next iteration to compare consecutive frames
            selected_region = matched_region
            selected_centroid = new_centroid
            #print z position
            print("z of tracked image:", core.get_position())

        '''Capture multiple FOV around the tracked features'''
        if iris_position in [1.5, 2.02, 2.50, 3.02, 3.50, 3.82]:
            mfov_current_x = core.get_x_position()
            mfov_current_y = core.get_y_position()
            mfov_current_z = core.get_position()
            print("x:",mfov_current_x,"y:",mfov_current_y)
            mfov_iris_pos = iris_position
            overlap = 80  # in percent

            done_message = multiple_FOV_acq(mfov_current_x, mfov_current_y, mfov_iris_pos,overlap,objective_mag,5,5,blur, desktop_path)  #change gridsize as required
            print(done_message)

            core.set_xy_position(mfov_current_x, mfov_current_y)
            core.set_position(mfov_current_z)
            #redo coarse autofocus to ensure image is in-focus
            start_focus_MFOV = mfov_current_z -20
            end_focus_MFOV = mfov_current_z + 20
            _1, _2, _3, _4 = find_best_focus(start_focus_MFOV, end_focus_MFOV, big_step, blur, iris_position, previous_best_score, desktop_path)

        '''Run iris control function to stretch gel'''
        # iris_speed = response.stdout.strip()
        response = iris_control(iris_com_port, n+1, iris_stretch_step, initial_val)
        iris_speed = response

        #add pause time for gel to fully stretch 
        sleep(10)  # Sleep for 5 seconds (adjust as needed)
        
        '''Apply Autofocus algorithm'''
        
        if n == 0:   # z-scan range for 1.1x - 1.5x
            start = start
            end = end + 300 #300
        if n >= 1 and n <= 10:   # z-scan range for 1.1x - 1.5x
            start = core.get_position()
            end = core.get_position() + 250  #300
        elif n >=11 and n <= 20:   # z-scan range for 1.55x - 2x
            start = core.get_position() - 5  #5
            end = core.get_position() + 150   #200
        elif n >=21 and n <= 30:  # z-scan range for 2.05x - 2.5x
            start = core.get_position() - 10
            end = core.get_position() + 50
        elif n >=31 and n <= 70:    # z-scan range for 2.55x - 3.8x
            start = core.get_position() - 20
            end = core.get_position() + 40
        #low resolution autofocus (big step size)
        best_focus_score_lowRes, best_focus_position_lowRes, focus_positions_lowRes, focus_scores_lowRes = find_best_focus(start, end, big_step, blur, iris_position, previous_best_score, desktop_path)
        
        #keep update the best focus score for find_best_score to compare
        best_focus_score_lowRes = previous_best_score

        # append autofocus data
        # print(f"The best focus ({best_focus_score}) was found at {best_focus_position}.")
        best_focus_scores_lowRes.append(best_focus_score_lowRes)
        best_focus_positions_lowRes.append(best_focus_position_lowRes)
        all_positions_and_scores_lowRes.append((focus_positions_lowRes, focus_scores_lowRes))
        
        end_time = time() 
        duration = end_time - start_time  #end time per 1 round of focus and track 
        
        metadata_filename = f"image_metadata_pos{iris_position}.txt"
        get_metadata(metadata_filename, laser, blur, big_step, iris_position, iris_speed, iris_stretch_step, selected_index, desktop_path)   #metadata for low resolution step size image
        timestamps.append(timestamp) #autofocus start time
        durations.append(duration)
        
        '''Switch laser to capture cells fluorescense at 488 and 561nm'''
        # if iris_position in [1.5, 2.0, 2.5, 3.0, 3.50, 3.80]:
        #     ello_beam_control(0)
        #     image = get_image()
        #     display_save_image(image, "Tracked image on 488nm", f"Tracked_Image_488nm_pos{iris_position}.tiff", desktop_path)
            
        #     ello_beam_control(1)
        #     image = get_image()
        #     display_save_image(image, "Tracked image on 561nm", f"Tracked_Image_561nm_pos{iris_position}.tiff", desktop_path)
            
        #     #Bring laser back to 633nm for nanoscribe tracking 
        #     ello_beam_control(2)

        
        # start = core.get_position() - 20
        # end = core.get_position() + 250
        iris_position += iris_stretch_step
        iris_position = round(iris_position, 2)
        
    print(f"Done full autofocus: {time() - t0}")  # unit = seconds
    
    #plot variance score vs z-position of both small and big step size 
    variance_plot(desktop_path, all_positions_and_scores_lowRes, input_name="lowRes")
    # variance_plot(all_positions_and_scores_highRes, input_name="highRes")
    save_positions_scores_raw_data_to_excel(desktop_path, all_positions_and_scores_lowRes, input_name="lowRes")
    # save_positions_scores_raw_data_to_excel(all_positions_and_scores_highRes, input_name="highRes")

    #save best_focus_scores and best_focus_positions as excel 
    for label, scores, positions in [
    ("lowRes", best_focus_scores_lowRes, best_focus_positions_lowRes),
    ]:
        
        df = pd.DataFrame({
            "Iris Position": [i * iris_stretch_step for i in range(N)],
            "Timestamp": timestamps,
            "Autofocus Duration (s)": durations,
            "Best Focus Score": scores,
            "Best Focus Z Position (μm)": positions, 
        })

        filename = os.path.join(desktop_path, f"autofocus_raw_data_{label}.xlsx")
        df.to_excel(filename, index=False)
        print(f"Focus data saved to: {filename}")


if __name__ == "__main__":
    main()
