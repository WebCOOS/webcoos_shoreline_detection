# getTimexShoreline.py
import cv2 
from datetime import datetime 
from itertools import chain
import json 
# import math 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from PIL import Image, ImageDraw 
import re 
import scipy.signal as signal 
from skimage.filters import threshold_otsu 
from skimage.measure import profile_line 
from statsmodels.nonparametric.kde import KDEUnivariate

# Add to your imports at the top of the file
from matplotlib.path import Path


def getStationInfo(ssPath):
    # Loads json and converts data to NumPy arrays.
    with open(ssPath, 'r') as setupFile:
        stationInfo = json.load(setupFile)

        assert stationInfo
        assert 'Dune Line Info' in stationInfo
        assert 'Shoreline Transects' in stationInfo

    #if missing dune line interpolation, check dune line points
    if 'Dune Line Interpolation' in stationInfo['Dune Line Info']:
        stationInfo['Dune Line Info']['Dune Line Interpolation'] = np.asarray(stationInfo['Dune Line Info']['Dune Line Interpolation'])
    else:
        stationInfo['Dune Line Info']['Dune Line Points'] = np.asarray(stationInfo['Dune Line Info']['Dune Line Points'])
        
    stationInfo['Shoreline Transects']['x'] = np.asarray(stationInfo['Shoreline Transects']['x'])
    stationInfo['Shoreline Transects']['y'] = np.asarray(stationInfo['Shoreline Transects']['y'])
    return stationInfo

def mapROI(stationInfo, photo):
    """
    Creates a mask from pre-defined ROI points and extracts ROI from the image.
    Uses the roi_points directly for more reliable polygon construction.
    """
    # Input validation
    if not isinstance(photo, np.ndarray) or photo.ndim not in [2, 3]:
            raise ValueError("photo must be a 2D or 3D numpy array")

    h, w = photo.shape[:2]
    is_color = photo.ndim == 3

    # Get ROI points
    roi_points = np.array(stationInfo['roi_points'], dtype=float)
    
    # Ensure ROI points are within image bounds
    roi_points[:, 0] = np.clip(roi_points[:, 0], 0, w-1)
    roi_points[:, 1] = np.clip(roi_points[:, 1], 0, h-1)

        # Close the polygon if not already closed
    if not np.array_equal(roi_points[0], roi_points[-1]):
        roi_points = np.vstack((roi_points, roi_points[0]))

    # Create mask
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    points = np.column_stack((x.ravel(), y.ravel()))
    path = Path(roi_points)
    mask = path.contains_points(points).reshape(h, w)

    # Apply slight dilation to include edge pixels
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(mask, structure=np.ones((3, 3)))

    # Apply mask
    maskedImg = photo.astype(np.float64)
    if is_color:
        mask_3d = np.repeat(mask[:, :, np.newaxis], photo.shape[2], axis=2)
        maskedImg[~mask_3d] = np.nan
    else:
        maskedImg[~mask] = np.nan

    if maskedImg.max() > 1:
        maskedImg /= 255.0

    return maskedImg

def improfile(rmb, stationInfo):
    # Extract intensity profiles along shoreline transects.
    transects = stationInfo['Shoreline Transects']
    xt = np.asarray(transects['x'], dtype=int)
    yt = np.asarray(transects['y'], dtype=int)
    n = len(xt)
    imProf = [profile_line(rmb, (yt[i,1], xt[i,1]), (yt[i,0], xt[i,0]), mode='constant') for i in range(n)]
    improfile = np.concatenate(imProf)[~np.isnan(np.concatenate(imProf))]
    return improfile

def ksdensity(P, **kwargs):
    # Univariate kernel density estimation.
    x_grid = np.linspace(P.max(), P.min(), 1000) # Could cache this.
    kde = KDEUnivariate(P)
    kde.fit(**kwargs)
    pdf = kde.evaluate(x_grid)
    return (pdf, x_grid)


def extract(stationInfo, rmb, maskedImg, threshInfo):
    # Uses otsu's threshold to find shoreline points based on water orientation.
    stationname = stationInfo['Station Name']
    slTransects = stationInfo['Shoreline Transects']
    dtInfo = stationInfo['Datetime Info']
    date = dtInfo.date()
    xt = np.asarray(slTransects['x'])
    yt = np.asarray(slTransects['y'])
    orn = stationInfo['Orientation']
    thresh = threshInfo['Thresh']
    thresh_otsu = threshInfo['Otsu Threshold']
    thresh_weightings = threshInfo['Threshold Weightings']
    length = min(len(xt), len(yt))
    trsct = range(0, length)
    values = [0]*length
    revValues = [0]*length
    yList = [0]*length
    xList = [0]*length

    def find_first_exceeding_index(values, threshold):
        values = np.array(values)
        for i in range(1, len(values)):
            if (values[i-1] < threshold and values[i] >= threshold) or (values[i-1] >= threshold and values[i] < threshold):
                return i
        return None

    if orn == 0:
        for i in trsct:
            x = int(xt[i][0])
            if 'roi_points' not in stationInfo: 
                yMax = int(yt[i][0]) # JWL flipped these for new cocoabeach station config
                yMin = int(yt[i][1]) # JWL flipped these for new cocoabeach station config
            else:
                yMax = int(yt[i][1]) 
                yMin = int(yt[i][0])
            y = yMax - yMin
            # y = abs(y)
            yList[i] = np.zeros(shape=y)
            val = [0]*(yMax - yMin)
            for j in range(len(val)):
                k = yMin + j
                val[j] = rmb[k][x]
            val = np.array(val)
            values[i] = val

        idx = [0]*len(xt)
        xPt = [0]*len(xt)
        yPt = [0]*len(xt)
        for i in range(len(values)):
            idx[i] = find_first_exceeding_index(values[i], thresh_otsu)
            if idx[i] is None:
                yPt[i] = None
                xPt[i] = None
            else:
                yPt[i] = min(yt[i]) + idx[i]
                xPt[i] = int(xt[i][0])
        shoreline = np.vstack((xPt, yPt)).T
    # if orn == 3, then we need to find the first exceeding index in the opposite direction of orn == 0
    elif orn == 3:    
        for i in trsct:
            x = int(xt[i][0])
            if 'roi_points' not in stationInfo:
                yMax = int(yt[i][0]) # JWL flipped these for new cocoabeach station config
                yMin = int(yt[i][1]) # JWL flipped these for new cocoabeach station config
            else:
                yMax = int(yt[i][1]) # flipped for the Ferry Beach station config
                yMin = int(yt[i][0]) # flipped for the Ferry Beach station config
            y = yMax - yMin
            y = abs(y)
            print(f"shape of y: {y}")
            yList[i] = np.zeros(shape=y)
            val = [0]*y
            for j in range(len(val)):
                k = yMin + j
                val[j] = rmb[k][x]
            val = np.array(val)
            values[i] = val
        # reverse the values for orn == 3
        revValues = [val[::-1] for val in values]
        idx = [0]*len(xt)
        xPt = [0]*len(xt)
        yPt = [0]*len(xt)
        for i in range(len(revValues)):
            idx[i] = find_first_exceeding_index(revValues[i], thresh_otsu)
            if idx[i] is None:
                yPt[i] = None
                xPt[i] = None
            else:
                yPt[i] = max(yt[i]) - idx[i]
                xPt[i] = int(xt[i][0])
        shoreline = np.vstack((xPt, yPt)).T
    # for orn == 1 or 2
    else:
        for i in trsct:
            xMax = int(xt[i][1])  # JWL chnged this from 0 Jeanettes ok, still ok for Oak Island
            y = int(yt[i][0])
            yList[i] = np.full(shape=xMax, fill_value=y)
            xList[i] = np.arange(xMax)
            values[i] = rmb[y][0:xMax]
            revValues[i] = rmb[y][::-1]

        idx = [0]*len(yt)
        xPt = [0]*len(yt)
        yPt = [0]*len(yt)
        for i in range(len(revValues)):
            idx[i] = find_first_exceeding_index(values[i], thresh_otsu)
            xPt[i] = idx[i]
            yPt[i] = int(yt[i][0])
        shoreline = np.vstack((xPt, yPt)).T

    # Convert numpy data types to native Python types and handle None values in shoreline
    slVars = {
        'Station Name': stationname,
        'Date': str(date),
        'Time Info': str(dtInfo),
        'Thresh': float(thresh),
        'Otsu Threshold': float(thresh_otsu),
        'Shoreline Transects': {
            'x': xt.tolist(),
            'y': yt.tolist()
        },
        'Threshold Weightings': [float(w) for w in thresh_weightings],
        'Shoreline Points': [[float(item) if item is not None else None for item in point] for point in shoreline]
    }

    try:
        del slVars['Time Info']['DateTime Object (UTC)']
        del slVars['Time Info']['DateTime Object (LT)']
    except:
        pass

    if isinstance(slVars['Shoreline Transects']['x'], np.ndarray):
        slVars['Shoreline Transects']['x'] = slVars['Shoreline Transects']['x'].tolist()
        slVars['Shoreline Transects']['y'] = slVars['Shoreline Transects']['y'].tolist()

    # Create directories if they do not exist
    base_dir = os.path.join(os.getcwd(), 'transect_jsons', stationname)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        print(f"Directory exists: {base_dir}")

    # Save JSON file to the directory
    fname = os.path.join(base_dir, f'{stationname}-{datetime.strftime(dtInfo, "%Y-%m-%d_%H%M")}.avg.slVars.json')
    with open(fname, "w") as f:
        json.dump(slVars, f)
    print(f"Saved JSON to: {fname}")
    
    return shoreline


def pltFig_tranSL(stationInfo, photo, tranSL):
    stationname = stationInfo['Station Name']
    dtInfo = stationInfo['Datetime Info']
    date = str(dtInfo.date())
    time = str(dtInfo.hour).zfill(2) + str(dtInfo.minute).zfill(2)  # Ensure two digits for hour and minute
    Di = stationInfo['Dune Line Info']
    # duneInt = Di['Dune Line Interpolation']
    duneInt = Di['Dune Line Points']
    xi, py = duneInt[:,0], duneInt[:,1]
    tranSL = np.array(tranSL, dtype=np.float64)
    
    # Filter rows with NaN values
    valid_mask = ~np.isnan(tranSL).any(axis=1)
    tranSL = tranSL[valid_mask]
    
    # Sort based on orientation
    if len(tranSL) > 0:  # Only sort if we have valid points
        if stationInfo['Orientation'] in [0, 3]:
            tranSL = tranSL[np.argsort(tranSL[:, 0])]
        elif stationInfo['Orientation'] in [1, 2]:
            tranSL = tranSL[np.argsort(tranSL[:, 1])]
    else:
     #   print("Warning: No valid shoreline points after filtering")
    
   # print(f"Sorted tranSL coordinates: {tranSL}")
         plt.ioff()
    fig_tranSL = plt.figure()
    plt.imshow(photo, interpolation='nearest')
    plt.xlabel("Image Width (pixels)", fontsize=10)
    plt.ylabel("Image Height (pixels)", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.plot(tranSL[:, 0], tranSL[:, 1], color='r', linewidth=2, label='Detected Shoreline')
    plt.plot(xi, py, color='blue', linewidth=2, label='Baseline', zorder=4)
    plt.title(('Transect Based Shoreline Detection (Time Averaged)\n' + stationname.capitalize() + 
            ' on ' + date + ' at ' + time[:2] + ':' + 
            time[2:] + ' UTC'), fontsize = 12)
    plt.legend(prop={'size': 9})
    plt.tight_layout()
    
    # Construct the save path for the figure
    base_dir = os.path.join(os.getcwd(), 'images', stationname, 'average')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        print(f"Directory exists: {base_dir}")

    print(f"Current Station Name from StationInfo: {stationname}")
    # print(f"Check Station Name from Input: {stationName}")
    saveName = os.path.join(base_dir, f'{stationname}-{date}_{time}.tranSL-avg.fix.jpeg')
    plt.savefig(saveName, bbox_inches='tight', dpi=400)
    plt.close()
    print(f"Saved fig_tranSL to: {saveName}")
    
    return fig_tranSL

def getTimexShoreline(stationName, imgName):
    # Main program.
    cwd = os.getcwd()
    stationPath = os.path.join(cwd, f'{stationName}.config.json')
    print(f"Current working directory: {cwd}")
    print(f"Station Path: {stationPath}")
#    if not os.path.exists(stationPath):
#        # Try the alternative path if the stationPath doesn't exist
#        stationPath = os.path.join(f'.\configs\{stationName}.config.json')
        
    stationInfo = getStationInfo(stationPath)
    dtObj = datetime.strptime(re.sub(r'\D', '', imgName), '%Y%m%d%H%M%S')
    stationInfo['Datetime Info'] = dtObj
    
    photoAvg = cv2.cvtColor(cv2.imread(imgName), cv2.COLOR_BGR2RGB)
    
    # If "Image Resize" is in the stationInfo, resize the image, i.e., 30 -> 0.3
    if 'Image Resize' in stationInfo:
        resize_factor = stationInfo['Image Resize']
        # convert to float from string
        resize_factor = float(resize_factor) / 100
        # Check if resize_factor is a valid number between 0 and 1
        if isinstance(resize_factor, (int, float)) and 0 < resize_factor <= 1:
            # Resize the image
            new_size = (int(photoAvg.shape[1] * resize_factor), int(photoAvg.shape[0] * resize_factor))
        else:
            print(f"Invalid resize factor: {resize_factor}. Skipping resizing.")
    else:
        # Default to 30% if not specified
        print("No resize factor specified. Defaulting to 30%.") 
        # Resizes image to 30% of original size.
        new_size = (int(photoAvg.shape[1] * 0.3), int(photoAvg.shape[0] * 0.3))
        
    resized_image = cv2.resize(photoAvg, new_size, interpolation=cv2.INTER_AREA)
    
    # Creating an array version of image dimensions for plotting.
    h, w = resized_image.shape[:2]
    xgrid, ygrid = np.linspace(0, w, w, dtype=int), np.linspace(0, h, h, dtype=int)
    X, Y = np.meshgrid(xgrid, ygrid, indexing = 'xy')
    
    # Maps regions of interest on plot.
    maskedImg = mapROI(stationInfo, resized_image)
    
    # Computes rmb.
    rmb = maskedImg[:,:,0] - maskedImg[:,:,2]
    P = improfile(rmb, stationInfo).reshape(-1, 1)
  
    # Computing probability density function and finds threshold points.
    pdfVals, pdfLocs = ksdensity(P)
    thresh_weightings = [(1/3), (2/3)]
    peaks = signal.find_peaks(pdfVals)
    peakVals = np.asarray(pdfVals[peaks[0]])
    peakLocs = np.asarray(pdfLocs[peaks[0]])  

    thresh_otsu = threshold_otsu(P)
    I1 = np.asarray(np.where(peakLocs < thresh_otsu))
    J1, = np.where(peakVals[:] == np.max(peakVals[I1]))
    I2 = np.asarray(np.where(peakLocs > thresh_otsu))
    J2, = np.where(peakVals[:] == np.max(peakVals[I2]))
    thresh = (thresh_weightings[0]*peakLocs[J1] +
            thresh_weightings[1]*peakLocs[J2])
    thresh = float(thresh)
    threshInfo = {
        'Thresh':thresh, 
        'Otsu Threshold':thresh_otsu,
        'Threshold Weightings':thresh_weightings
        }

    # Generates final json and figure for shoreline products.
    tranSL = extract(stationInfo, rmb, maskedImg, threshInfo)
    fig_tranSL = pltFig_tranSL(stationInfo, resized_image, tranSL)
    
    return(tranSL, fig_tranSL)

#######################################################################


station_name = 'jennette_south'
# station_name = 'currituck_sailfish_roi'

imgName = "timex.jennette_south-2025-05-12-113922Z.jpg"
tranSL, fig_tranSL = getTimexShoreline(station_name, imgName)