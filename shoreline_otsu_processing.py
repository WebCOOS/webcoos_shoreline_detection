import os
from typing import Union
import numpy as np
from pathlib import Path
from result import ShorelineDetectionResult
from method_version import (
    MethodFramework,
    MethodName,
    ShorelineOtsuVersion,
)
from shorelines import Shoreline
from metrics import increment_shoreline_counter

# From getTimexShoreline
import cv2
from datetime import datetime, timezone
from itertools import chain
import json
# import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.path import Path as MatplotlibPath
import numpy as np
import os
from PIL import Image, ImageDraw
# import re
import scipy.signal as signal
from scipy.ndimage import binary_dilation
from skimage.filters import threshold_otsu
from skimage.measure import profile_line
from statsmodels.nonparametric.kde import KDEUnivariate

import logging

logger = logging.getLogger( __name__ )

CFG_FOLDER = Path(os.environ.get(
    "CFG_DIRECTORY",
    str(Path(__file__).parent / 'cfg')
))


class AbstractShorelineImplementation(object):

    def __init__(self) -> None:
        pass

    @classmethod
    def get_shoreline(cls, stationName, frame ):

        raise NotImplementedError(
            f"Must implement {cls.get_shoreline.__name__}(...) with subclass."
        )


class ShorelineOtsuMethodV1Implementation(AbstractShorelineImplementation):

    config_folder: str

    def __init__(self, config_folder: str ) -> None:

        assert config_folder
        assert Path( config_folder ).exists(), f"{config_folder} must exist"
        assert Path( config_folder ).is_dir(), f"{config_folder} must be dir"
        self.config_folder = config_folder

        return

    @classmethod
    def getStationInfo( cls, ssPath ):

        stationInfo = None
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

    @classmethod
    def mapROI( cls, stationInfo, photo):
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
        path = MatplotlibPath(roi_points)
        mask = path.contains_points(points).reshape(h, w)

        # Apply slight dilation to include edge pixels
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

    @classmethod
    def improfile( cls, rmb, stationInfo):
        # Extract intensity profiles along shoreline transects.
        transects = stationInfo['Shoreline Transects']
        xt = np.asarray(transects['x'], dtype=int)
        yt = np.asarray(transects['y'], dtype=int)
        n = len(xt)
        imProf = [profile_line(rmb, (yt[i, 1], xt[i, 1]), (yt[i, 0], xt[i, 0]), mode='constant') for i in range(n)]
        improfile = np.concatenate(imProf)[~np.isnan(np.concatenate(imProf))]
        return improfile

    @classmethod
    def ksdensity( cls, P, **kwargs):
        # Univariate kernel density estimation.
        x_grid = np.linspace(P.max(), P.min(), 1000)  # Could cache this.
        kde = KDEUnivariate(P)
        kde.fit(**kwargs)
        pdf = kde.evaluate(x_grid)
        return (pdf, x_grid)

    @classmethod
    def extract( cls, stationInfo, rmb, threshInfo):
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

            # Iterate through values to find the first crossing
            for i in range(1, len(values)):
                if (values[i-1] < threshold and values[i] >= threshold) or (values[i-1] >= threshold and values[i] < threshold):
                    return i
            return None

        # Checks orientation.
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
        except Exception:
            pass

        if isinstance(slVars['Shoreline Transects']['x'], np.ndarray):
            slVars['Shoreline Transects']['x'] = slVars['Shoreline Transects']['x'].tolist()
            slVars['Shoreline Transects']['y'] = slVars['Shoreline Transects']['y'].tolist()

        # JAR: Don't emit the JSON file here, instead return the slVars to the
        # calling function along with the shoreline points.

        # fname = (stationname + '.' + datetime.strftime(dtInfo,'%Y-%m-%d_%H%M') + '.avg.slVars.json')
        # with open(fname, "w") as f:
        #     json.dump(slVars, f)

        return ( shoreline, slVars )

    @classmethod
    def pltFig_tranSL( cls, stationInfo, photo, tranSL):
        stationname = stationInfo['Station Name']
        # dtInfo = stationInfo['Datetime Info']
        # date = str(dtInfo.date())
        # time = str(dtInfo.hour).zfill(2) + str(dtInfo.minute).zfill(2)  # Ensure two digits for hour and minute
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
            # print("Warning: No valid shoreline points after filtering")
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
        plt.title(
            (
                'Transect Based Shoreline Detection (Time Averaged)\n'
                + stationname.capitalize()
                # + ' on ' + date + ' at ' + time[:2] + ':' + time[2:] + ' UTC'
            ),
            fontsize = 12
        )
        plt.legend(prop={'size': 9})
        plt.tight_layout()

        # saveName = (stationname + '.' + date + '_' + time + '.' + 'tranSL-avg.fix.jpeg')
        # plt.savefig(saveName, bbox_inches = 'tight', dpi=400)
        # plt.close()
        return(fig_tranSL)

    @classmethod
    def getTimexShoreline(cls, config_folder: Union[str, Path], stationName, frame):
        # Main program.

        assert config_folder is not None
        assert isinstance( config_folder, ( str, Path ) )

        if isinstance( config_folder, str ):
            config_folder = Path( config_folder )

        # Check to make sure that this is one of the shorelines we're handling
        stationName_enum = Shoreline( stationName )

        stationPath = Path( config_folder ) / f"{stationName_enum.value}.config.json"

        stationInfo = cls.getStationInfo( stationPath)
        # TODO: Not sure how to handle date/time here, as previous implementation
        # relied on the name of the file object. Use 'now', for now.

        # dtObj = datetime.strptime(re.sub(r'\D', '', imgName), '%Y%m%d%H%M%S')
        dtObj = datetime.now( tz=timezone.utc )
        stationInfo['Datetime Info'] = dtObj

        # Converts image color scale.
        #new_size = (int(image.shape[1] * 0.3), int(image.shape[0] * 0.3))
        #resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        photoAvg = frame

        # By default, resize to 30% original size
        new_size = (int(photoAvg.shape[1] * 0.3), int(photoAvg.shape[0] * 0.3))
        # Unless otherwise defined in station config
        if 'Image Resize' in stationInfo:
            resize_factor = stationInfo['Image Resize']
            # convert to float from string
            resize_factor = float(resize_factor) / 100
            # Check if resize_factor is a valid number between 0 and 1
            if isinstance(resize_factor, (int, float)) and 0 < resize_factor <= 1:
                # Resize the image
                new_size = (int(photoAvg.shape[1] * resize_factor), int(photoAvg.shape[0] * resize_factor))

        resized_image = cv2.resize(photoAvg, new_size, interpolation=cv2.INTER_AREA)

        # Creating an array version of image dimensions for plotting.
        h, w = resized_image.shape[:2]
        xgrid, ygrid = np.linspace(0, w, w, dtype=int), np.linspace(0, h, h, dtype=int)
        X, Y = np.meshgrid(xgrid, ygrid, indexing = 'xy')

        # Maps regions of interest on plot.
        maskedImg = cls.mapROI(stationInfo, resized_image)

        # Computes rmb.
        rmb = maskedImg[:, :, 0] - maskedImg[:, :, 2]
        P = cls.improfile(rmb, stationInfo).reshape(-1, 1)

        # Computing probability density function and finds threshold points.
        pdfVals, pdfLocs = cls.ksdensity(P)
        thresh_weightings = [(1/3), (2/3)]
        peaks = signal.find_peaks(pdfVals)
        peakVals = np.asarray(pdfVals[peaks[0]])
        peakLocs = np.asarray(pdfLocs[peaks[0]])
        thresh_otsu = threshold_otsu(P)
        I1 = np.asarray(np.where(peakLocs < thresh_otsu))
        J1, = np.where(peakVals[:] == np.max(peakVals[I1]))
        I2 = np.asarray(np.where(peakLocs > thresh_otsu))
        J2, = np.where(peakVals[:] == np.max(peakVals[I2]))
        thresh = (
            thresh_weightings[0]
            * peakLocs[J1]
            + thresh_weightings[0]
            * peakLocs[J2]
        )
        thresh = float(thresh[0])
        threshInfo = {
            'Thresh': thresh,
            'Otsu Threshold': thresh_otsu,
            'Threshold Weightings': thresh_weightings
        }

        # Generates final json and figure for shoreline products.
        ( tranSL, slVars ) = cls.extract(stationInfo, rmb, threshInfo)
        fig_tranSL = cls.pltFig_tranSL(stationInfo, resized_image, tranSL)

        return ( tranSL, fig_tranSL, slVars )

    def get_shoreline( self, stationName, frame ):

        return self.__class__.getTimexShoreline(
            self.config_folder,
            stationName,
            frame
        )


SKIMAGE_METHODS = {
    # public-facing model name
    "shoreline_otsu": {
        # public-facing model version
        "v1": ShorelineOtsuMethodV1Implementation(
            config_folder=CFG_FOLDER
        )
    }
}

# Criteria for successful shoreline detections based on input imagery:
#
# 1) At least a certain number of detected shorline points
DEFAULT_MINIMUM_SHORELINE_DETECTION_POINTS = 5
# 2) TODO: error detection (stddev, etc.)


def shoreline_otsu_process_image(
    shoreline_method: AbstractShorelineImplementation,
    output_path: Path,
    model: Union[MethodName, str],
    version: Union[ShorelineOtsuVersion, str],
    shoreline_name: Union[Shoreline, str],
    name: str,
    bytedata: bytes,
    minimum_shoreline_points: int = DEFAULT_MINIMUM_SHORELINE_DETECTION_POINTS
) -> ShorelineDetectionResult:

    assert shoreline_method is not None, \
        f"Must have shoreline method passed to {shoreline_otsu_process_image.__name__}"

    assert isinstance( shoreline_method, AbstractShorelineImplementation ), \
        f"Shoreline method must implement {AbstractShorelineImplementation.__name__}"

    assert output_path and isinstance( output_path, Path ), \
        f"output_path parameter for {shoreline_otsu_process_image.__name__} is not Path"

    assert output_path.exists() and output_path.is_dir(), \
        (
            f"output_path parameter for {shoreline_otsu_process_image.__name__} must exist "
            "and be a directory"
        )

    assert isinstance( model, ( MethodName, str ) )
    assert isinstance( version, ( ShorelineOtsuVersion, str ) )

    assert minimum_shoreline_points is not None
    assert isinstance( minimum_shoreline_points, int )
    assert minimum_shoreline_points > 0

    if( isinstance( model, MethodName ) ):
        model = model.value

    if( isinstance( version, ShorelineOtsuVersion ) ):
        version = version.value

    assert isinstance( shoreline_name, ( Shoreline, str ) )

    if( isinstance( shoreline_name, Shoreline ) ):
        shoreline_name = shoreline_name.value

    ret: ShorelineDetectionResult = ShorelineDetectionResult(
        MethodFramework.skimage.name,
        model,
        version,
        shoreline_name
    )

    output_file_jpg = (
        output_path / model / version
        / Path( f"{shoreline_name}.{name}" ).with_suffix(
            '.transSL-avg.fix.jpg'
        )
    )

    # Ensure that the directory to house the plotted figure exists
    os.makedirs(
        str( output_file_jpg.parent ),
        exist_ok=True
    )

    npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # img_boxes = frame

    ( _, fig_tranSL, slVars ) = shoreline_method.get_shoreline(
        shoreline_name,
        frame
    )

    # Valid by default...
    ret.is_valid = True

    # Unless the validation criteria mark the result as invalid
    if len( slVars['Shoreline Points'] ) < minimum_shoreline_points:

        logger.warning(
            f"[{shoreline_name}] Less than {minimum_shoreline_points} shoreline"
            f" points ({len( slVars['Shoreline Points'] )}), marking shoreline"
            " detection as invalid."
        )
        ret.is_valid = False

    if ret.is_valid is True:
        # If still valid, take the time to write out the output image, otherwise
        # don't bother to generate a figure.

        assert isinstance( fig_tranSL, Figure )

        fig_tranSL.savefig(
            output_file_jpg,
            bbox_inches = 'tight',
            dpi = 400
        )

        ret.shoreline_plot_uri = str( output_file_jpg )

    ret.detected_shoreline = slVars

    increment_shoreline_counter(
        MethodFramework.skimage.value,
        MethodName.shoreline_otsu.value,
        ShorelineOtsuVersion.v1.value,
        shoreline_name
    )

    return ret
