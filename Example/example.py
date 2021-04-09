import rasterio as rio
import numpy as np
from matplotlib import pyplot

## In case the code files were downloaded from Github, they can be imported
#  by using the sys and path:
# import sys
# sys.path.insert(1, 'PATH_TO_FOLDER/Descriptools')
# the import is as follows:

import descriptools.topoindexes as topoindexes
import descriptools.downslope as downslope
import descriptools.slope as slope
import descriptools.flowhand as flowhand
import descriptools.gfi as gfi
import descriptools.evaluation as evaluation

def main():
    """
    Example code.
    - Data import
    - Descriptor calculation step
    - Calibration and validation steps
    """
    ## ---------------------
    ## ---  Read inputs  ---
    ## ---------------------

    # Inputs are different for each descriptor.
    # Check each descriptor which input ir requires, so no extra memory is expended

    # Digital elevation model - DEM
    dem = rio.open('input/12_dem.tif').read(1).astype('int16')

    # Flow direction - D8
    fdr = rio.open('input/12_fdr.tif').read(1)

    # Flow accumulation
    fac = rio.open('input/12_fac.tif').read(1).astype('int')

    # no_data value correction
    dem = np.where(dem==dem[0,0],-100,dem)
    fac = np.where(fac==fac[0,0],-100,fac)

    # pixel size (important for some descriptors)
    px = 12.5

    # river mask from flow accumulation
    # the network initiation threshold should consider the pixel size
    # in this example, the initiation area is 20 km2 and pixel size is 12,5m
    # so the threshold is: (20000000/(12.5*12.5)) = 128000
    river = np.where(fac>128000, 1, 0).astype('int8') #20


    ## ---------------------
    ## --- descriptors  ---
    ## ---------------------
    # Slope. The returns are in %
    sl = slope.sloper(dem, px).astype('float32')

    # Topographic indexes
    # This descriptor requires slope. It needs to be transformed into radians
    sl = np.arctan(sl/100).astype('float32') # transformation from % to radian
    sl = np.where(dem==-100,-100,sl) #Correcting no_data values

    # For the moment the methods calculate both indexes.
    # Modified topographic index required an expoent n. For this example we use 0.1
    # The flow accumulation is not the area. The gpu function calculate the area
    TopoI, ModTi = topoindexes.topographic_index(fac, sl, px, 0.1)

    # Downslope, similar to slope. 
    # It is not in %. For this, multiply by 100
    # This descriptor requires a potential energy/height difference. We used 5 meters.
    down = downslope.downsloper(dem,fdr,px,5)

    # flow distance, HAND and river cell index
    # Flow distance to the nearest drainage and HAND follow the hydrological path
    #   to find the river cell that it drains to. Since this step is required for
    #   both indexes, it was devised to return both and do this only once.
    # The river cell position is required for others descriptors, and is also used
    #   to calculate the HAND.
    flow, indices, hand = flowhand.flow_hand_index(dem,fdr,river,px)

    # Geomorphic flood index - GFI
    # This descriptor require two variables. An expoent n and a scale factor b
    #   These values should com from the basin. In this example we use n=0.3 and b=0.1
    geofi = gfi.gfi_calculator(hand, fac, indices, 0.4, 0.1, px)

    # ln(hl/H)
    # Similar to the GFI. Also uses the n and b variables
    lnhlh = gfi.ln_hl_H_calculator(hand, fac, 0.4, 0.1, px)


    ## These methods also allow matrix partitioning, wirh arguments div_col and div_row
    ##  Both are defaulted as 0, but can be increased.
    ##  div_col refers to number of vertical divisions (columns)
    ##  div_row refers to number of horizontal divisions (rows)
    ##  It should be noted that HAND and downslope have special strategies, 
    ##  due to their localities steps

    ## --------------------------------
    ## -- Calibration and validation --
    ## --------------------------------

    # Bnchmark flood map is a binary map where 1 is flooded and 0 is not flooded
    flood = rio.open('input/WB_12_100y.tif').read(1).astype('int8')

    # In this example we will use HAND
    # It should follow this steps

    # 1. Normalizing. This should transform the matrix values to 0 and 1
    # 1.1 Min and max values need to be found first
    elements,count = np.unique(hand, return_counts=True)
    mx = elements[-1]
    mn = elements[1] # since no_data is -100, the min value is the second lowest

    # 1.2 defining no_data
    no_data = -100 

    # 1.3 Normalizing values
    desc = evaluation.minMaxScale(hand, mn, mx, no_data)

    # Note: This normalization need to consider the max and min values 
    #   of the intire dataset. Not only the calibration area.
    #   In this example we are doing this to show the steps to use the toolbox.
    #   If not, the threshold value of the calibration step will represent
    #   another non normalized value

    # 2. Calibration. This function returns a threshold value that find
    #   the best FIT index.
    # Part of the descriptors consider the values under the threhold as the one in danger
    #   while others is the contrary. For the HAND, the flood valures are under.

    # 2.1 Threshold value. 
    # th = evaluation.calibration_new(desc, flood, 'under')
    th = evaluation.calibration(desc, flood, 'under')

    # 2.2 Binary map. 1 flooded, 0 not flooded
    binary = evaluation.binary_map(desc, th, 'under')

    # 3. Perfomance. After calibration, the found value is used in the whole basin.
    #   For this example we use the same floodmap. Check the note for normalization.
    # c - Correctness index.
    # f - Fit index
    # class_map - Classified map: 0 true negative, 1 false positive, 
    #                             2 false negative, 3 true positive.
    c, f, class_map = evaluation.avaliacao(binary, flood)


    # -----------------------------------------------------------------------------
    # -- The descriptors can be visualized with matplotlib, in the following code -
    # -----------------------------------------------------------------------------
    
    sl = np.where(sl == -100,0,sl)
    pyplot.imshow(sl, cmap='BrBG_r')
    pyplot.show()  

    TopoI = np.where(TopoI == -100,0,TopoI)
    pyplot.imshow(TopoI, cmap='RdYlBu_r')
    pyplot.show()  

    ModTi = np.where(ModTi == -100,0,ModTi)
    pyplot.imshow(ModTi, cmap='RdYlBu_r')
    pyplot.show()  

    down = np.where(down == -100,0,down*100)
    pyplot.imshow(down, cmap='BrBG_r')
    pyplot.show()  

    flow = np.where(flow == -100,0,flow)
    pyplot.imshow(flow, cmap='Blues_r')
    pyplot.show()  

    hand = np.where(hand == -100,0,hand)
    pyplot.imshow(hand, cmap='Blues_r')
    pyplot.show()  

    geofi = np.where(geofi == -100,0,geofi)
    pyplot.imshow(geofi, cmap='RdYlBu_r')
    pyplot.show()  

    lnhlh = np.where(lnhlh == -100,0,lnhlh)
    pyplot.imshow(lnhlh, cmap='RdYlBu_r')
    pyplot.show()  

    # -------------------------------------------------------------
    # -- The binary maps can be visualized in the following code --
    # -------------------------------------------------------------

    pyplot.imshow(binary, cmap='RdYlBu_r')
    pyplot.show()  

    pyplot.imshow(class_map, cmap='RdYlBu_r')
    pyplot.show()  

    # -----------------------------
    # -- Exporting the binary map--
    # -----------------------------

    # get meta data from a reference tif file. In this case the DEM file
    meta = rio.open('input/12_dem.tif').meta

    # update type and no_data value
    meta.update(dtype=rio.uint8)
    meta.update(nodata=0)

    # reshape matrix to fit the .tif
    # class_map = class_map.astype('uint8')
    # class_map = class_map.reshape(1,len(class_map),len(class_map[0]))
    class_map = class_map.astype('uint8')
    class_map = class_map.reshape(1,len(class_map),len(class_map[0]))

    # Exporting with rasterio
    # out_file = 'class_map.tif' #Replace for the desired path
    out_file = 'output/hand_class.tif' #Replace for the desired path
    with rio.open(out_file, "w", **meta) as dist:
        dist.write(class_map.astype(rio.uint8))


if __name__ == "__main__":
    main()