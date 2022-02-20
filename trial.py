# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:03:20 2022
with rasterio.open('S2A_MSIL2A_20220122T012951_N0301_R074_T54SVG_20220122T034337_Chl.tiff', 'r') as ds:
    arr = ds.read()  # read all raster values
@author: acmor
"""
import gdal
import rasterio
import numpy as np
from rasterio.transform import Affine

dataset = rasterio.open('data\S2A_MSIL2A_20220122T012951_N0301_R074_T54SVG_20220122T034337_Chl.tiff')
Z=np.array(dataset.read())
Z=Z[0]
Z=np.nan_to_num(Z)
print(Z.shape)
new_dataset = rasterio.open( 'new.tif','w',driver='GTiff',
                            height=Z.shape[0],width=Z.shape[1],count=1,
                            dtype=Z.dtype,crs='+proj=latlong' )
print(Z)
new_dataset.write(Z,1)
new_dataset.close()