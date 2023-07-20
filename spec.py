

import numpy as np

#from dustmaps.config import config
#config.reset()

from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord

from hetdex_tools.get_spec import get_spectra
from hetdex_api.shot import Fibers

import pickle

import pandas as pd

values = pd.read_csv('/work/09464/kristinagatto/revisedlist.csv',header=None)

#imlist = []

imlist = {"id":np.array(values[5][1:]),
          "ra":np.array(values[1][1:]),
          "dec":np.array(values[2][1:]),
          "wave":np.array(values[3][1:]),
          "shot":np.array(values[4][1:]).astype(int),
          "fib2d": []    
}

image_filename = []

for i in range(1,len(values[1])):
    print(i)
    coord = SkyCoord(ra=(values[1][i])*u.deg, dec= (values[2][i])*u.deg)
    wave_obj = values[3][i]
    shotid = np.array(values[4][i], dtype = int)
    
    spec_tab = get_spectra(coords= coord, shotid=shotid, return_fiber_info=True, survey='hdr4', loglevel='WARNING', multiprocess=False)
    fiber_table = Table(spec_tab['fiber_info'][0], names=['fiber_id', 'multiframe', 'ra','dec','weight'], dtype=[str,str,float, float, float])
    fiber_table.sort( 'weight', reverse=True)
    F = Fibers(shotid, survey='hdr4')
    fiber_id = fiber_table['fiber_id'][0]
    
    width=100 #width of cutout in pixels in spectral direction
    height=9 #height in pixels in fiber profile direction


    im_fib = F.get_fib_image2D(
        wave_obj=wave_obj,
        fiber_id = fiber_table['fiber_id'][0],
        #fibnum_obj=fibnum_obj,
        #multiframe_obj=multiframe_obj,
        #expnum_obj=expnum_obj,
        width=width, 
        height=height,
    )
    
    im_sum = np.zeros_like(im_fib)

    for row in fiber_table:
        im_fib = F.get_fib_image2D(
            wave_obj=wave_obj,
            fiber_id = row['fiber_id'],
            #fibnum_obj=fibnum_obj,
            #multiframe_obj=multiframe_obj,
            #expnum_obj=expnum_obj,
            width=width, 
            height=height,
        )
        im_sum = im_sum + row['weight']*im_fib
        
        #imlist.append(im_sum)
        imlist['fib2d'].append(im_sum)
	#print(imlist['fib2d'])
       # print(np.array(values[1][i]))
       # print(imlist['id'][i])


with open('imfilelist.pkl','wb') as f:
        pickle.dump(imlist, f)
