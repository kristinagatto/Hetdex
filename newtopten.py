import numpy as np
#from matplotlib import pyplot as plt
from astropy.table import Table
import pandas as pd
import glob
from tqdm import tqdm
import os

colnames = ['wave', 'wave_err','flux','flux_err','linewidth','linewidth_err',
                    'continuum','continuum_err','sn','sn_err','chi2','chi2_err','ra','dec',
                    'datevshot','noise_ratio','linewidth_fix','chi2_fix', 'chi2fib',
                    'src_index','multiname', 'exp','xifu','yifu','xraw','yraw','weight',
                    'apcor','sn_cen', 'flux_noise_1sigma', 'sn_3fib', 'sn_3fib_cen']                               
             
gattomcfiles = (sorted(glob.glob('/scratch/09464/kristinagatto/alldet/detect_out/*.mc')))
gebhardtmcfiles = []  # (sorted(glob.glob('/scratch/00115/gebhardt/alldet/detect_out/20220430v009*.mc')))
#print(len(gattomcfiles1))
#print()
#print(len(gebhardtmcfiles1))
gebhardt = '/scratch/00115/gebhardt/alldet/detect_out/'


for f1 in gattomcfiles:
	fn = os.path.basename(f1)
	f2 = os.path.join(gebhardt,fn)

	if not os.path.exists(f2):
		print('Oh no!')
		exit(0)
	
	gebhardtmcfiles.append(f2)


def remove_similar_rows(larger_df, smaller_df, threshold_wave, threshold_ra_dec):
    similar_indices = []

    for _, row_smaller in smaller_df.iterrows():
        ra_smaller, dec_smaller, wave_smaller = row_smaller['ra'], row_smaller['dec'], row_smaller['wave']
        distances_ra_dec = np.sqrt(((larger_df['ra'] - ra_smaller) ** 2)*((np.cos(larger_df['dec']/(180/np.pi)))**2) + (larger_df['dec'] - dec_smaller) ** 2)
        distances_wave = np.abs(larger_df['wave'] - wave_smaller)
        similar_mask = (distances_ra_dec <= threshold_ra_dec) & (distances_wave <= threshold_wave)
        similar_indices.extend(larger_df[similar_mask].index.tolist())

    filtered_df = larger_df.drop(similar_indices)
    return filtered_df
    
    
    
def keep_top_10_sn_rows(dataset):
    nl = len(dataset)
    nl = min(nl,10)
    top_10_indices = dataset.nlargest(nl, 'sn').index
    filtered_dataset = dataset.loc[top_10_indices]
    return filtered_dataset
        
thefile = open("thefile.txt", "w+")

    
for i in tqdm(range(len(gattomcfiles))):
    data_dict = {}
 #   print('before')
    detectgatto = Table.read(gattomcfiles[i], format='ascii.no_header', names=colnames)
    detectgebhardt = Table.read(gebhardtmcfiles[i], format='ascii.no_header', names=colnames)
#    print('after')
    gattodf = detectgatto.to_pandas()
    gebhardtdf = detectgebhardt.to_pandas()
    

    newdata = remove_similar_rows(gattodf,gebhardtdf,3,2)
   # top10 = keep_top_10_sn_rows(newdata)
    data_dict[f'df_{i}'] = newdata
    specific_columns = ['ra', 'dec','wave','datevshot','linewidth','sn','continuum','chi2fib']  
    subset_df = newdata[specific_columns]
    
    
   # with open(f'thefile_{i}.txt', 'w') as file:
    thefile.write(subset_df.to_string(index=False,header=None))
    thefile.write('\n')

    

