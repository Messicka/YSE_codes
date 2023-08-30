#!/usr/bin/env python3

from qso_fit import *
import numpy as np
import pandas as pd
import os

bands = ['g','r','i','z','y']

target_file = "targets/YSE_targets.csv"
target_data = pd.read_csv(target_file)

lc_dir = 'dGal'
i,j = 0,0
for filename in os.listdir(lc_dir):
	obj = filename.split('_')[0]
	num = int(obj[-5:])
	infile = os.path.join(lc_dir, filename)
	df = pd.read_table(infile, sep='\s+').dropna()
	clean_df = df[(df['dq']==0) & (df['flux']>0)].sort_values('mjd')
	for color in bands:
		c_data = clean_df[clean_df['filt']==color]
		if not c_data.size: continue
		gal_flag = c_data['gal_flag'].values[0]
		time, flux, flux_err = c_data.loc[:,['mjd','flux','flux_err']].T.values
		mag = 23.9 - 2.5 * np.log10(flux)
		mag_err = 2.5 * flux_err / (np.log(10) * flux)
		mean_mag = 23.9 - 2.5 * np.log10(flux.mean())
		std_dev = mag.std()
		n = flux.size

		if n > 10:
			qso_dict = qso_fit(time, mag, mag_err, color)
			
			flux_var = flux.var()
			mean_err2 = (flux_err**2).mean()
			if flux_var >= mean_err2:
				mean2_flux = flux.mean()**2
				frac_var = np.sqrt((flux_var-mean_err2) / mean2_flux)
				NXS_err = np.sqrt((np.sqrt(2/n) + 4*frac_var**2/n) * mean_err2 / mean2_flux)
				var_unc = np.sqrt(frac_var**2 + NXS_err) - frac_var
				flag = 0
			else:
				frac_var = np.nan
				var_unc = np.nan
				flag = 1

			row = pd.DataFrame([num,color,mean_mag,std_dev,frac_var,var_unc,*qso_dict.values(),gal_flag,flag]).T
		else:
			row = pd.DataFrame([num,color,mean_mag,std_dev,*[np.nan]*12,gal_flag,1]).T
		
		results = row.copy() if i == 0 else pd.concat((results,row),axis=0,ignore_index=True)
		i += 1
	j += 1

results.columns = ['dGal_num','filt','mean_mag','mag_stddev','frac_var','frac_unc',*qso_dict.keys(),'obs_flag','flag']
results = results.sort_values('dGal_num')
results.to_csv('var_results.csv', index=False)
