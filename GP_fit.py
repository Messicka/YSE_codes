#!/usr/bin/env python3

from taufit.taufit import *
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from astropy import units as u
import os

bands = ['g','r','i','z','y']
cols = ['dGal_num','filt','n_obs','baseline','log_tau','ltau_err','sigma_drw','sigma_drw_err','sigma_n','sigma_n_err']

var_file = "var_results.csv"
var_df = pd.read_csv(var_file)
qso_df = var_df[var_df['QSO']==1]
outfile = "ltau_fit.csv"
if os.path.isfile(outfile): results = pd.read_csv(outfile)
else: results = pd.DataFrame(columns=cols)

lc_dir = 'dGal'
i = results['dGal_num'].nunique()
l = qso_df['dGal_num'].nunique()
for num in qso_df['dGal_num'].unique():
    if num in results['dGal_num'].values: continue
    i += 1
    print(f"Working on {i} out of {l}")
    obj = f"dGal{num:05d}"
    infile = os.path.join(lc_dir, f"{obj}_phot.dat")
    try: df = pd.read_table(infile, sep='\s+').dropna()
    except:
        print(f"Empty dataset for {obj}")
        os.remove(infile)
        continue
    clean_df = df[(df['dq']==0) & (df['flux']>0)].sort_values('mjd').dropna()
    for color in bands:
        c_data = clean_df[clean_df['filt']==color]		#selecting data corresponding to one band
        if not c_data.size: continue
        clip = sigma_clip(c_data['flux'],5)                     #5-sigma clipping
        c_data = c_data[~clip.mask]                             #applying mask from clipping

        time, flux, f_err = c_data.loc[:,['mjd','flux','flux_err']].T.values
        n_obs = flux.size
        baseline = time.max() - time.min()
        if n_obs <= 10: continue		

        mag = 23.9 - 2.5 * np.log10(flux)
        mag_err = 2.5 * f_err / (np.log(10) * flux)
        gp, samples, _ = fit_drw(time*u.day, mag*u.mag, mag_err*u.mag, plot=False, verbose=False, target_name=obj)

        log_tau = -samples[:,1] / np.log(10)
        sigma_drw = np.sqrt(np.exp(samples[:,0]/2))
        sigma_n = np.exp(samples[:,2])

        row = pd.DataFrame([[num, color, n_obs, baseline, np.median(log_tau), log_tau.std(), np.median(sigma_drw),
					sigma_drw.std(), np.median(sigma_n), sigma_n.std()]], columns=cols)
        results = pd.concat((results,row), axis=0, ignore_index=True)
print(f"{l} out of {l} done")

results = results.sort_values('dGal_num')
results.to_csv(outfile, index=False)
