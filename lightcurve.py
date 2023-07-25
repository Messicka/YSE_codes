#!/usr/bin/env python3

#from astropy.table import Table
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(usage=
""" python lightcurve.py [-h] [-d <directory>] [-p <file prefix>] -n <galaxy number> [-k]
OR\npython lightcurve.py [-h] -f <file location> [-k]
""")
parser.add_argument('-d','--dir', type=str, default='dGal',
		help='directory containing photometry files')
parser.add_argument('-p','--prefix', type=str, default='dGal',
		help='file prefix for numbered sample: <prefix><number>_phot.dat')
parser.add_argument('-n','--number', type=int, default=None,
		help='galaxy number (if using a numbered sample)')
parser.add_argument('-f','--file', type=str, default=None,
		help='explicit file location')
parser.add_argument('-k','--keep_negs', default=False, action="store_true",
		help="if true, will keep all negative values and plot fluxes; \
if false, will discard all negative values and plot magnitudes")
args = parser.parse_args()

if args.file is None:
	if None in [args.prefix,args.number]:
		infile = input("Please specify a file location: ")
	else: infile = f"{args.dir}/{args.prefix}{args.number:05.0f}_phot.dat"
else: infile = args.file
while not os.path.isfile(infile): infile = input("No such file, try again: ")

df = pd.read_table(infile, sep='\s+')
clean_df = df[(df['dq']+df['gal_flag'])==0].sort_values('mjd').dropna()
if not args.keep_negs:
	clean_df = clean_df[clean_df['flux'] > 0]
	clean_df['mag'] = 23.9 - 2.5 * np.log10(clean_df['flux'])
	clean_df['mag_err'] = 2.5 * clean_df['flux_err'] / (np.log(10) * clean_df['flux'])

plt.figure()
bands_all = ['g', 'r', 'i', 'z', 'y']
colors_all = ['tab:green', 'tab:red', 'k', 'tab:blue', 'tab:pink']
bands = [x for x in bands_all if x in np.unique(clean_df['filt'])]
colors = [colors_all[i] for i, x in enumerate(bands_all) if x in bands]
for i, b in enumerate(bands):
	c_data = clean_df[clean_df['filt']==b]
	col = c=colors[i]
	if args.keep_negs:
		plt.errorbar('mjd', 'flux', yerr='flux_err', data=c_data, c=col)
		plt.axhline(c_data['gal_flux'].values[0], ls='--', c=col)
	else:
		plt.errorbar('mjd', 'mag', yerr='mag_err', data=c_data, c=col)
		plt.axhline(23.9-2.5*np.log10(c_data['gal_flux'].values[0]), ls='--', c=col)

ax = plt.gca()
x1, x2 = ax.get_xlim()
y1, y2 = ax.get_ylim()
ax.set_aspect(np.abs((x2-x1)/(y2-y1)))
if not args.keep_negs: ax.invert_yaxis()
plt.legend(bands,loc='lower left',bbox_to_anchor=(1.0,0.0))
plt.title(infile.rsplit('_',1)[0].rsplit('/',1)[1])
plt.xlabel('MJD')
plt.ylabel('Flux' if args.keep_negs else 'Mag')
plt.show()
