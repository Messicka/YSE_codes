#!/usr/bin/env python
# D. Jones - 6/21/23
# A. Messick - 6/28/23
### Query IPP forced photometry for any given ra,dec

import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import argparse
import os
import astropy.table as at
import astropy.units as u
from astropy.io import fits
from astropy.visualization import PercentileInterval, AsinhStretch
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from photutils.aperture import SkyCircularAperture, ApertureStats
import time
from io import BytesIO
from lxml import html
import tempfile
import shutil
import pylab as plt
from astropy.time import Time

def date_to_mjd(date):
    time = Time(date,scale='utc')
    return time.mjd

def mjd_to_date(mjd):
    time = Time(mjd,format='mjd',scale='utc')
    return time.isot

### hack to get photometric band names - avoids one more API query ###
phot_band_dict = {'https://ziggy.ucolick.org/yse/api/photometricbands/119/':None,	#'i',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/120/':None,     #'z',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/211/':None,     #'g',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/212/':None,     #'r',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/216/':'g',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/217/':'r',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/218/':'i',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/219/':'z',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/220/':'y',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/36/':'g',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/37/':'r',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/39/':'i',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/40/':'z',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/41/':'y'}

### This is FITS Header boilerplate stuff from Mark ###
default_stamp_header = fits.Header()
default_stamp_header['XTENSION'] = 'BINTABLE'
default_stamp_header['BITPIX']  = 8
default_stamp_header['NAXIS']   = 2
default_stamp_header['NAXIS1']  = 476
default_stamp_header['NAXIS2']  = 9
default_stamp_header['PCOUNT']  = 0
default_stamp_header['GCOUNT']  = 1
default_stamp_header['TFIELDS'] = 24
default_stamp_header['TTYPE1']  = 'ROWNUM  '
default_stamp_header['TFORM1']  = 'J       '
default_stamp_header['TTYPE2']  = 'PROJECT '
default_stamp_header['TFORM2']  = '16A     '
default_stamp_header['TTYPE3']  = 'SURVEY_NAME'
default_stamp_header['TFORM3']  = '16A     '
default_stamp_header['TTYPE4']  = 'IPP_RELEASE'
default_stamp_header['TFORM4']  = '16A     '
default_stamp_header['TTYPE5']  = 'JOB_TYPE'
default_stamp_header['TFORM5']  = '16A     '
default_stamp_header['TTYPE6']  = 'OPTION_MASK'
default_stamp_header['TFORM6']  = 'J       '
default_stamp_header['TTYPE7']  = 'REQ_TYPE'
default_stamp_header['TFORM7']  = '16A     '
default_stamp_header['TTYPE8']  = 'IMG_TYPE'
default_stamp_header['TFORM8']  = '16A     '
default_stamp_header['TTYPE9']  = 'ID      '
default_stamp_header['TFORM9']  = '16A     '
default_stamp_header['TTYPE10'] = 'TESS_ID '
default_stamp_header['TFORM10'] = '64A     '
default_stamp_header['TTYPE11'] = 'COMPONENT'
default_stamp_header['TFORM11'] = '64A     '
default_stamp_header['TTYPE12'] = 'COORD_MASK'
default_stamp_header['TFORM12'] = 'J       '
default_stamp_header['TTYPE13'] = 'CENTER_X'
default_stamp_header['TFORM13'] = 'D       '
default_stamp_header['TTYPE14'] = 'CENTER_Y'
default_stamp_header['TFORM14'] = 'D       '
default_stamp_header['TTYPE15'] = 'WIDTH   '
default_stamp_header['TFORM15'] = 'D       '
default_stamp_header['TTYPE16'] = 'HEIGHT  '
default_stamp_header['TFORM16'] = 'D       '
default_stamp_header['TTYPE17'] = 'DATA_GROUP'
default_stamp_header['TFORM17'] = '64A     '
default_stamp_header['TTYPE18'] = 'REQFILT '
default_stamp_header['TFORM18'] = '16A     '
default_stamp_header['TTYPE19'] = 'MJD_MIN '
default_stamp_header['TFORM19'] = 'D       '
default_stamp_header['TTYPE20'] = 'MJD_MAX '
default_stamp_header['TFORM20'] = 'D       '
default_stamp_header['TTYPE21'] = 'RUN_TYPE'
default_stamp_header['TFORM21'] = '16A     '
default_stamp_header['TTYPE22'] = 'FWHM_MIN'
default_stamp_header['TFORM22'] = 'D       '
default_stamp_header['TTYPE23'] = 'FWHM_MAX'
default_stamp_header['TFORM23'] = 'D       '
default_stamp_header['TTYPE24'] = 'COMMENT '
default_stamp_header['TFORM24'] = '64A     '
default_stamp_header['EXTNAME'] = 'PS1_PS_REQUEST'
default_stamp_header['REQ_NAME'] = 'yse.meh_stamp_testid200410'
default_stamp_header['EXTVER']  = '2       '
default_stamp_header['ACTION']  = 'PROCESS '
default_stamp_header['EMAIL']   = 'yse@qub.ac.uk'

default_forcedphot_header = fits.Header()
default_forcedphot_header['XTENSION'] = 'BINTABLE'
default_forcedphot_header['BITPIX']   = 8
default_forcedphot_header['NAXIS']    = 2
default_forcedphot_header['NAXIS1']   = 84
default_forcedphot_header['NAXIS2']   = 8
default_forcedphot_header['PCOUNT']   = 0
default_forcedphot_header['GCOUNT']   = 1
default_forcedphot_header['TFIELDS']  = 9
default_forcedphot_header['TTYPE1']   = 'ROWNUM  '
default_forcedphot_header['TFORM1']   = '20A     '
default_forcedphot_header['TTYPE2']   = 'RA1_DEG '
default_forcedphot_header['TFORM2']   = 'D       '
default_forcedphot_header['TTYPE3']   = 'DEC1_DEG'
default_forcedphot_header['TFORM3']   = 'D       '
default_forcedphot_header['TTYPE4']   = 'RA2_DEG '
default_forcedphot_header['TFORM4']   = 'D       '
default_forcedphot_header['TTYPE5']   = 'DEC2_DEG'
default_forcedphot_header['TFORM5']   = 'D       '
default_forcedphot_header['TTYPE6']   = 'FILTER  '
default_forcedphot_header['TFORM6']   = '20A     '
default_forcedphot_header['TTYPE7']   = 'MJD-OBS '
default_forcedphot_header['TFORM7']   = 'D       '
default_forcedphot_header['TTYPE8']   = 'FPA_ID  '
default_forcedphot_header['TFORM8']   = 'J       '
default_forcedphot_header['TTYPE9']   = 'COMPONENT'
default_forcedphot_header['TFORM9']   = '64A     '
default_forcedphot_header['EXTNAME']  = 'MOPS_DETECTABILITY_QUERY'
default_forcedphot_header['QUERY_ID'] = 'yse.meh_det_test200410'
default_forcedphot_header['EXTVER']   = '2       '
default_forcedphot_header['OBSCODE']  = '566     '
default_forcedphot_header['STAGE']    = 'WSdiff  '
default_forcedphot_header['EMAIL']    = 'yse@qub.ac.uk'

def getskycell(ra,dec,user,pwd):
    """
    Finds the skycell from the PanSTARRS pipeline associated with a given RA and Dec

    parameters
    --------
    ra  : float; right ascension
    dec : float; declination
    user: str; IfA username
    pwd : str; IfA password

    returns
    --------
    skycell: str; name of the resulting skycell
    """
    session = requests.Session()
    session.auth = (user,pwd)
    skycellurl = 'http://pstamp.ipp.ifa.hawaii.edu/findskycell.php'
	
    # First login. Returns session cookie in response header. Even though status_code=401, it is ok
    page = session.post(skycellurl)
    info = {'ra': (None, ra), 'dec': (None, dec)}
    page = session.post(skycellurl, data=info)

    skycell = page.text.split("<tr><td>RINGS.V3</td><td>skycell.")[-1].split('</td>')[0]
    #xpos = page.text.split("<tr><td>RINGS.V3</td><td>skycell.")[-1].split('<td>')[1].split('</td>')[0]
    #ypos = page.text.split("<tr><td>RINGS.V3</td><td>skycell.")[-1].split('<td>')[2].split('</td>')[0]
	
    return skycell


### just a regex for the cameras using exp name ### 
def get_camera(exp_name):
    """
    Parses the PanSTARRS ceamera from an image_id
    
    parameters
    --------
    exp_name: str; PanSTARRS image ID

    returns
    --------
    cam: str; the camera that made the given observation
    """
    if exp_name[0] == 'o' and exp_name[-1] == 'o':
        if 'g' in exp_name: cam = 'GPC1'
        elif 'h' in exp_name: cam = 'GPC2'
        else: cam = 'None'

    return cam


### making PNG files ###
def fits_to_png(ff,outfile=None):
    """
    Creates a png image of an observation from a fits file and either saves or returns it

    parameters
    --------
    ff     : FITs object; a fits object of the observation
    outfile: str; a location to save the PNG file

    returns
    --------
    plt: matplotlib object; an object containing the created plot
    """
    plt.clf()
    ax = plt.axes()
    fim = ff[1].data
    # replace NaN values with zero for display
    fim[np.isnan(fim)] = 0.0
    # set contrast to something reasonable
    transform = AsinhStretch() + PercentileInterval(99.5)
    bfim = transform(fim)
    ax.imshow(bfim,cmap="gray",origin="lower")
    circle = plt.Circle((np.shape(fim)[0]/2-1, np.shape(fim)[1]/2-1), 15, color='r',fill=False)
    ax.add_artist(circle)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    if outfile is None: return plt
    else: plt.savefig(outfile,bbox_inches = 'tight',pad_inches = 0)

def fluxToMicroJansky(adu, exptime, zp):
    """
    Converts from Analog-to-Digital Units to microJanskys

    parameters
    --------
    adu    : float or array; observed ADU value(s)
    exptime: float or array; exposure time(s)
    zp     : float or array; zero-point(s)

    returns
    ---------
    uJy; float or array; the resulting fluxes in microJanskys
    """

    factor = 10**(0.4*(23.9-zp))
    uJy = adu * factor / exptime
    return uJy

def parse_mdc(mdc_text):
    """
    Parses the mdc(?) from the text given from text from the
    IPP stamps page

    parameters
    --------
    mdc_text: str; string of text from the IPP website

    returns
    --------
    mdc: dict; dictionary containing the individualized mdc(?)
    """

    mdc = {}
    for line in mdc_text.split('\n'):
        if line.startswith('ROWNUM'):
            rownum = int(line.split()[-1])
            mdc[rownum] = {}
        elif line.startswith('ROW'): continue
        elif line.startswith('END'): continue
        elif not line: continue
        else:
            mdc[rownum][line.split()[0]] = line.split()[2]
            
    return mdc

def getRADecBox(ra,dec,size=None):
    """
    Defines a box in coordinate space around a given RA and Dec

    parameters
    --------
    ra : float; right ascension
    dec: float; declination

    returns
    --------
    ramin : minimum right ascension
    ramax : maximum right ascension
    decmin: minimum declination
    decmax: maximum declination
    """
    RAboxsize = DECboxsize = size

    # get the maximum 1.0/cos(DEC) term: used for RA cut
    minDec = dec-0.5*DECboxsize
    if minDec<=-90.0:minDec=-89.9
    maxDec = dec+0.5*DECboxsize
    if maxDec>=90.0:maxDec=89.9

    invcosdec = max(1.0/np.cos(dec*np.pi/180.0),
                    1.0/np.cos(minDec  *np.pi/180.0),
                    1.0/np.cos(maxDec  *np.pi/180.0))

    ramin = ra-0.5*RAboxsize*invcosdec
    ramax = ra+0.5*RAboxsize*invcosdec
    decmin = dec-0.5*DECboxsize
    decmax = dec+0.5*DECboxsize

    if ra<0.0: ra+=360.0
    if ra>=360.0: ra-=360.0

    if ramin!=None:
        if (ra-ramin)<-180:
            ramin-=360.0
            ramax-=360.0
        elif (ra-ramin)>180:
            ramin+=360.0
            ramax+=360.0
    return(ramin,ramax,decmin,decmax)

class YSE_Forced_Pos:
    def __init__(self):
        pass

    def add_args(self,parser=None, usage=None):
        """
        Parses the arguments from the command line at runtime

        parameters
        --------
        parser: argparse ArgumentParser; initializaed parser object onto which to save the arguments
        usage : str; usage string to pass into parser initialization

        returns
        --------
        parser: argparse ArgumentParser; parser object with arguments attached
        """

        if parser == None:
            parser = argparse.ArgumentParser(
                usage=usage, conflict_handler="resolve")

        parser.add_argument('-n','--name', type=str, default=None,
                            help='give a name, just helps with bookkeeping')
        parser.add_argument('-r','--ra', type=float, default=None,
                            help='RA')
        parser.add_argument('-d','--dec', type=float, default=None,
                            help='Dec')
        parser.add_argument('-c','--coordlist', type=str, default=None,
                            help='file location of coordinate list')
        parser.add_argument('-o','--outdir', type=str, default=None,
                            help='output directory for images and photometry files')
        parser.add_argument('--use_csv', default=False, action="store_true",
                            help="use CSV field for field info if set")
        parser.add_argument('--csv_filename', default='YSE_observations_full.csv', type=str,
                            help="CSV file name")
        parser.add_argument('--nofitsdownload', default=False, action="store_true",
                            help="don't download fits files if set")
        parser.add_argument('--checkfiles', default=False, action="store_true",
                            help="won't analyze objects for which files already exist")
        parser.add_argument('--keepnans', default=False, action="store_true",
                            help="will keep nan values in output files if set")
        parser.add_argument('--ysepzurl',type=str,default='https://ziggy.ucolick.org/yse',
                            help='base URL for YSE-PZ, so you can reach the API and get field IDs')
        parser.add_argument('-u','--ysepzuser',type=str,default=None,
                            help='YSE-PZ username')
        parser.add_argument('-p','--ysepzpass',type=str,default=None,
                            help='YSE-PZ password')
        parser.add_argument('--ifauser',type=str,default='ps1sc',
                            help='IfA username')
        parser.add_argument('--ifapass',type=str,default='skysurveys',
                            help='IfA password')

        return parser

    def request_forcedphot(self,targ_table,obs_table,skycelldict):
        """
        Inputs a table of transients, finds the corresponding difference images, and sends the request to the IPP

        parameters
        --------
        targ_table : astropy Table; contains the transient IDs, RAs, and Decs
        obs_table  : astropy Table; contains the names, coordinates (ra & dec), filters, diff IDs,
                     mjds, and cameras for all the images for each transient
        skycelldict: dict; contains the skycells for the images

        returns
        --------
        request_names: list of str; list of requests 
        """
        
        request_names = []
        diff_count =  0
        diff_hdr = at.Table(names=('ROWNUM','PROJECT','RA1_DEG','DEC1_DEG','RA2_DEG','DEC2_DEG','FILTER','MJD-OBS','FPA_ID','COMPONENT_ID'),
                             dtype=('S20','S16','>f8','>f8','>f8','>f8','S20','>f8','>i4','S64'))
        for trans_id in targ_table['transient_id']:
            diff_data = diff_hdr.copy()
            for name,ra,dec,filt,diff_id,mjd,cam in obs_table:
                if name == trans_id:
                    diff_data.add_row((f'forcedphot_ysebot_{diff_count}',cam,ra,dec,ra,dec,filt,mjd,diff_id,skycelldict[name]))
                    diff_count += 1

            if len(diff_data):
                request_name = f'YSE-phot.{trans_id}.{diff_id}.{int(time.time())}'
                request_names += [request_name]
                hdr = default_forcedphot_header.copy()
                hdr['QUERY_ID'] = request_name
                hdr['EXTNAME'] = 'MOPS_DETECTABILITY_QUERY'
                hdr['EXTVER'] = '2'
                hdr['OBSCODE'] = '566'
                hdr['STAGE'] = 'WSdiff'
                ff = fits.BinTableHDU(diff_data, header=hdr)
                s = BytesIO()
                ff.writeto(s, overwrite=True)
                self.submit_to_ipp(s)
            else:
                print(f'No diff IDs for transient {trans_id}!')

        return request_names

    def request_templates(self,trans_data,skycelldict,width=300,height=300):
        """
        Requests the templates/stack images for given transient objects

        parameters
        --------
        trans_data : astropy Table; contains the names, stack IDs, coordinates (ra & dec),
                     and cameras for the transients
        skycelldict: dict; contains the skycells for the images
        width      : float; width of the requested stamp image
        height     : float; height of the requested stamp image

        returns
        ---------
        requestname: str; name of the request sent to the IPP
        """

        data = at.Table(names=('ROWNUM', 'PROJECT', 'SURVEY_NAME', 'IPP_RELEASE', 'JOB_TYPE',
                               'OPTION_MASK', 'REQ_TYPE', 'IMG_TYPE', 'ID', 'TESS_ID',
                               'COMPONENT', 'COORD_MASK', 'CENTER_X', 'CENTER_Y', 'WIDTH',
                               'HEIGHT', 'DATA_GROUP', 'REQFILT', 'MJD_MIN', 'MJD_MAX',
                               'RUN_TYPE', 'FWHM_MIN', 'FWHM_MAX', 'COMMENT'),
                        dtype=('>i4','S16','S16','S16','S16','>i4','S16','S16','S16','S64',
                               'S64','>i4','>f8','>f8','>f8','>f8','S64','S16','>f8','>f8',
                               'S16','>f8','>f8','S64'))

        count = 1
        for trans_id,stack_id,ra,dec,camera in trans_data:
            if stack_id in [None,'None','NULL']: continue
            skycell_str = skycelldict[trans_id]
            data.add_row((count,camera,'null','null','stamp',2049,'byid','stack',stack_id,'RINGS.V3',
                          skycell_str,2,ra,dec,width,height,'null','null',0,0,'null',0,0,f'stack.for.{trans_id}'))
            count += 1

        hdr = default_stamp_header.copy()
        request_name = f'YSE-stamp.{int(time.time())}'
        hdr['REQ_NAME'] = request_name
        ff = fits.BinTableHDU(data, header=hdr)

        s = BytesIO()
        ff.writeto(s, overwrite=True)
        self.submit_to_ipp(s)
        return request_name


    def get_phot(self,request_names,trans_data):
        """
        Parses the results of the difference image request, and saves the flux and other information

        parameters
        --------
        request_names: list of str; contains the names of the phot requests previously sent to the IPP
        trans_data   : astropy Table; contains the names, coordinates (ra & dec), filters, diff IDs,
                       mjds, and cameras for the observations of each transient

        returns
        --------
        phot_table: astropy Table; contains the names, stack IDs, mjd, filters, exposure times, zero-
                    points, difference flux and error, a quality flag, coordinates (ra & dec), and
                    cameras for the resulting difference images
        """
        phot_table = at.Table(names=('transient_id','stack_id','mjd','filt','exp_time',
                                     'zpt','diff_flux','diff_err','dq','ra','dec','camera'),
                              dtype=(str,str,float,str,*[float]*7,str))
        phot_link = 'http://datastore.ipp.ifa.hawaii.edu/pstampresults/'
        rn_len = len(request_names)
        for i, request_name in enumerate(request_names):
            trans_id,trans_ra,trans_dec = trans_data[i]
            sct = SkyCoord(trans_ra,trans_dec,unit=u.deg)
            print(f"Getting photometry: {i} out of {rn_len} done", end='\r')
            phot_results_link = f'{phot_link}/{request_name}/'
            phot_page = requests.get(url=phot_results_link)
            if phot_page.status_code != 200:
                raise RuntimeError('Results page {phot_results_link} does not exist')

            tree = html.fromstring(phot_page.content)
            fitsfiles = tree.xpath('//a/text()')
            for f in fitsfiles:
                if 'detectability' in f:
                    phot_fits_link = f'{phot_results_link}/{f}'
                    fits_response = requests.get(url=phot_fits_link,stream=True)

                    # this is a pain but it seems necessary - D. Jones
                    tmpfits = tempfile.NamedTemporaryFile(delete=False)
                    shutil.copyfileobj(fits_response.raw, tmpfits)
                    tmpfits.close()
                    ff = fits.open(tmpfits.name)
                    os.remove(tmpfits.name)
                    for j in range(len(ff[1].data)):
                        mjd = ff[0].header['MJD-OBS']
                        exptime = ff[0].header['EXPTIME']
                        filt = ff[0].header['FPA.FILTER'].split('.')[0]
                        zpt = ff[0].header['FPA.ZP']
                        factor = fluxToMicroJansky(1, exptime, zpt)
                        flux = factor * ff[1].data['PSF_INST_FLUX'][j]
                        flux_err = factor * ff[1].data['PSF_INST_FLUX_SIG'][j]
                        cam = ff[0].header['FPA.INSTRUMENT']
                        # http://svn.pan-starrs.ifa.hawaii.edu/trac/ipp/browser/trunk/psModules/src/objects/pmSourceMasks.h?order=name
                        # FLAGS corresponding to diff spike, ghost, off chip, saturated, defect, saturation
                        # 0x20000000, 0x40000000, 0x80000000, 0x00001000, 0x00000800, 0x00000080
                        if ff[1].data['PSF_QF'][j] < 0.9 or \
                           (ff[1].data['FLAGS'][j] & 0x20001880) or \
                           (ff[1].data['FLAGS'][j] & 0x40001880) or \
                           (ff[1].data['FLAGS'][j] & 0x80001880): dq = 1.0
                        else: dq = 0.0
                        try: stack_id = ff[0].header['PPSUB.REFERENCE'].split('.')[-3]
                        except: stack_id = None

                        ra, dec = ff[1].data['RA_PSF'][j], ff[1].data['DEC_PSF'][j]
                        sc = SkyCoord(ra,dec,unit=u.deg)
                        sep = sc.separation(sct).arcsec

                        if np.min(sep) > 2: print(f'Couldn\'t find transient match for RA,Dec={ra:.7f},{dec:.7f}!')
                        elif not self.options.keepnans and np.isnan((flux,flux_err)).sum(): pass
                        elif stack_id not in [None, 'NULL']: phot_table.add_row((trans_id,stack_id,mjd,filt,exptime,zpt,flux,flux_err,dq,ra,dec,cam))

        print(f"Getting photometry: {rn_len} out of {rn_len} done")
        return phot_table

    def write_photometry(self,phot_table,stack_table):
        """
        Combines the forced photometry and stack photometry tables together before saving the contents in a file

        parameters
        --------
        phot_table : astropy Table; contains the results of the difference image photometry
        stack_table: astropy Table; contains the results of the stack image photometry
        """

        gals = np.unique(stack_table['transient_id'])
        for gal in gals:
            phot_match = phot_table['transient_id'] == gal
            stack_match = stack_table['transient_id'] == gal
            if not phot_match.sum() * stack_match.sum(): print(f"{'Diff' if stack_match.sum() else 'Stack'} data for {gal} empty!"); continue
            data_table = at.join(phot_table[phot_match]['stack_id','mjd','filt','exp_time','zpt','diff_flux','diff_err','dq'],
                                 stack_table[stack_match]['stack_image_id','gal_flux','gal_err','gal_flag'],
                                 keys_left='stack_id',keys_right='stack_image_id')
            if not len(data_table): print(f"No stack and difference image matches for {gal}!"); continue
            data_table['flux'] = data_table['gal_flux'] + data_table['diff_flux']
            data_table['flux_err'] = np.sqrt(data_table['gal_err']**2 + data_table['diff_err']**2)

            with open(f"{self.options.outdir}/{gal}_phot.dat",'w') as fout:
                print("mjd filt exp_time zpt diff_flux diff_err gal_flux gal_err flux flux_err gal_flag dq",file=fout)
                for _,m,f,exp,zpt,df,dfe,dq,_,gf,gfe,flag,fl,fle in data_table:
                    print(f"{m:.3f} {f} {exp} {zpt:.4f} {df:.4f} {dfe:.4f} {gf:.4f} {gfe:.4f} {fl:.4f} {fle:.4f} {flag} {dq}",file=fout)
        return

    def get_stamps(self,request_name,coord_data,warning=False):
        """
        Parses the results of the requested stack stamps, collecting the image link and other
        information in a table

        parameters
        --------
        request_name: str; name of stack images request sent to IPP
        coord_data  : astropy Table; contains the transient name, ra, and dec

        returns
        --------
        image_dict: dict; contains the IDs and links for the requested images (as well
                    as coordinates)
        """

        stamp_link = 'http://datastore.ipp.ifa.hawaii.edu/yse-pstamp-results/'
        stamp_fitsfile_link = stamp_link + f'{request_name}/'
        stamp_results_link = stamp_fitsfile_link + 'results.mdc'
        
        stamps_page = requests.get(url=stamp_results_link)
        if stamps_page.status_code != 200: raise RuntimeError(f'Results page {stamp_results_link} does not exist')
        mdc_stamps = parse_mdc(stamps_page.text)

        image_table = at.Table(names=('transient_id','stack_image_link','stack_image_id','ra','dec'),dtype=(str,str,str,float,float))
        for k in mdc_stamps.keys():
            err = mdc_stamps[k]['ERROR_STR']
            if 'SUCCESS' not in err:
                if warning: print(f'Warning: part of job {request_name} failed, {k} has error {err}')
            else:
                img_name, img_type = mdc_stamps[k]['IMG_NAME'], mdc_stamps[k]['IMG_TYPE']
                transient, img_id = mdc_stamps[k]['COMMENT'].split('.')[-1], mdc_stamps[k]['ID']
                match = coord_data['transient_id'] == transient
                _,ra,dec = coord_data[match].values()
                if img_type == 'stack' and img_id not in image_table['stack_image_id']:
                    link = f'{stamp_fitsfile_link}/{img_name}'
                    image_table.add_row((transient,link,img_id,ra,dec))
                elif img_type not in ['stack','warp','diff']: 
                    raise RuntimeError(f'image type {img_type} not found')

        return image_table
 
    def get_status(self,request_name,warning=False):
        """
        Retrieves the status of a request sent to the IPP

        parameters
        --------
        request_name: str; name of the request previously sent to the IPP

        returns
        --------
        done: bool; boolean indicating whether or not the request is ready
        """

        status_link = 'http://pstamp.ipp.ifa.hawaii.edu/status.php'
        session = requests.Session()
        session.auth = (self.options.ifauser,self.options.ifapass)
        session.post(status_link)					#First session returns cookie in header
        page = session.post(status_link)				#Second session returns page
        
        if page.status_code == 200:
            lines_out = []
            for line in page.text.split('<pre>')[-1].split('\n'):
                if line and '------------------' not in line and '/pre' not in line:
                    lines_out += [line[1:]]
            text = '\n'.join(lines_out)
            tbl = at.Table.read(text,format='ascii',delimiter='|',data_start=1,header_start=0)

            idx = tbl['name'] == request_name
            if not idx.sum():
                if warning: print(f'warning: could not find request named {request_name}')
                return False
            if tbl['Completion Time (UTC)'][idx]: done = True
            else: done = False

            jobs = float(tbl['Total Jobs'][idx])
            jobs_succ = float(tbl['Successful Jobs'][idx])
            if jobs_succ != jobs and warning: print(f'warning: {jobs-jobs_succ} of {jobs} jobs failed')
        else:
            print(f'Error occured with request {request_name}')
            done = False
        return done

    def get_gal_flux(self,img_data):
        """
        Downloads each stack image stamp, performs aperture photometry, and returns the results

        parameters
        --------
        img_data: astropy Table; contains the name, stack image link and ID, and coordinates (ra & dec)
                  for the transients

        returns
        --------
        gal_table: astropy Table; contains the names, stack IDs, stack flux and error, and a quality flag
                   for each of the transients
        """

        basedir = self.options.outdir
        session = requests.Session()
        session.auth = (self.options.ifauser,self.options.ifapass)

        gal_table = at.Table(names=('transient_id','stack_image_id','gal_flux','gal_err','gal_flag'),dtype=(str,str,float,float,float))
        for trans_id,link,img_id,ra,dec in img_data:
            if img_id is None: continue 
            outdir = f"{basedir}/{trans_id}/{img_id}"
            if not os.path.exists(outdir): os.makedirs(outdir)

            filename = link.split('/')[-1]
            outfile = f"{outdir}/{filename}"

            fits_response = session.get(link,stream=True)
            with open(outfile,'wb') as fout: shutil.copyfileobj(fits_response.raw, fout)
            ff = fits.open(outfile)

            pos = SkyCoord(ra,dec,unit=u.deg)
            ap = SkyCircularAperture(pos, r = 2.5*u.arcsec)
            ap_pix = ap.to_pixel(WCS(ff[1].header)) 
            apstats = ApertureStats(ff[1].data, ap_pix)
            factor =  10**0.44 * ff[1].header['EXPTIME']

            flux = apstats.sum / factor
            flux_err = apstats.std / factor
            flag = 1.0 if apstats.sum/apstats.std < 5 else 0.0
            gal_table.add_row((trans_id,img_id,flux,flux_err,flag))

            if self.options.nofitsdownload: shutil.rmtree(outdir)
            else: fits_to_png(ff,outfile.replace('fits','png'))

        gals = np.unique(gal_table['transient_id'])
        for gal in gals:
            #match = gal_table['transient_id'] == gal
            #flag_set = gal_table[match]['gal_flag']
            #if np.prod(flag_set): gal_table = gal_table[~match]
            updir = f"{basedir}/{gal}"
            if not len(os.listdir(updir)): os.rmdir(updir)

        return gal_table

    def submit_to_ipp(self,filename_or_obj):
        """
        Submits a request to the IPP from a file or fits table object

        parameters
        ---------
        filename_or_obj: str or fits file object: reuest to be sent to the IPP
        """

        session = requests.Session()
        session.auth = (self.options.ifauser,self.options.ifapass)
        stampurl = 'http://pstamp.ipp.ifa.hawaii.edu/upload.php'

        # First login. Returns session cookie in response header. Even though status_code=401, it is ok
        page = session.post(stampurl)

        if type(filename_or_obj) == str: files = {'filename':open(filename,'rb')}
        else: files = {'filename':filename_or_obj.getvalue()}
        page = session.post(stampurl, files=files)
    
    def main(self):
        """
        Parses the information given at runtime and sends the target transients to be analyzed
        """

        if self.options.coordlist is not None:
            targets = at.Table.read(self.options.coordlist, names=['transient_id','ra','dec'], data_start=0)
            keep = [not os.path.exists(f'{self.options.outdir}/{obj}_phot.dat') for obj in targets['transient_id']] if self.options.checkfiles else np.ones_like(targets['transient_id'], dtype=bool)
            if np.sum(keep): self.all_stages(targets[keep])
        else:
            if self.options.checkfiles and os.path.exists(f'{self.options.outdir}/{self.options.name}_phot.dat'): pass
            else: self.all_stages(at.Table([[self.options.name],[self.options.ra],[self.options.dec]], names=['transient_id','ra','dec']))

    def all_stages(self,targets):
        """ 
        Performs all of the functions on a list of transients to analyze, saving the resulting fluxes in
        files for each object

        parameters
        ---------
        targets: astropy Table; contains the names and coordinates (ra & dec) of each transient
        """

        ### 1. query the YSE-PZ API for all obs matching ra/dec ###
        # this might be slow if the list is long!
        if self.options.use_csv:
            csvdata = at.Table.read(self.options.csv_filename,format='ascii.csv')
            scall = SkyCoord(csvdata['radeg'],csvdata['decdeg'],unit=u.deg)

        skycelldict = {}
        obs_data = at.Table(names=('transient_id','ra','dec','filt','diff_id','obs_mjd','camera'),dtype=(str,float,float,str,str,float,str))
        for name,ra,dec in targets:
            # construct an ra/dec box search
            # PanSTARRS images are 3.1x3.1 deg, approximately
            if self.options.use_csv:
                sc = SkyCoord(ra,dec,unit=u.deg)
                iClose = (sc.separation(scall).deg < 1.65) & np.isin(csvdata['diff_id'],[None,'None','NULL'],invert=True)
                if iClose.sum() > 1000: raise RuntimeWarning("There are more than 1000 images containing this coordinate!")
                elif iClose.sum(): skycelldict[name] = 'skycell.'+getskycell(ra,dec,self.options.ifauser,self.options.ifapass)
                else: print(f"Object {name} not found in YSE fields")
                for c in np.unique(csvdata[iClose]):
                    cam = get_camera(c['exp_name'])
                    filt = c['filter'][0]
                    mjd = date_to_mjd(c['dateobs'])
                    diff_id = c['diff_id']
                    obs_data.add_row((name,ra,dec,filt,diff_id,mjd,cam))

            else:
                ramin,ramax,decmin,decmax = getRADecBox(ra,dec,size=3.0)

                # check ra min/max are sensible, otherwise we have to be more clever
                if ramin > 0 and ramax < 360:
                    query_str = f"ra_gt={ramin}&ra_lt={ramax}&dec_gt={decmin}&dec_lt={decmax}&status_in=Successful&limit=100"
                    obs_data_response = requests.get(
                        f"{self.options.ysepzurl}/api/surveyobservations/?{query_str}",
                        auth=HTTPBasicAuth(self.options.ysepzuser,self.options.ysepzpass))

                    if obs_data_response.status_code != 200:
                        raise RuntimeError('Issue communicating with the YSE-PZ server')
                    obs_data_results = obs_data_response.json()['results']

                else:
                    ramin %= 360
                    ramax %= 360
                    # might have to do two queries
                    query_str_1 = f"ra_gt={ramin}&dec_gt={decmin}&dec_lt={decmax}&status_in=Successful&limit=100"
                    query_str_2 = f"ra_lt={ramax}&dec_gt={decmin}&dec_lt={decmax}&status_in=Successful&limit=100"

                    obs_data_response_1 = requests.get(
                        f"{self.options.ysepzurl}/api/surveyobservations/?{query_str_1}",
                        auth=HTTPBasicAuth(self.options.ysepzuser,self.options.ysepzpass))
                    obs_data_response_2 = requests.get(
                        f"{self.options.ysepzurl}/api/surveyobservations/?{query_str_2}",
                        auth=HTTPBasicAuth(self.options.ysepzuser,self.options.ysepzpass))

                    if obs_data_response_1.status_code != 200 or obs_data_response_2.status_code != 200:
                        raise RuntimeError('issue communicating with the YSE-PZ server')
                    obs_data_1 = obs_data_response_1.json()
                    obs_data_2 = obs_data_response_2.json()
                    obs_data_results = np.append(obs_data_1['results'],obs_data_2['results'])

                image_counts = 0
                for c in np.unique(at.Table(obs_data_results)):
                    cam = get_camera(c['image_id'])
                    filt = phot_band_dict[c['photometric_band']]
                    mjd = c['obs_mjd']
                    diff_id = c['diff_id']
                    if filt and diff_id: obs_data.add_row((name,ra,dec,filt,diff_id,mjd,cam)); image_counts += 1
                if image_counts > 1000: raise RuntimeWarning("There are more than 1000 images containing this coordinate!")

        ### 2. stamp images & photometry ###
        if not len(obs_data): print('No observations found!'); return
        observed = [obs in np.unique(obs_data['transient_id']) for obs in targets['transient_id']]
        targets = targets[observed]
        phot_request_names = self.request_forcedphot(targets,obs_data,skycelldict)
        
        ### 2c) check status of the stamp images/photometry jobs ###
        max_time = 15
        print(f'Jobs were submitted, waiting up to {max_time} minutes for them to finish...')
        # wait until the jobs are done
        jobs, jobs_done = len(phot_request_names), 0
        tstart = time.time()
        while jobs_done < jobs and time.time()-tstart < max_time*60:
            time.sleep(60)
            jobs_done = 0
            for phot_request_name in phot_request_names: jobs_done += self.get_status(phot_request_name)
            print(f"Requesting images: {jobs_done} out of {jobs} done", end='\r')
        print(f"Requesting images: {jobs_done} out of {jobs} done")

        if jobs_done < jobs: raise RuntimeError('Diff request timeout!')

        ### 2d) download the stamp images
        # save the data
        diff_table = self.get_phot(phot_request_names,targets)
        if not len(diff_table): print('No difference images found!'); return

        ### 3. template images
        ### 3a) request the templates
        stack_table = at.Table(np.unique(diff_table['transient_id','stack_id','ra','dec','camera']))
        stack_request_name = self.request_templates(stack_table,skycelldict)
        print(f'Submitted stack request {stack_request_name}...')

        ### 3b) check status of the templates
        job_done = False
        tstart = time.time()
        while not job_done and time.time()-tstart < max_time*60:
            time.sleep(60)
            job_done = self.get_status(stack_request_name)
        if not job_done: raise RuntimeError('Stack request timeout!')

        ### 3c) download the stamp images
        print("Downloading stamp images...")
        stack_img_data = self.get_stamps(stack_request_name, targets)
        if not len(stack_img_data): print("No stack data found!"); return
        gal_table = self.get_gal_flux(stack_img_data)
        self.write_photometry(diff_table, gal_table)        

        return
     
if __name__ == "__main__":
    usagestr = """
Tool for getting YSE (IPP) forced photometry for a given coordinate or set of coordinates.
Takes an individual name/RA/dec (decimal degrees) or a list of name/RA/dec (comma-delimited, ra/dec in decimal degrees).
    Having a unique name for each ra/dec coord is just for bookkeeping purposes.
    
Usage:
    python YSE_Forced_Position.py -n <name> -r <ra> -d <dec> --use_csv
    python YSE_Forced_Position.py -n <name> -r <ra> -d <dec> -u <YSE-PZ username> -p <YSE-PZ password>
    python YSE_Forced_Position.py -c <coordinate list> --use_csv
    python YSE_Forced_Position.py -c <coordinate list> -u <YSE-PZ username> -p <YSE-PZ password>
""" 

    ys = YSE_Forced_Pos()

    parser = ys.add_args(usage=usagestr)
    args = parser.parse_args()
    ys.options = args

    if ys.options.coordlist is None:
        if None in [ys.options.ra,ys.options.dec,ys.options.name]:
            raise RuntimeError('Must specify name/ra/dec or coordlist arguments, run:\n\tYSE_Forced_Position.py --help\nfor more info')
    elif not os.path.exists(ys.options.coordlist):
        raise RuntimeError(f'Coordinate list file {ys.options.coordlist} does not exist')
    
    ys.main()

