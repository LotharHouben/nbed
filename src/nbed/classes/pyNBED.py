import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pims
import trackpy as tp
import time
from IPython import display
import os
from tqdm import tqdm 
from ..helpers import ParabolaFit2D,convolve2D,read_empad,bytscl

# A processing class for 4D-STEM nanodiffraction data

class pyNBED:
    
    def __init__(self):
        self.data = None
        self.fname='' 
        self.type=''
        self.dim=[]
        self.peakdetpar={'feat_sz':7,'feat_minmass':1,'feat_sep':6,'feat_perc':50.,'noise_sz':1,'thresh':1,'smooth_sz':None,'cmin':1.,'cmax':10,'conv_kernel':1./9*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),'toggleconv':True}        # keep noise_sz at 1.0, filtering means somehow that the peaks are displaced
        self.metadata=None
        self.qsamplinggrid=None
        self.qsamplinggrid2=None

    def PrepareSamplingGrid(self):
        #
        # prepare the sampling grids in reciprocal space and their powers
        # results are stored in self.distpow1rec ... self.distpow4rec
        #
        # no formal parameters needed, the object instance knows them
        print("pyNBED: Preparing reciprocal space sampling grid")
        distx = np.arange(self.dim[2],dtype=np.int64)-np.ceil(self.dim[2]/2)
        disty = np.arange(self.dim[3],dtype=np.int64)-np.ceil(self.dim[3]/2)
        # dist-x = distance from centre at dimx/2 
        # dist-y = distance from centre at dimy/2 
        REPdistx = numpy.matlib.repmat(distx, self.dim[3],1)
        REPdisty = numpy.matlib.repmat(disty, self.dim[2],1)
        self.qsamplinggrid2 = (REPdistx*REPdistx).transpose() + (REPdisty*REPdisty)
        # distpow2 is the square distance from the centre
        self.qsamplinggrid=np.sqrt(self.qsamplinggrid2)
        return

    def LoadFile (self, fname, type='PantaRhei'):
        """ Load a 4D STEM dataset  

        Parameters:
        fname:   complete filename including path and suffix
        type:    'PantaRhei' or 'EMPAD' [default: PantaRhei]
 
        Returns:

        void

        """
        self.fname=fname
        self.type=type
        if self.type == 'PantaRhei':
            self.type=type
            print("pyNBED: Loading object descriptor: "+self.fname)
            fh=np.load(self.fname, allow_pickle=True)
            self.metadata=fh['meta_data']
            # load data array and report dimensions
            print("pyNBED: Loading data ...")
            self.data=fh['data']
            print("        finished")
            self.dim=self.data.shape
            print("pyNBED: array dimensions:", self.dim) 
            self.PrepareSamplingGrid()
        if self.type == 'EMPAD':
            self.type=type
            print("pyNBED: Loading EMPAD data from "+self.fname)
            self.data=read_empad(self.fname)
            self.metadata=None
            self.dim=self.data.shape
            print("pyNBED: array dimensions:", self.dim) 
            self.PrepareSamplingGrid()
        return

    def ShowFrame(self, i=None,j=None, Log=None, sd=3):
        """ Display a diffraction frame  

        Parameters:
        i,j:    spatial pixel indices [default: center of frame]
        Log:    display log scale image
        sd:     half the auto contrast cutoff range in factors of the standard deviation 

        Returns:

        None

        """
        #show frame in 4D array
        if i is None:
            i=np.rint(self.dim[0]/2).astype(np.int16)
        if j is None:
            j=np.rint(self.dim[1]/2).astype(np.int16)
        if not ((0 <= i < self.dim[0]) and (0 <= j < self.dim[1])):
            print("Range error. Check frame indices.")
            return 
        img=np.copy(self.data[i,j,::])
        print("frame coordinates:", i, j)
        print("frame dimensions: ",img.shape)       
        plt.figure()
        if Log:
            img[img < 0]=0.
            img=np.log(img+1.)
            mean=np.mean(img)
            sdev=np.std(img)
            plt.imshow(img,vmin=0, vmax=mean+sd*sdev) 
        else:
            mean=np.mean(img)
            sdev=np.std(img)
            plt.imshow(img,vmin=mean-sd*sdev, vmax=mean+sd*sdev) 
        plt.show()  # display it
        return np.copy(self.data[i,j,::])


    def GetFrame(self, i=None,j=None, clip=False, Log=None, sd=3):
        """ Return a diffraction frame  

        Parameters:
        i,j:    spatial pixel indices [default: center of frame]
        clip:   clip contrast if True
        Log:    return log scale image
        sd:     half the auto contrast cutoff range in factors of the standard deviation 

        Returns:

        2D array

        """
        #show frame in 4D array
        if i is None:
            i=np.rint(self.dim[0]/2).astype(np.int16)
        if j is None:
            j=np.rint(self.dim[1]/2).astype(np.int16)
        if not ((0 <= i < self.dim[0]) and (0 <= j < self.dim[1])):
            print("Range error. Check frame indices.")
            return None
        img=np.copy(self.data[i,j,::])
        print("frame coordinates:", i, j)
        print("frame dimensions: ",img.shape)       
        if Log:
            img[img < 0]=0.
            img=np.log(img+1., where=img>=1)
            if clip:
                mean=np.mean(img)
                sdev=np.std(img)
                img=bytscl(img,vmin=0, vmax=mean+sd*sdev) 
        else:
            if clip:
                mean=np.mean(img)
                sdev=np.std(img)
                img=bytscl(img,vmin=mean-sd*sdev, vmax=mean+sd*sdev) 
        return img        
        
    # find maximum position in all slices
    def DiffractionShift(self, searchradius=None, searchoffset=None, reject_sd=3, Stats=True, Plot=True):
        """ Find the strongest peak position in all frames  

        Parameters:
        searchradius:    optional mask radius to restrict the search range around the center [default: None (means search in the full frame)]
        searchoffset:    optional displacement of the search mask from the center in pixels [default: None]
                         offset=[20,100] shifts the search mask 20 pixels horizontally and 100 pixels vertically
        reject_sd:       exclude outliers for robust statistics, outliers are those shifts beyond reject_sd times 
                         the standard deviation of shifts [default: 3]
        Stats:           report displacement statistics [defualt: True]
        Plot:            Show a pretty plot of the displacements horizontally and vertically [default: True]

        Returns:

        (dx,dy) tuple of 2D arrays, dtype np.float64

        """
        # reshape array for analysis
        newa=self.data.reshape(self.dim[0]*self.dim[1],self.dim[2],self.dim[3]) 
        adim=newa.shape
        dx=np.zeros(adim[0],dtype=np.int16)
        dy=np.zeros(adim[0],dtype=np.int16)
        cx=np.floor_divide(adim[1],2)
        cy=np.floor_divide(adim[2],2)
        print("locating strongest pixel in all frames ...")
        if not(searchradius is None):
            mask=(self.qsamplinggrid2 < searchradius*searchradius)
            if not(searchoffset is None):
                mask=np.roll(np.roll(mask,searchoffset[0],axis=1),searchoffset[1],axis=0)
            mask= (mask == False)
        for ind in range(adim[0]):
            img=np.copy(newa[ind,::])
            if not(searchradius is None):       
                img[mask]=0.
            loc=np.unravel_index(np.argmax(img, axis=None), img.shape)
            dx[ind]=loc[0]-cx
            dy[ind]=loc[1]-cy
        print("finished")
        self.data.reshape(self.dim[0],self.dim[1],self.dim[2],self.dim[3]) 
        if not(reject_sd is None):
            # linear sorting of dx, dy
            dx=dx.reshape(self.dim[0]*self.dim[1])
            dy=dy.reshape(self.dim[0]*self.dim[1])
            # reject outliers
            dxfilt=dx[abs(dx - np.mean(dx)) < reject_sd * np.std(dx)] 
            dyfilt=dy[abs(dy - np.mean(dy)) < reject_sd * np.std(dy)]
            # plot a histogram of the locations without outliers
            if Plot:
                plt.hist([dxfilt,dyfilt],bins=100)
                plt.show()
            # clean up dx, dy
            cleanx=np.where(abs(dx - np.mean(dxfilt)) <= reject_sd * np.std(dxfilt))
            cleany=np.where(abs(dy[cleanx] - np.mean(dy[cleanx])) <= reject_sd * np.std(dy[cleanx]))
            clean=cleanx[0][cleany[0]]
            clean=np.unravel_index(clean, (self.dim[0],self.dim[1]))
            # back to two-d sorting
            dx=dx.reshape(self.dim[0],self.dim[1])
            dy=dy.reshape(self.dim[0],self.dim[1])
            if Stats:
                # Print statistics without outliers
                print("Displacement statistics along x, filtered for outliers (all indices):")
                print("Mean = ",np.mean(dx[clean[0],clean[1]]), "(",np.mean(dx),")")
                print("Max  = ",np.max(dx[clean[0],clean[1]]), "(",np.max(dx),")")
                print("Min  = ",np.min(dx[clean[0],clean[1]]), "(",np.min(dx),")")
                print("Displacement statistics along y, filtered for outliers (all indices):")
                print("Mean = ",np.mean(dy[clean[0],clean[1]]), "(",np.mean(dy),")")
                print("Max  = ",np.max(dy[clean[0],clean[1]]), "(",np.max(dy),")")
                print("Min  = ",np.min(dy[clean[0],clean[1]]), "(",np.min(dy),")")
        else:
            # Print statistics without outliers
            if Stats:
                print("Displacement statistics along x, filtered for outliers (all indices):")
                print("Mean = ",np.mean(dx),")")
                print("Max  = ",np.max(dx),")")
                print("Min  = ",np.min(dx),")")
                print("Displacement statistics along y, filtered for outliers (all indices):")
                print("Mean = ",np.mean(dy),")")
                print("Max  = ",np.max(dy),")")
                print("Min  = ",np.min(dy),")")
        if Plot:
            # Plot a map of the shifts in x and y
            plt.figure()
            #subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(1,2) 
            # use the created array to output your multiple images. In this case I have stacked 4 images vertically
            axarr[0].title.set_text("dx")
            axarr[1].title.set_text("dy")
            axarr[0].imshow(dx,vmin=np.min(dx), vmax=np.max(dx))
            axarr[1].imshow(dy ,vmin=np.min(dy), vmax=np.max(dy))
        return (dx,dy)

    def CompensateShift(self, dx, dy, optimize=True):
        """ Compensate diffraction shifts to center the diffraction frames 

        Parameters:
        dx, dy:    2D arrays with the shift values horizontally and vertically
                   pre-calculate these arrays with the method DiffractionShift()
        optimize:  optional regression with a polynomial of second order in order to smoothen the shifts and retain 
                   local variations e.g. for a center-of-mass analysis [default: True]

        Returns:

        void (the raw data frames are updated)

        """
        if optimize: 
            dxfit=ParabolaFit2D(dx)
            plt.figure()
            #subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(1,3) 
            # use the created array to output your multiple images
            axarr[0].title.set_text("dx")
            axarr[1].title.set_text("fitted dx")
            axarr[2].title.set_text("residuum")
            axarr[0].imshow(dx ,vmin=np.min(dx), vmax=np.max(dx))
            axarr[1].imshow(dxfit ,vmin=np.min(dx), vmax=np.max(dx))
            axarr[2].imshow(dx-dxfit ,vmin=np.min(dx), vmax=np.max(dx))
            plt.show()
            # Some residuum  statistics
            print("Residuum statistics dx-dxfit")
            print("Mean deviation:    ", np.mean(dx-dxfit))
            print("Max abs deviation: ",np.max(np.abs(dx-dxfit)))
            print("Mean abs deviation:", np.mean(np.abs(dx-dxfit)))
            dyfit=ParabolaFit2D(dy)
            plt.figure()
            #subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(1,3) 
            # use the created array to output your multiple images
            axarr[0].title.set_text("dy")
            axarr[1].title.set_text("fitted dy")
            axarr[2].title.set_text("residuum")
            axarr[0].imshow(dy ,vmin=np.min(dy), vmax=np.max(dy))
            axarr[1].imshow(dyfit ,vmin=np.min(dy), vmax=np.max(dy))
            axarr[2].imshow(dy-dyfit ,vmin=np.min(dy), vmax=np.max(dy))
            plt.show()
            # Some residuum  statistics
            print("Residuum statistics dy-dyfit")
            print("Mean deviation:    ", np.mean(dy-dyfit))
            print("Max abs deviation: ",np.max(np.abs(dy-dyfit)))
            print("Mean abs deviation:", np.mean(np.abs(dy-dyfit)))
            dx=dxfit
            dy=dyfit
        # compensate shift
        compdx=-np.rint(dxfit.reshape(self.dim[0]*self.dim[1])).astype(np.int16)
        compdy=-np.rint(dyfit.reshape(self.dim[0]*self.dim[1])).astype(np.int16)
        self.data=self.data.reshape(self.dim[0]*self.dim[1],self.dim[2],self.dim[3])
        for ind in range(self.data.shape[0]):
            self.data[ind,::]=np.roll(self.data[ind,::],(compdy[ind],compdx[ind]),axis=(1,0))
        print("finished")
        self.data=self.data.reshape(self.dim[0],self.dim[1],self.dim[2],self.dim[3])
        return

    def LinearIndexArray(self, rows, cols, is_roi=True):
        """ Convert an array of columns and row vectors into a linear set of indices 

        Parameters:
        rows:  1D-array of row indices
        cols:  1D-array of column indices
        is_roi: if True then the list of indices will have n(cols) x m(rows) indices (each column value will have a full row)
                if False then cols[i] will be paired with rows[i] for each i, the length of rows and cols have to be the same 

        Returns:

        1-D array res, each element represents a coordinate with 
        res=res // self.dim[0] + res % self.dim[0]
        the x-coordinate is res // self.dim[1]  
        the y-coordinate is res % self.dim[1]
        the row is equivalent with the x coordinate, i.e. the vertical coordinate in Python 

        """
        l=list()
        if is_roi:
            for i in rows:
                for j in cols:
                    l.append(i*self.dim[1]+j) 
        else: 
            maxi=np.min([cols.shape[0],rows.shape[0]])
            # should match
            for i in np.arange(maxi):
                l.append(rows[i]*self.dim[1]+cols[i])                 
        return np.array(l)

    def PreparePeakDetectionPars(self, shortcut=None):
        """ Returns a dictionary with preset parameters for peak detection

        Parameters:
        shortcut:  optional parameter to retrieve presets for a specific detector,
                   possible values are "ELA" or "ARINA" or "EMPAD"
                
        Returns: 

        parameter dictionary

        """
        if shortcut is None:
            par={
                'feature_size':7, 
                'feature_minmass':2.,
                'feature_separation':8,
                'feature_percentile':65.,
                'feature_threshold':1.,
                'noise_size':1.,
                'smooth_size': None,
                'frame_cmin':1.,
                'frame_cmax':10.,
                'frame_cutoff': 0.,
                'conv2D':True
            }
        if (shortcut == 'ELA'):
            par={
                'feature_size':7, 
                'feature_minmass':2.,
                'feature_separation':8,
                'feature_percentile':65.,
                'feature_threshold':1.,
                'noise_size':1.,
                'smooth_size': None,
                'frame_cmin':1.,
                'frame_cmax':10.,
                'frame_cutoff': 0.,
                'conv2D':True
            } 
        if (shortcut == 'ARINA'):
            par={
                'feature_size':7, 
                'feature_minmass':2.,
                'feature_separation':8,
                'feature_percentile':65.,
                'feature_threshold':1.,
                'noise_size':1.,
                'smooth_size': None,
                'frame_cmin':1.,
                'frame_cmax':10.,
                'frame_cutoff': 0.,
                'conv2D':True,
            } 
        if (shortcut == 'EMPAD'):
            par={
                'feature_size':7, 
                'feature_minmass':1.,
                'feature_separation':8,
                'feature_percentile':65.,
                'feature_threshold':10.,
                'noise_size':1.4,
                'smooth_size': None,
                'frame_cmin':5.,
                'frame_cmax':16.,
                'frame_cutoff': 10.,
                'conv2D':False
            }
        return par
      

    def AnimateFrames(self, idxarray, params=None):
        
        if params is None:
            params=PreparePeakDetectionPars()
        feat_sz=params['feature_size']
        feat_minmass=params['feature_minmass']
        feat_sep=params['feature_separation']
        feat_perc=params['feature_percentile']
        noise_sz=params['noise_size'] # keep this at 1.0, filtering means somehow that the peaks are displaced
        thresh=params['feature_threshold']
        smooth_sz=params['smooth_size']
        cmin=params['frame_cmin']
        cmax=params['frame_cmax']
        conv2D=params['conv2D']
        conv_kernel=1./9*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        # overwrite with self.peakdetpar
        # return table
        img=self.data[idxarray[0]//self.dim[0],idxarray[0] % self.dim[0],:,:]
        if params["frame_cutoff"] > 0.:
            img[img < params["frame_cutoff"]]=0.
        else:
            img[(img < 0.)]=0       
        plt.figure()
        plt.imshow(bytscl(img,vmin=cmin,vmax=cmax),origin="lower") 
        plt.show()
        txt=plt.figtext(0.55, 0.8, "i,j = ",fontsize = 12,color ="black") 
        #
        halfframedim=[self.dim[2]/2,self.dim[3]/2]
        #
        for ind in idxarray:  
            i=ind // self.dim[0]
            j=ind % self.dim[0]
            if conv2D:
                filtim=convolve2D(self.data[i,j,::], conv_kernel, padding=2)
            else:
                filtim=self.data[i,j,::]   
            # filtim=img
            dispim=bytscl(np.log(filtim+1.),vmin=params["frame_cmin"],vmax=params["frame_cmax"])
            txt=plt.figtext(0.55, 0.8, "i,j = "+str(i)+","+str(j)+";"+str(len(f)),fontsize = 12,color ="black") 
            #display.clear_output(wait=True)
            #display.display(plt.gcf(),clear=True)
            display.clear_output(wait=True)
            time.sleep(0.001)
        return

    def PeakDetection(self, idxarray, params=None, animate=True, raw=False):        
 
        """ Auto-detect diffraction peaks in a set of frames      
        
        Uses trackpy.py, an implementation of the Crocker-Grier centroid-finding algorithm.
        (Crocker, J.C., Grier, D.G. http://dx.doi.org/10.1006/jcis.1996.0217)

        Parameters:
        idxarray:   list of spatial pixel indices in linear order
        raw:        boolean, if true then the filtered raw data is used
                    if false the the fit is on the log scale
        
        params:     dict with peak search parameters
                    
                    keys:
                    'feature_size': feature size in pixels [default: 7] 
                        The typical diameter of the peaks. The number must be an odd integer. 
                        When in doubt, round up.
                    'feature_minmass': The minimum integrated brightness [default: 2.]
                        This is a crucial parameter for eliminating
                        spurious features. Recommended minimum values are 100 for integer images 
                        and 1 for float images. Defaults to 0 (no filtering).
                    'feature_separation': Minimum separation between features.
                        Default is diameter + 1.
                    'feature_percentile': 
                        Features must have a peak brighter than pixels in this percentile. 
                        This helps eliminate spurious peaks. [default: 65.] 
                    'noise_size': Width of Gaussian blurring kernel in pixels [default: 1.]     
                        HINT: keep this at 1.0, noise_size filtering means somehow that the peaks 
                        are displaced
                    'feature_threshold': Clip bandpass result below this value. Thresholding is done
                        on the already background-subtracted image.
                        [default: 1 for integer images and 1/255 for float images]
                    'smooth_size': The size of the sides of the square kernel used in boxcar (rolling
                        average) smoothing, in pixels [default: None]
                    'frame_cmin': frame contrast clipping minimum cut-off after low-pass filtering 
                        or bandpass filtering [default: 1]
                    'frame_cmax': frame contrast clipping maximum cut-off after low-pass filtering 
                        or bandpass filtering [default: 10.]
                    'conv2D': boolean, perform 2D low pass convolution filtering before processing 
                        [default: true]
                    

        Returns:

        peaklist, list of hashes, a list entry has the following hash keys and values:
        {'i': (spatial) frame index horizontally,
         'j': (spatial) frame index horizontally,
         'x': array of q_x-coordinates,
         'y': array of q_y-coordinates,
         'q': array of magnitude of q,
         'mass': array of peak masses,
         'size': array of peak sizes,
         'raw_mass': array of peak raw masses}

        """
        if params is None:
            params=PreparePeakDetectionPars()
        feat_sz=params['feature_size']
        feat_minmass=params['feature_minmass']
        feat_sep=params['feature_separation']
        feat_perc=params['feature_percentile']
        noise_sz=params['noise_size'] # keep this at 1.0, filtering means somehow that the peaks are displaced
        thresh=params['feature_threshold']
        smooth_sz=params['smooth_size']
        cmin=params['frame_cmin']
        cmax=params['frame_cmax']
        conv2D=params['conv2D']
        conv_kernel=1./9*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        # overwrite with self.peakdetpar
        # return table
        img=self.data[idxarray[0]//self.dim[1],idxarray[0] % self.dim[1],:,:]
        if params["frame_cutoff"] > 0.:
            img[img < params["frame_cutoff"]]=0.
        else:
            img[(img < 0.)]=0
        if animate:
            plt.figure()
            plt.imshow(bytscl(img,vmin=cmin,vmax=cmax),origin="lower") 
            plt.show()
            scat=plt.scatter([], [])
            txt=plt.figtext(0.55, 0.8, "i,j;#peaks = ",fontsize = 12,color ="black") 
        #
        framepeaklist=list()
        halfframedim=[self.dim[2]/2,self.dim[3]/2]
        # TQDM AND ANIMATED VIEW DON'T WORK WELL TOGETHER
        if animate:
            for ind in idxarray:  
                i=ind // self.dim[1]
                j=ind % self.dim[1]
                if conv2D:
                    filtim=convolve2D(self.data[i,j,::], conv_kernel, padding=2)
                else:
                    filtim=self.data[i,j,::]   
                # filtim=img
                scale=cmax/np.max(img) 
                dispim=bytscl(np.log(filtim+1.),vmin=params["frame_cmin"], vmax=params["frame_cmax"])
                plt.imshow(dispim,cmap='gray_r',origin="lower")
                # detect on linear scale
                #f=detect_peaks(bytscl(filtim,vmin=cmin,vmax=cmax),feat_sz,noise_sz=noise_sz,thresh=thresh,feat_minmass=feat_minmass,feat_sep=feat_sep,feat_perctile=feat_perc,inv=False,smoothing_size=smooth_sz)
                # detect on log scale
                if raw:
                    f=tp.locate(filtim, feat_sz, noise_size=noise_sz, threshold=thresh,minmass=feat_minmass,percentile=feat_perc,separation=feat_sep,invert=False, smoothing_size=smooth_sz,engine="python")
                else:
                    f=tp.locate(dispim, feat_sz, noise_size=noise_sz, threshold=thresh,minmass=feat_minmass,percentile=feat_perc,separation=feat_sep,invert=False, smoothing_size=smooth_sz,engine="python")
                if len(f) > 0:
                    #f=detect_peaks(dispim,feat_sz,noise_sz=noise_sz,thresh=thresh,feat_minmass=feat_minmass,feat_sep=feat_sep,feat_perctile=feat_perc,inv=False,smoothing_size=smooth_sz)
                    #calculate distances from center
                    q=np.sqrt((f['x'].values-halfframedim[0])**2+(f['y'].values-halfframedim[1])**2)
                    # add column to the pandas frame f
                    f.insert(1, "q", q, True)
                    # sort pandas frame according to distance to the center
                    f=f.sort_values(by=['q'], ascending=True)
                    # add frame results to list
                    framepeaklist.append({'i': i,'j': j,'x':f['x'].values-halfframedim[0],'y':f['y'].values-halfframedim[1],'q':f['q'].values,'mass':f['mass'].values,'size':f['size'].values,'raw_mass':f['raw_mass'].values})
                    colors = np.random.rand(len(f))
                    scat.remove()
                    scat=plt.scatter(f.x, f.y, s=f.mass/2, c=colors, alpha=0.5)
                    txt.remove()
                    txt=plt.figtext(0.55, 0.8, "i,j;#peaks = "+str(i)+","+str(j)+";"+str(len(f)),fontsize = 12,color ="black") 
                    #display.clear_output(wait=True)
                    display.display(plt.gcf(),clear=True)
                    display.clear_output(wait=True)
                    time.sleep(0.001)
        else:
            for ind in tqdm(idxarray,desc='Processing Frames '):  
                i=ind // self.dim[1]
                j=ind % self.dim[1]
                if conv2D:
                    filtim=convolve2D(self.data[i,j,::], conv_kernel, padding=2)
                else:
                    filtim=self.data[i,j,::]   
                # filtim=img
                scale=cmax/np.max(img) 
                dispim=bytscl(np.log(filtim+1.),vmin=params["frame_cmin"], vmax=params["frame_cmax"])
                # detect on linear scale
                #f=detect_peaks(bytscl(filtim,vmin=cmin,vmax=cmax),feat_sz,noise_sz=noise_sz,thresh=thresh,feat_minmass=feat_minmass,feat_sep=feat_sep,feat_perctile=feat_perc,inv=False,smoothing_size=smooth_sz)
                # detect on log scale
                if raw:
                    f=tp.locate(filtim, feat_sz, noise_size=noise_sz, threshold=thresh,minmass=feat_minmass,percentile=feat_perc,separation=feat_sep,invert=False, smoothing_size=smooth_sz,engine="python")
                else:
                    f=tp.locate(dispim, feat_sz, noise_size=noise_sz, threshold=thresh,minmass=feat_minmass,percentile=feat_perc,separation=feat_sep,invert=False, smoothing_size=smooth_sz,engine="python")
                if len(f) > 0:
                    #f=detect_peaks(dispim,feat_sz,noise_sz=noise_sz,thresh=thresh,feat_minmass=feat_minmass,feat_sep=feat_sep,feat_perctile=feat_perc,inv=False,smoothing_size=smooth_sz)
                    #calculate distances from center
                    q=np.sqrt((f['x'].values-halfframedim[0])**2+(f['y'].values-halfframedim[1])**2)
                    # add column to the pandas frame f
                    f.insert(1, "q", q, True)
                    # sort pandas frame according to distance to the center
                    f=f.sort_values(by=['q'], ascending=True)
                    # add frame results to list
                    framepeaklist.append({'i': i,'j': j,'x':f['x'].values-halfframedim[0],'y':f['y'].values-halfframedim[1],'q':f['q'].values,'mass':f['mass'].values,'size':f['size'].values,'raw_mass':f['raw_mass'].values})
        return framepeaklist


    def DebyeScherrerPlot(self, framepeaklist,refine=False):
        """ Create a scatter plot of peak locations from a framepeaklist 
            generated by PeakDetection

        Parameters:
        framepeaklist:    list of frame peak data, output of PeakDetection()
        refine:           subtract the exact locaton of the central beam [default: False]

        Returns:

        void

        """
        plt.figure()
        scat=plt.scatter([], [])
        for f in framepeaklist:
            x0=0
            y0=0
            if refine:
                # subtract the exact location of the central beam
                x0=f['x'][0]
                y0=f['y'][0]
            scat=plt.scatter(f['x']-x0, f['y']-y0, s=1, alpha=0.5)
        plt.show
        return

    def PeakDistanceHistogram(self,farmepeaklist, refine=True):
        """ Create a histogram plot of peak distances from a framepeaklist 
            generated by PeakDetection

        Parameters:
        framepeaklist:    list of frame peak data, output of PeakDetection()
        refine:           subtract the exact locaton of the central beam [default: False]

        Returns:

        (counts, bins)

        """
        qarr=np.empty([],dtype=np.float64)
        refine=False
        for f in framepeaklist:
            if refine:
                # subtract the exact location of the central beam
                x=f['x']
                x=x-x[0]
                y=f['y']
                y=y-y[0]
                qarr=np.append(qarr,sqrt(x*x+y*y))
            else:
                qarr=np.append(qarr,f['q'])
        # assume 1/3 pixel size precision    
        counts, bins = np.histogram(qarr, bins=self.dim[2]*3)
        plt.stairs(counts, bins)
        return (counts, bins)

    def VirtualApertureImage(self, iradius=0, radius=10, offset=None, invert=False):
        """ Create a virtual detector image with a disk-like aperture

        Parameters:
        radius:    radius of the aperture in pixels [default:10]
        iradius:   inner radius if applicable [default: 0]
        offset:    optional displacement from the center in pixels [default: None]
                   offset=[20,100] shifts the aperture mask 20 pixels horizontally and 100 pixels vertically
        invert:    invert the aperture mask, e.g. to create a dark-field image instead of a bright-field image

        Note: 
        
        the mask will span the range [iradius,oradius[
        
        Examples: 
        
        BF detector - iradius=0, oradius=radius of the bright field mask, invert=False
        DF detector - iradius=0, oradius=inner cut off of the dark field mask, invert=True
        Annular DF detector - iradius=inner cut off, oradius=outer cut off of the dark field mask, invert=False
 
        Returns:

        2D array, dtype np.float64

        """
        # create a virtual   image
        img=np.zeros((self.dim[0],self.dim[1]),dtype=np.float64)
        mask=((self.qsamplinggrid2 >= iradius*iradius) & (self.qsamplinggrid2 < radius*radius))
        if not(offset) is None:
            mask=np.roll(np.roll(mask,offset[0],axis=1),offset[1],axis=0)
            # shift center            
        if invert:
            mask = (mask == False)
        #for i in np.arange(self.dim[0]):
        #    for j in np.arange(self.dim[1]):
        #        img[i,j]=np.sum(self.data[i,j,mask])
        img=np.sum(self.data[:,:,mask],axis=2)
        return img    
    

    def StackExport(self, format='raw'):
        """ Export data set

        Parameters:
        format:  'raw' for raw binary

        Returns:

        void
        """        
        dirname = os.path.dirname(self.fname)
        basename_without_ext = os.path.splitext(os.path.basename(self.fname))[0]
        basename=dirname+os.sep+basename_without_ext
        if (export == "raw"):
            # save as 3D binary data
            # split self.fname into path and suffix            
            fnout=basename+'-'+str(self.dim[1])+'x'+str(self.dim[0])+'_'+str(self.dim[2])+'x'+str(self.dim[3])+'_'+str(self.dim[1]*self.dim[0])+'frames'+'_uint32.raw'
            print("saving raw binary data to ",fnout)
            (self.data.reshape(self.dim[0]*self.dim[1],self.dim[2],self.dim[3])).astype('uint32').tofile(fnout)
            print("finished")
    
