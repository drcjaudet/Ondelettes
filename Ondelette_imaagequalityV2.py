# coding: utf-8

import unittest
from slicer.ScriptedLoadableModule import *
import logging
from __main__ import vtk, qt, ctk, slicer
from math import *
import numpy as np
from vtk.util import numpy_support
import SimpleITK as sitk
import sitkUtils as su
import time
import codecs
import datetime
import vtkSegmentationCorePython as vtkSegmentationCore
import dicom
import sys, time, os

import pywt # https://github.com/PyWavelets/pywt/blob/master/pywt/_multilevel.py

NomDeLImage='template'

def Ondelette_raconte(NomDeLImage):
    timeRMR1 = time.time()
    image=su.PullFromSlicer(NomDeLImage)
    NumpyImage=sitk.GetArrayFromImage(image)
    max_lev = 2       # how many levels of decomposition to draw
    c = pywt.wavedecn(NumpyImage, 'db2', mode='zero', level=max_lev) #voir https://pywavelets.readthedocs.io/en/latest/ref/nd-dwt-and-idwt.html#pywt.wavedecn  
    #coeffs[-2] = {k: np.zeros_like(v) for k, v in coeffs[-2].items()}
    #matrice_ondelette=pywt.waverecn(c, 'db2') mode periodic ou zero
    #image_ondelette=sitk.GetImageFromArray(matrice_ondelette)
    #su.PushToSlicer(image_ondelette,'image_ondelette')
    c_arr,c_slices= pywt.coeffs_to_array(c, padding=0, axes=None)
    ddd=c_arr[c_slices[2]['ddd']] #ddd=sitk.GetImageFromArray(c_arr[c_slices[2]['ddd']]) #details
    aaa=c_arr[c_slices[0]] #aaa=sitk.GetImageFromArray(c_arr[c_slices[0]]) #average
    IndiceQualite=SpatialFrequencyOptim2(ddd)/SpatialFrequencyOptim2(aaa)
    print IndiceQualite
    timeRMR2 = time.time()
    TimeForrunFunctionRMR2 = timeRMR2 - timeRMR1
    print(u"La fonction de traitement s'est executée en " + str(TimeForrunFunctionRMR2) +" secondes")



def Ondelette_raconte(NomDeLImage):
    image=su.PullFromSlicer(NomDeLImage)
    NumpyImage=sitk.GetArrayFromImage(image)
    max_lev = 2
    c = pywt.wavedec2(NumpyImage, 'db2', mode='zero',axes=(-2,-1), level=max_lev)
    c_arr,c_slices= pywt.coeffs_to_array(c, padding=0, axes=(-2,-1)
    aa=c_arr[c_slices[0]]   
    image_ondelette=sitk.GetImageFromArray(aa)
    image_ondelette.SetSpacing(image.GetSpacing())
    image_ondelette.SetDirection(image.GetDirection())
    image_ondelette.SetOrigin(image.GetOrigin())
    su.PushToSlicer(image_ondelette,'image_aa')




def SpatialFrequency(image):
    SizeMatrix=image.GetSize()
    Square_diff_x=0
    Square_diff_y=0
    Square_diff_z=0
    Nvoxel=0
    for x in range(SizeMatrix[0]-1):
        for y in range(SizeMatrix[1]-1):
            for z in range(SizeMatrix[2]-1):
                    Square_diff_x=Square_diff_x+(image.GetPixel(x+1,y,z)-image.GetPixel(x,y,z))**2
                    Square_diff_y=Square_diff_y+(image.GetPixel(x,y+1,z)-image.GetPixel(x,y,z))**2
                    Square_diff_z=Square_diff_z+(image.GetPixel(x,y,z+1)-image.GetPixel(x,y,z))**2
                    Nvoxel=Nvoxel+1
    SF=(Square_diff_x+Square_diff_y+Square_diff_z)**0.5 #for testing
    #SF=(Square_diff_x/(Nvoxel)+Square_diff_y/(Nvoxel)+Square_diff_z/(Nvoxel))**0.5
    return SF


def reechantillonage_translateOnly(image_ref, tranformation,MinimumImage):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(MinimumImage)
    resampler.SetTransform(tranformation)
    ImageRecaler = resampler.Execute(image_ref)
    return ImageRecaler 

def SpatialFrequencyOptim(image):
    dimension = 3        
    offset =(1,0,0) # offset can be any vector-like data  
    translation_dx = sitk.TranslationTransform(dimension, offset)
    image_dx=reechantillonage_translateOnly(image,translation_dx,0)      
    offset =(0,1,0) # offset can be any vector-like data  
    translation_dy = sitk.TranslationTransform(dimension, offset)
    image_dy=reechantillonage_translateOnly(image,translation_dy,0)
    offset =(0,0,1) # offset can be any vector-like data  
    translation_dz = sitk.TranslationTransform(dimension, offset)
    image_dz=reechantillonage_translateOnly(image,translation_dz,0)
    imageSF=(image-image_dx)**2+(image-image_dy)**2+(image-image_dz)**2
    imageSF=sitk.SumProjection(imageSF,2)
    imageSF=sitk.SumProjection(imageSF,1)
    imageSF=sitk.SumProjection(imageSF,0)
    #SF=imageSF.GetPixel(0,0,0)**0.5 #for testing
    SF=(imageSF.GetPixel(0,0,0)/(image.GetSize()[0]*image.GetSize()[1]*image.GetSize()[2]))**0.5
    return SF

def SpatialFrequencyOptim2(matrix):
    sq_diff = 0.0
    size=matrix.shape
    dim=len(size)  
    for i in range(dim): #iterate over all image dimensions
        slc1 = [slice(None)]*dim
        slc1[i] = slice(0,size[i]-1)
        slc2 = [slice(None)]*dim
        slc2[i] = slice(1,size[i])
        sq_diff+= np.sum((matrix[tuple(slc2)]- matrix[tuple(slc1)])**2)
    return sq_diff/np.prod(size)

        
dim=3

img = sitk.GaussianSource(outputPixelType=sitk.sitkUInt8, size=[128]*dim, sigma=[20]*dim, mean=[60]*dim)

sitk.Show(img)



res = SpatialFrequencyOptim(img)
    return SF



NomDeLImage='FOUMA_1m'
Nlevel=2

def wavelet_denoising(NomDeLImage, Nlevel):
    image=su.PullFromSlicer(NomDeLImage)
    NumpyImage=sitk.GetArrayFromImage(image)
    max_lev = 6       # how many levels of decomposition to draw
    coeffs = pywt.wavedecn(NumpyImage, 'db2', mode='zero', level=max_lev) #voir https://pywavelets.readthedocs.io/en/latest/ref/nd-dwt-and-idwt.html#pywt.wavedecn
    for i in range(Nlevel-max_lev):
        coeffs[(max_lev-i)] = {k: np.zeros_like(v) for k, v in coeffs[(max_lev-i)].items()} #remove highest frequency
        coeffs[-(max_lev-i)] = {k: np.zeros_like(v) for k, v in coeffs[-(max_lev-i)].items()} #remove highest frequency
    matrice_ondelette=pywt.waverecn(coeffs, 'db2') #mode periodic ou zero
    image_ondelette=sitk.GetImageFromArray(matrice_ondelette)
    image_ondelette.SetSpacing(image.GetSpacing())
    image_ondelette.SetDirection(image.GetDirection())
    image_ondelette.SetOrigin(image.GetOrigin())
    su.PushToSlicer(image_ondelette,'image_DenoisWave_level0-'+str(Nlevel))
  

###########autre test

def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff=1000, sample_rate=sample_rate):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """   
    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist  
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    # scipy version 1.2.0
    #sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')
    
    # scipy version 1.1.0
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)
    return filtered_sig

def denoise_signal( x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """   
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )   
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest( coeff[-level] )
    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode='per' )



def wavelet_denoising2(NomDeLImage, Nlevel):
    image=su.PullFromSlicer(NomDeLImage)
    NumpyImage=sitk.GetArrayFromImage(image)
    max_lev = 6       # how many levels of decomposition to draw
    coeffs = pywt.wavedecn(NumpyImage, 'db2', mode='zero', level=max_lev) #voir https://pywavelets.readthedocs.io/en/latest/ref/nd-dwt-and-idwt.html#pywt.wavedecn
    for levelR in range (max_lev-Nlevel):
        sigma = (1/0.6745) * maddest( coeffs[max_lev-levelR] )
        uthresh = sigma * np.sqrt( 2*np.log( len( NumpyImage ) ) )
        coeffs[(max_lev-levelR)] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeffs[(max_lev-levelR)] )
    matrice_ondelette=pywt.waverecn(coeffs, 'db2', mode='per') #mode periodic ou zero
    image_ondelette=sitk.GetImageFromArray(matrice_ondelette)
    image_ondelette.SetSpacing(image.GetSpacing())
    image_ondelette.SetDirection(image.GetDirection())
    image_ondelette.SetOrigin(image.GetOrigin())
    su.PushToSlicer(image_ondelette,'image_DenoisWave_level0-'+str(Nlevel))



from pip._internal import main as pip_main
pip_modules = ['scipy', 'sklearn', 'PyWavelets']
for module_ in pip_modules:
    try:
        module_obj = __import__(module_)
    except ImportError:
        logging.info("{0} was not found.\n Attempting to install {0} . . ."
                     .format(module_))
        pip_main(['install', module_])



pip_main(['install','scikit-image'])


from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio


Nom_image="FOUMA_1m"

def denoising_BayesShrinkAndVIsuShrink(Nom_image):
    image=su.PullFromSlicer(NomDeLImage)
    NumpyImage=sitk.GetArrayFromImage(image)
    # Estimate the average noise standard deviation across color channels.
    sigma_est = estimate_sigma(NumpyImage, multichannel=True, average_sigmas=True)
    # Due to clipping in random_noise, the estimate will be a bit smaller than the
    # specified sigma.
    print(f"Estimated Gaussian noise standard deviation = {sigma_est}")
    im_bayes = denoise_wavelet(NumpyImage, multichannel=True, convert2ycbcr=True, method='BayesShrink', mode='soft',rescale_sigma=True)
    im_visushrink = denoise_wavelet(NumpyImage, multichannel=True, convert2ycbcr=True, method='VisuShrink', mode='soft',sigma=sigma_est, rescale_sigma=True)
    su.PushToSlicer(im_bayes,'image_DenoisWave_level0-'+str(Nlevel))
    su.PushToSlicer(im_visushrink,'image_DenoisWave_level0-'+str(Nlevel))
    # VisuShrink is designed to eliminate noise with high probability, but this
    # results in a visually over-smooth appearance.  Repeat, specifying a reduction
    # in the threshold by factors of 2 and 4.
    #im_visushrink2 = denoise_wavelet(NumpyImage, multichannel=True, convert2ycbcr=True, method='VisuShrink', mode='soft', sigma=sigma_est/2, rescale_sigma=True)
    #im_visushrink4 = denoise_wavelet(NumpyImage, multichannel=True, convert2ycbcr=True,method='VisuShrink', mode='soft', sigma=sigma_est/4, rescale_sigma=True)

#list all the python module
import pip
installed_packages = pip._internal.get_installed_distributions()
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
     for i in installed_packages])
print(installed_packages_list)    

help('modules') #to find teir corresponding name    



# coding: utf-8

import unittest
from slicer.ScriptedLoadableModule import *
import logging
from __main__ import vtk, qt, ctk, slicer
from math import *
import numpy as np
from vtk.util import numpy_support
import SimpleITK as sitk
import sitkUtils as su
import time
import datetime
import sys, time, os

import pywt # https://github.com/PyWavelets/pywt/blob/master/pywt/_multilevel.py

Nom_image="imSh_1"
Nom_label="FOUMA_1m-label"

def denoising_nonlocalmeans(Nom_image, Nom_label):
    image=su.PullFromSlicer(Nom_image)
    image=sitk.Shrink(image, [2,2,2])
    label=su.PullFromSlicer(Nom_label)
    timeRMR1 = time.time()
    DenoiseFilter=sitk.PatchBasedDenoisingImageFilter() #Execute (const Image &image1, double kernelBandwidthSigma, uint32_t patchRadius, 
    #uint32_t numberOfIterations, uint32_t numberOfSamplePatches, double sampleVariance, PatchBasedDenoisingImageFilter::NoiseModelType noiseModel, 
    #double noiseSigma, double noiseModelFidelityWeight, bool alwaysTreatComponentsAsEuclidean, bool kernelBandwidthEstimation, double kernelBandwidthMultiplicationFactor, 
    #uint32_t kernelBandwidthUpdateFrequency, double kernelBandwidthFractionPixelsForEstimation)
    DenoiseFilter.SetAlwaysTreatComponentsAsEuclidean(True)
    DenoiseFilter.SetKernelBandwidthEstimation(True)
    DenoiseFilter.SetKernelBandwidthFractionPixelsForEstimation(0.5) #double KernelBandwidthFractionPixelsForEstimation
    #DenoiseFilter.SetKernelBandwidthMultiplicationFactor() #(double KernelBandwidthMultiplicationFactor)
    #DenoiseFilter.SetKernelBandwidthSigma(400) #(double KernelBandwidthSigma)  #faible voire pas d'influence
    #DenoiseFilter.SetKernelBandwidthUpdateFrequency() #(uint32_t KernelBandwidthUpdateFrequency 1 par defaut)
    DenoiseFilter.SetNoiseModel(3) #(NoiseModelType NoiseModel) #NoiseModelType { NOMODEL:0, GAUSSIAN:1, RICIAN:2,  POISSON:3}
    DenoiseFilter.SetNoiseModelFidelityWeight(0.05) #(double NoiseModelFidelityWeight entre 0 et 1)# This weight controls the balance between the smoothing and the closeness to the noisy data. 
    #DenoiseFilter.SetNoiseSigma(0.50) #(double NoiseSigma)#usualy 5% of min max of an image ##############pas d'influence  
    #DenoiseFilter.SetNumberOfIterations(1) #(uint32_t NumberOfIterations 1 par defaut)
    DenoiseFilter.SetNumberOfSamplePatches(200) #(uint32_t NumberOfSamplePatches)#200->100, 41 a 23s mais filtre plus
    DenoiseFilter.SetPatchRadius(4) #(uint32_t PatchRadius) # 2->10s 4->41s 6->121s ##############paramétre critique
    #DenoiseFilter.SetSampleVariance(400) #(double SampleVariance) #pas d'influence?
    ImageDenoised=DenoiseFilter.Execute(image)
    timeRMR2 = time.time()
    TimeForrunFunctionRMR2 = timeRMR2 - timeRMR1
    print(u"La fonction de traitement s'est executée en " + str(TimeForrunFunctionRMR2) +" secondes")
    print("\n")
    print(DenoiseFilter.GetNumberOfSamplePatches()) #200
    print("\n")
    print (DenoiseFilter.GetSampleVariance()) #400
    print("\n")
    print(DenoiseFilter.GetNoiseSigma()) #0.0
    print("\n")
    print(DenoiseFilter.GetNumberOfIterations()) #1
    print("\n")
    print(DenoiseFilter.GetKernelBandwidthSigma()) #400.0
    print("\n")
    stat_filter=sitk.LabelIntensityStatisticsImageFilter()
    stat_filter.Execute(label,image) #attention à l'ordre
    print(stat_filter.GetStandardDeviation(1)/stat_filter.GetMean(1))
    print("\n") 
    stat_filter.Execute(label,ImageDenoised) #attention à l'ordre 
    print(stat_filter.GetStandardDeviation(1)/stat_filter.GetMean(1))   
    su.PushToSlicer(ImageDenoised,'ImageDenoisedbyPatchBasedDenoisingImageFilter')


denoising_nonlocalmeans(Nom_image, Nom_label)


Nom_image="template"

def denoising_nonlocalmeans2(Nom_image):
    image=su.PullFromSlicer(Nom_image)
    Shrinkfactor=2
    image=sitk.Shrink(image, [Shrinkfactor,Shrinkfactor,Shrinkfactor])
    timeRMR1 = time.time()
    DenoiseFilter_init=sitk.PatchBasedDenoisingImageFilter() #Execute (const Image &image1, double kernelBandwidthSigma, uint32_t patchRadius, 
    #uint32_t numberOfIterations, uint32_t numberOfSamplePatches, double sampleVariance, PatchBasedDenoisingImageFilter::NoiseModelType noiseModel, 
    #double noiseSigma, double noiseModelFidelityWeight, bool alwaysTreatComponentsAsEuclidean, bool kernelBandwidthEstimation, double kernelBandwidthMultiplicationFactor, 
    #uint32_t kernelBandwidthUpdateFrequency, double kernelBandwidthFractionPixelsForEstimation)
    DenoiseFilter_init.SetAlwaysTreatComponentsAsEuclidean(True)
    DenoiseFilter_init.SetKernelBandwidthEstimation(True)
    DenoiseFilter_init.SetKernelBandwidthFractionPixelsForEstimation(0.5) #double KernelBandwidthFractionPixelsForEstimation
    #DenoiseFilter.SetKernelBandwidthMultiplicationFactor() #(double KernelBandwidthMultiplicationFactor)
    #DenoiseFilter.SetKernelBandwidthSigma(400) #(double KernelBandwidthSigma)  #faible voire pas d'influence
    #DenoiseFilter.SetKernelBandwidthUpdateFrequency() #(uint32_t KernelBandwidthUpdateFrequency 1 par defaut)
    DenoiseFilter_init.SetNoiseModel(3) #(NoiseModelType NoiseModel) #NoiseModelType { NOMODEL:0, GAUSSIAN:1, RICIAN:2,  POISSON:3}
    DenoiseFilter_init.SetNoiseModelFidelityWeight(0.05) #(double NoiseModelFidelityWeight entre 0 et 1)# This weight controls the balance between the smoothing and the closeness to the noisy data. 
    #DenoiseFilter.SetNoiseSigma(0.50) #(double NoiseSigma)#usualy 5% of min max of an image ##############pas d'influence  
    #DenoiseFilter.SetNumberOfIterations(1) #(uint32_t NumberOfIterations 1 par defaut)
    #DenoiseFilter.SetNumberOfSamplePatches(200) #(uint32_t NumberOfSamplePatches)#200->100, 41 a 23s mais filtre plus
    DenoiseFilter_init.SetPatchRadius(2) #(uint32_t PatchRadius) # 2->10s 4->41s 6->121s ##############paramétre critique
    #DenoiseFilter.SetSampleVariance(400) #(double SampleVariance) #pas d'influence?
    ImageDenoised_init=DenoiseFilter_init.Execute(image)
    timeRMR2 = time.time()
    TimeForrunFunctionRMR2 = timeRMR2 - timeRMR1
    print(u"La fonction de traitement intiale s'est executée en " + str(TimeForrunFunctionRMR2) +" secondes")
    timeRMR1 = time.time()
    DenoiseFilter=sitk.PatchBasedDenoisingImageFilter() #Execute (const Image &image1, double kernelBandwidthSigma, uint32_t patchRadius, 
    #uint32_t numberOfIterations, uint32_t numberOfSamplePatches, double sampleVariance, PatchBasedDenoisingImageFilter::NoiseModelType noiseModel, 
    #double noiseSigma, double noiseModelFidelityWeight, bool alwaysTreatComponentsAsEuclidean, bool kernelBandwidthEstimation, double kernelBandwidthMultiplicationFactor, 
    #uint32_t kernelBandwidthUpdateFrequency, double kernelBandwidthFractionPixelsForEstimation)
    DenoiseFilter.SetAlwaysTreatComponentsAsEuclidean(True)
    DenoiseFilter.SetKernelBandwidthEstimation(False)
    #DenoiseFilter.SetKernelBandwidthFractionPixelsForEstimation(0.5) #double KernelBandwidthFractionPixelsForEstimation
    #DenoiseFilter.SetKernelBandwidthMultiplicationFactor() #(double KernelBandwidthMultiplicationFactor)
    DenoiseFilter.SetKernelBandwidthSigma(DenoiseFilter_init.GetKernelBandwidthSigma()) #(double KernelBandwidthSigma)  #faible voire pas d'influence
    #DenoiseFilter.SetKernelBandwidthUpdateFrequency() #(uint32_t KernelBandwidthUpdateFrequency 1 par defaut)
    DenoiseFilter.SetNoiseModel(3) #(NoiseModelType NoiseModel) #NoiseModelType { NOMODEL:0, GAUSSIAN:1, RICIAN:2,  POISSON:3}
    DenoiseFilter.SetNoiseModelFidelityWeight(0.05) #(double NoiseModelFidelityWeight entre 0 et 1)# This weight controls the balance between the smoothing and the closeness to the noisy data. 
    DenoiseFilter.SetNoiseSigma(DenoiseFilter_init.GetNoiseSigma()) #(double NoiseSigma)#usualy 5% of min max of an image ##############pas d'influence  
    #DenoiseFilter.SetNumberOfIterations(1) #(uint32_t NumberOfIterations 1 par defaut)
    DenoiseFilter.SetNumberOfSamplePatches(DenoiseFilter_init.GetNumberOfSamplePatches()) #(uint32_t NumberOfSamplePatches)#200->100, 41 a 23s mais filtre plus
    DenoiseFilter.SetPatchRadius(DenoiseFilter_init.GetPatchRadius()*Shrinkfactor) #(uint32_t PatchRadius) # 2->10s 4->41s 6->121s ##############paramétre critique
    DenoiseFilter.SetSampleVariance(DenoiseFilter_init.GetSampleVariance()) #(double SampleVariance) #pas d'influence?
    ImageDenoised=DenoiseFilter.Execute(image)
    timeRMR2 = time.time()
    TimeForrunFunctionRMR2 = timeRMR2 - timeRMR1
    print(u"La fonction de traitement final s'est executée en " + str(TimeForrunFunctionRMR2) +" secondes")
    su.PushToSlicer(ImageDenoised,'ImageDenoisedbyPatchBasedDenoisingImageFilter')


denoising_nonlocalmeans2(Nom_image)


# coding: utf-8

import unittest
from slicer.ScriptedLoadableModule import *
import logging
from __main__ import vtk, qt, ctk, slicer
from math import *
import numpy as np
from vtk.util import numpy_support
import SimpleITK as sitk
import sitkUtils as su
import time
import datetime
import sys, time, os
from itertools import *
import six

import pywt # https://github.com/PyWavelets/pywt/blob/master/pywt/_multilevel.py

Nom_image="FOUMA_1m"

a=2.0
d=0.5
DecompMatrixSpacingfactor={
    'aad':[a,a,d], 
    'ada':[a,d,a], 
    'add':[a,d,d], 
    'daa':[d,a,a], 
    'dad':[d,a,d],
    'dda':[d,d,a],
    'ddd':[d,d,d],
}

def reechantillonage_identity(image_ref,image_to_transform):
    identity = sitk.TranslationTransform(3, (0,0,0))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(identity)
    ImageRecaler = resampler.Execute(image_to_transform)
    return ImageRecaler 

def realspace_spacing(list_elem, Zoom):
    DMSF=np.asarray(DecompMatrixSpacingfactor[list_elem], dtype=np.float64)
    Zoom=np.asarray(Zoom, dtype=np.float64)
    RSS=DMSF*Zoom
    return RSS

def cropImagefctLabel(image, LowerBondingBox, UpperBondingBox  ):
  crop=sitk.CropImageFilter()
  image_cropper=crop.Execute(image, LowerBondingBox, UpperBondingBox  )
  return image_cropper


def CreateGaussianKernel(RS, matrice_spacing ):  #to modify to mono exponential
    imageGaussian=sitk.GaussianImageSource()
    imageGaussian.SetOutputPixelType(sitk.sitkUInt16)
    size=ceil(3*RS/matrice_spacing)
    if (size % 2)==0 :
        size=size+1
    imageGaussian.SetSize([size,size,size])   #taille=size*spacing
    sigma=(RS/2.35)/matrice_spacing
    imageGaussian.SetSigma([sigma,sigma,sigma])  #FWHM/2.35 remaruqe ten 4.29*sigma
    imageGaussian.SetMean([0,0,0])  #centre image=mean/spacing
    imageGaussian.SetScale(100)
    imageGaussian.SetOrigin([-((size-1)/2),-((size-1)/2),-((size-1)/2)])
    imageGaussian.SetSpacing([1,1,1])
    imageGaussian.SetDirection([1,0,0,0,1,0,0,0,1])
    kernel=imageGaussian.Execute()
    #♣su.PushToSlicer(kernel,"kernel",1)
    return kernel  

def RLdeconvolutionTV(image,kernel,alpha):
    ################initialisation#############
    RL=sitk.RichardsonLucyDeconvolutionImageFilter()
    laplacian= sitk.LaplacianImageFilter()
    normgradient=sitk.GradientMagnitudeImageFilter()
    divide=sitk.DivideImageFilter()
    multiply=sitk.MultiplyImageFilter()
    Substract=sitk.SubtractImageFilter()
    Cast=sitk.CastImageFilter()
    ##########terme regularisation##############
    image_cast=sitk.Cast(image,sitk.sitkFloat64)
    L=laplacian.Execute(image_cast)
    NG=normgradient.Execute(image_cast)
    NG=sitk.Cast(NG,sitk.sitkFloat64)
    i1=divide.Execute(L, NG )
    i2=multiply.Execute( i1, alpha) #landaTV=0.02
    i3=Substract.Execute(1,i2)
    i4=divide.Execute(1,i3)
    ##############deconvolution#########
    Niteration=1
    Normalise=True
    BoundaryCondition=1 #zerofluxNemaanpad
    OutputRegionMode=0 #same
    image_cast=sitk.Cast(image,sitk.sitkUInt16)
    imagedecon=RL.Execute(image_cast,kernel,Niteration, Normalise, BoundaryCondition,OutputRegionMode)
    imagedecon=sitk.Cast(imagedecon,sitk.sitkFloat64)
    imagedeconRLTV=multiply.Execute(imagedecon,i4)
    return imagedeconRLTV


def SpatialFrequencyOptim2(matrix):
    sq_diff = 0.0
    size=matrix.shape
    dim=len(size)  
    for i in range(dim): #iterate over all image dimensions
        slc1 = [slice(None)]*dim
        slc1[i] = slice(0,size[i]-1)
        slc2 = [slice(None)]*dim
        slc2[i] = slice(1,size[i])
        sq_diff+= np.sum((matrix[tuple(slc2)]- matrix[tuple(slc1)])**2)
    return sq_diff/np.prod(size)

def denoising_nonlocalmeans(image, nom, radius, Niteration):
    timeRMR1 = time.time()
    #su.PushToSlicer(image,'image_Origine'+str(nom))
    DenoiseFilter=sitk.PatchBasedDenoisingImageFilter() 
    DenoiseFilter.SetAlwaysTreatComponentsAsEuclidean(True)
    DenoiseFilter.SetKernelBandwidthEstimation(True)
    #DenoiseFilter.SetKernelBandwidthFractionPixelsForEstimation(0.5) #double KernelBandwidthFractionPixelsForEstimation
    #DenoiseFilter.SetKernelBandwidthMultiplicationFactor() #(double KernelBandwidthMultiplicationFactor)
    #DenoiseFilter.SetKernelBandwidthSigma(400) #(double KernelBandwidthSigma)  #faible voire pas d'influence
    #DenoiseFilter.SetKernelBandwidthUpdateFrequency() #(uint32_t KernelBandwidthUpdateFrequency 1 par defaut)
    DenoiseFilter.SetNoiseModel(0) #(NoiseModelType NoiseModel) #NoiseModelType { NOMODEL:0, GAUSSIAN:1, RICIAN:2,  POISSON:3}
    DenoiseFilter.SetNoiseModelFidelityWeight(0.05) #(double NoiseModelFidelityWeight entre 0 et 1)# This weight controls the balance between the smoothing and the closeness to the noisy data. 
    #DenoiseFilter.SetNoiseSigma(0.50) #(double NoiseSigma)#usualy 5% of min max of an image ##############pas d'influence  
    DenoiseFilter.SetNumberOfIterations(Niteration) #(uint32_t NumberOfIterations 1 par defaut)
    #DenoiseFilter.SetNumberOfSamplePatches(200) #(uint32_t NumberOfSamplePatches)#200->100, 41 a 23s mais filtre plus
    DenoiseFilter.SetPatchRadius(radius) #(uint32_t PatchRadius) # 2->10s 4->41s 6->121s ##############paramétre critique
    #DenoiseFilter.SetSampleVariance(400) #(double SampleVariance) #pas d'influence?
    ImageDenoised=DenoiseFilter.Execute(image)
    #su.PushToSlicer(ImageDenoised,'image_Origine_Denoised'+str(nom))
    timeRMR2 = time.time()
    TimeForrunFunctionRMR2 = timeRMR2 - timeRMR1
    print(u"    NLM-denoising of " + str(nom) +" matrix:")
    print(u"    Le rayon analyser est " + str(radius) +" voxel")
    print(u"    La fonction denoising_nonlocalmeans s'est executée en " + str(TimeForrunFunctionRMR2) +" secondes")
    print("\n")
    return ImageDenoised


def ParchBasedandOndeletteDenoising(Nom_image,denoising,correctPVE):
    timeRMR1 = time.time()
    image=su.PullFromSlicer(Nom_image)
    ####crop pour acceleration############################################
    label_complet=sitk.BinaryThreshold(image, 0.1, 500, 1,0)
    label_complet=sitk.ConnectedComponent(label_complet, True)
    label_complet=sitk.RelabelComponent(label_complet)
    stats= sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(label_complet,image)
    delta=0 #extention du label pour eviter les problemes aux bords
    LowerBondingBox=[stats.GetBoundingBox(1)[0]-delta,stats.GetBoundingBox(1)[1]-delta,stats.GetBoundingBox(1)[2]-delta]
    UpperBondingBox=[image.GetSize()[0]-(stats.GetBoundingBox(1)[0]+stats.GetBoundingBox(1)[3]+delta),image.GetSize()[1]-(stats.GetBoundingBox(1)[1]+stats.GetBoundingBox(1)[4]+delta),image.GetSize()[2]-(stats.GetBoundingBox(1)[2]+stats.GetBoundingBox(1)[5]+delta)]
    image=cropImagefctLabel(image, LowerBondingBox, UpperBondingBox  )
    ###############################################################
    ##########################wavelets decomposition#############
    ###########################nlm denoising###############
    image_spacing=image.GetSpacing()
    image_size=image.GetSize()
    NumpyImage=sitk.GetArrayFromImage(image)
    max_lev = 2       # how many levels of decomposition to draw
    #pywt.swt_max_level(input_len) #give the maw level
    radius=16  # in mm critique pour le temps et dans quelle rayon on peut trouver des voxels similaires
    Niteration=1 #nombre d'iteration pour le denoising
    c = pywt.wavedecn(NumpyImage, 'coif1', mode='zero', level=max_lev) #voir https://pywavelets.readthedocs.io/en/latest/ref/nd-dwt-and-idwt.html#pywt.wavedecn
    #c = pywt.swtn(NumpyImage, wavelet='db2', level=max_lev, start_level=0, axes=None, trim_approx=False, norm=False) #voir https://pywavelets.readthedocs.io/en/latest/ref/nd-dwt-and-idwt.html#pywt.wavedecn   
    #c= pywt.swtn(NumpyImage, 'coif1', 0, max_lev, None)
    #c_arr,c_slices=swt3(image, "coif1", max_lev, 0)
    c_arr,c_slices= pywt.coeffs_to_array(c, padding=0, axes=None) #separe les sous matrices aprés decomposition et leur indices
    list_elem=[]
    for row in c_slices:
        for elem in row:
            list_elem.append(elem) #list de tous les elements de matrix [aad,add,ddd] a: average, d: detail
    print(list_elem)
    for level in range(1,max_lev+1):
        for keys in range(int((len(list_elem)-3)/max_lev)):
            matrix=c_arr[c_slices[level][list_elem[keys+3]]]
            matrix_size=matrix.shape
             ##in mm# image have to be isotropic
            #radius_voxels=ceil(radius/matrice_spacing)
            print(u"NLM-denoising of "+str(level)+" level " + str(list_elem[keys+3]) +" matrix")
            print(u"Matrix size "+str(matrix_size)+" matrice_spacing " + str(matrice_spacing) +" mm")
            print(u"Le rayon analyser est de " + str(radius_voxels) +" voxel")
            print(u"La frequence spatiale est de  " + str(SpatialFrequencyOptim2(matrix)) +" voxel")
            print("\n")
            image_ondelette=sitk.GetImageFromArray(matrix)
            Zoom=[image_size[0]/matrix_size[0],image_size[1]/matrix_size[1], image_size[2]/matrix_size[2]]
            SpacingImageOndelette_realspace=realspace_spacing(list_elem, Zoom)
            SpacingImageOndelette=np.asarray(DecompMatrixSpacingfactor[list_elem], dtype=np.float64)
            image_ondelette.SetSpacing(SpacingImageOndelette)
            #image_ondelette=reechantillonage_identity(image,image_ondelette)
            ######################################################################################
            #####################################deconvolution###################################
            if (correctPVE==1):
                limitRC=13 #limite taille en mm pour RC<0.95
                RS=4 #spatiale resolution en mm of the system 
                if (<limitRC):
                    kernel=CreateGaussianKernel(RS, min(SpacingImageOndelette_realspace))
                    alpha=0.02
                    RC=0.0937*min(SpacingImageOndelette_realspace)
                    iterRC=0
                    while (image.GetMaximum() <(np.max(matrix)/(3*RC)) ):
                        iterRC=iterRC+1    
                        image_ondelette=RLdeconvolutionTV(image_ondelette,kernel, alpha)
                    print(iterRC) 
                    print("deconvolution ok")
            ##################################################################################
            ##################################################################################
            #il faudrait faire une deconvolution RL avant le denoising? avec comme citére d'arret les coefficient de recovery RC
            #if (2*np.std(matrix)/(np.max(matrix)+abs(np.min(matrix))<0.01):
            #####################################Denoising########################################
            ######################################################################################
            if (denoising==1):
                if (SpatialFrequencyOptim2(matrix)>0.1 and (radius>max(SpacingImageOndelette_realspace))):# contraintes suffisament d'information pour impact sup a suv de 0.01 et radius pas trop grand par rapport image
                    image_ondelette=denoising_nonlocalmeans(image_ondelette,list_elem[keys+3],ceil(radius/max(SpacingImageOndelette_realspace)), Niteration )
            ########################################################################################
            ######################################################################################
            image_ondelette_TranformSpace=
            image_ondelette=reechantillonage_identity(image_ondelette,image_ondelette_realspace)
            image_ondelette_realspace=sitk.GetArrayFromImage(image_ondelette_denoised)
            c_arr[c_slices[level][list_elem[keys+3]]]=matrix_ondelette
    c=pywt.array_to_coeffs(c_arr,c_slices) #recombine les sous matrices apres decomposition et leur indices
    matrice_ondelette=pywt.waverecn(c, 'db2') #decomposition en ondeleet inverse
    #matrice_ondelette=pywt.iswtn(c_arr, 'db2',max_lev)
    image_ondelette=sitk.GetImageFromArray(matrice_ondelette)
    image_ondelette.SetSpacing(image.GetSpacing())
    image_ondelette.SetDirection(image.GetDirection())
    image_ondelette.SetOrigin(image.GetOrigin())
    su.PushToSlicer(image_ondelette,'image_Denoised_final')
    timeRMR2 = time.time()
    TimeForrunFunctionRMR2 = timeRMR2 - timeRMR1
    print(u"La fonction de traitement total s'est executée en " + str(TimeForrunFunctionRMR2) +" secondes")

ParchBasedandOndeletteDenoising(Nom_image,denoising=False,correctPVE=False)

#################################################deconvolution
#inspirer de py radiomics

def getWaveletImage(inputImage, **kwargs):
  """
  Apply wavelet filter to image and compute signature for each filtered image.

  Following settings are possible:

  - start_level [0]: integer, 0 based level of wavelet which should be used as first set of decompositions
    from which a signature is calculated
  - level [1]: integer, number of levels of wavelet decompositions from which a signature is calculated.
  - wavelet ["coif1"]: string, type of wavelet decomposition. Enumerated value, validated against possible values
    present in the ``pyWavelet.wavelist()``. Current possible values (pywavelet version 0.4.0) (where an
    aditional number is needed, range of values is indicated in []):

    - haar
    - dmey
    - sym[2-20]
    - db[1-20]
    - coif[1-5]
    - bior[1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, 6.8]
    - rbio[1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, 6.8]

  Returned filter name reflects wavelet type:
  wavelet[level]-<decompositionName>

  N.B. only levels greater than the first level are entered into the name.

  :return: Yields each wavelet decomposition and final approximation, corresponding filter name and ``kwargs``
  """
  global logger

  logger.debug("Generating Wavelet images")

  approx, ret = _swt3(inputImage, kwargs.get('wavelet', 'coif1'), kwargs.get('level', 1), kwargs.get('start_level', 0))

  for idx, wl in enumerate(ret, start=1):
    for decompositionName, decompositionImage in wl.items():
      print('Computing Wavelet %s', decompositionName)

      if idx == 1:
        inputImageName = 'wavelet-%s' % (decompositionName)
      else:
        inputImageName = 'wavelet%s-%s' % (idx, decompositionName)
      print('Yielding %s image', inputImageName)
      yield decompositionImage, inputImageName, kwargs

  if len(ret) == 1:
    inputImageName = 'wavelet-LLL'
  else:
    inputImageName = 'wavelet%s-LLL' % (len(ret))
  print('Yielding approximation (%s) image', inputImageName)
  yield approx, inputImageName, kwargs



#def _swt3(inputImage, wavelet="coif1", level=1, start_level=0):
def swt3(inputImage, wavelet, level, start_level):
  matrix = sitk.GetArrayFromImage(inputImage)
  matrix = np.asarray(matrix)
  if matrix.ndim != 3:
    raise ValueError("Expected 3D data array")
  original_shape = matrix.shape
  adjusted_shape = tuple([dim + 1 if dim % 2 != 0 else dim for dim in original_shape])
  data = matrix.copy()
  data.resize(adjusted_shape, refcheck=False)
  if not isinstance(wavelet, pywt.Wavelet):
    wavelet = pywt.Wavelet(wavelet)
  for i in range(0, start_level):
    H, L = _decompose_i(data, wavelet)
    LH, LL = _decompose_j(L, wavelet)
    LLH, LLL = _decompose_k(LL, wavelet)
    data = LLL.copy()
  ret = []
  for i in range(start_level, start_level + level):
    H, L = _decompose_i(data, wavelet)
    HH, HL = _decompose_j(H, wavelet)
    LH, LL = _decompose_j(L, wavelet)
    HHH, HHL = _decompose_k(HH, wavelet)
    HLH, HLL = _decompose_k(HL, wavelet)
    LHH, LHL = _decompose_k(LH, wavelet)
    LLH, LLL = _decompose_k(LL, wavelet)
    data = LLL.copy()
    dec = {'HHH': HHH,
           'HHL': HHL,
           'HLH': HLH,
           'HLL': HLL,
           'LHH': LHH,
           'LHL': LHL,
           'LLH': LLH}
    for decName, decImage in six.iteritems(dec):
      decTemp = decImage.copy()
      decTemp = np.resize(decTemp, original_shape)
      sitkImage = sitk.GetImageFromArray(decTemp)
      sitkImage.CopyInformation(inputImage)
      dec[decName] = sitkImage
    ret.append(dec)
  data = np.resize(data, original_shape)
  approximation = sitk.GetImageFromArray(data)
  approximation.CopyInformation(inputImage)
  #return approximation
  return approximation, ret


def _decompose_i(data, wavelet):
  # process in i:
  H, L = [], []
  i_arrays = chain.from_iterable(data)
  for i_array in i_arrays:
    cA, cD = pywt.swt(i_array, wavelet, level=1, start_level=0)[0]
    H.append(cD)
    L.append(cA)
  H = np.hstack(H).reshape(data.shape)
  L = np.hstack(L).reshape(data.shape)
  return H, L


def _decompose_j(data, wavelet):
  # process in j:
  s = data.shape
  H, L = [], []
  j_arrays = chain.from_iterable(np.transpose(data, (0, 2, 1)))
  for j_array in j_arrays:
    cA, cD = pywt.swt(j_array, wavelet, level=1, start_level=0)[0]
    H.append(cD)
    L.append(cA)
  H = np.hstack(H).reshape((s[0], s[2], s[1])).transpose((0, 2, 1))
  L = np.hstack(L).reshape((s[0], s[2], s[1])).transpose((0, 2, 1))
  return H, L


def _decompose_k(data, wavelet):
  # process in k:
  H, L = [], []
  k_arrays = chain.from_iterable(np.transpose(data, (2, 1, 0)))
  for k_array in k_arrays:
    cA, cD = pywt.swt(k_array, wavelet, level=1, start_level=0)[0]
    H.append(cD)
    L.append(cA)
  H = np.asarray([slice for slice in np.split(np.vstack(H), data.shape[2])]).T
  L = np.asarray([slice for slice in np.split(np.vstack(L), data.shape[2])]).T
  return H, L



########################test########################
import unittest
from slicer.ScriptedLoadableModule import *
import logging
from __main__ import vtk, qt, ctk, slicer
from math import *
import numpy as np
from vtk.util import numpy_support
import SimpleITK as sitk
import sitkUtils as su
import time
import datetime
import sys, time, os
from itertools import *
import six

import pywt # https://github.com/PyWavelets/pywt/blob/master/pywt/_multilevel.py
from skimage.restoration import denoise_wavelet, cycle_spin
from skimage.metrics import peak_signal_noise_ratio
import skimage

Nom_image="FOUMA_1m"

def skimage_shift_invariant_wavelet(Nom_image):
    image=su.PullFromSlicer(Nom_image)
    NumpyImage=sitk.GetArrayFromImage(image)
    # Repeat denosing with different amounts of cycle spinning.  e.g.
    # max_shift = 0 -> no cycle spinning
    # max_shift = 1 -> shifts of (0, 1) along each axis
    # max_shift = 3 -> shifts of (0, 1, 2, 3) along each axis
    # etc...
    denoise_kwargs = dict(multichannel=False, convert2ycbcr=False, wavelet='db2', rescale_sigma=True,method='BayesShrink')
    all_psnr = []
    max_shifts = [0, 1, 3, 5]
    for n, s in enumerate(max_shifts):
        im_bayescs = cycle_spin(NumpyImage, func=denoise_wavelet, max_shifts=s, func_kw=denoise_kwargs, multichannel=False)
        #psnr = peak_signal_noise_ratio(NumpyImage, im_bayescs, )
        #all_psnr.append(psnr)
        #print("shift: "+str(s+1)+" psnr "+str(psnr))
        #print("\n")
        image_ondelette=sitk.GetImageFromArray(im_bayescs)
        image_ondelette.SetSpacing(image.GetSpacing())
        image_ondelette.SetDirection(image.GetDirection())
        image_ondelette.SetOrigin(image.GetOrigin())
        su.PushToSlicer(image_ondelette,'image_Denoised_final_shifts_'+str(s+1))

skimage_shift_invariant_wavelet(Nom_image)

skimage.measure.compare_psnr(im_true, im_test)
skimage.measure.compare_mse(im1, im2)
skimage.measure.compare_ssim()
sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)