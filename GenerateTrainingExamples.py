from skimage import io
import numpy as np
import pandas as pd
import h5py
from PIL import Image
import scipy.signal
from os.path import abspath
import argparse
import matplotlib.pyplot as plt
from ClearFromBounduary import ClearFromBoundary

#Skypt służący przygotowani danych treningowych na podstawie obrazów wygenerownych przez Thunderstorm i odpowiadającym 
#im współrzędnym emiterów

def Generate_training_examples (tiff, csv, camera_pixelsize, savename, \
                                 upsampling_factor=8,  gaussian_sigma=1, emiters = 7):
    
    #rozmiar treningowe obazka; musi być podzielny przez 8, ze względu na kontrukscję modelu
    patch_size = 26*upsampling_factor #  [pixels]
    
    #maksymalna liczba obrazków z jednego obrazu
    num_patches = 500
    
    #maksymalna liczba obrazków treningowych
    maxExamples = 10000
    
    #minimalna liczba emiterów na jednym obrazków, by został przyjęty
    minEmitters = emiters
    
    #Zaczytanie danych wygenerowanych przez ThunderStorm
    ImageStack = io.imread(tiff, plugin='tifffile')
    
    #Wymiary pierowtnych obrazów
    numImages, M, N= ImageStack.shape
    
    #Wymiary wyjściowych obrazków
    Mhr = upsampling_factor*M
    Nhr = upsampling_factor*N
    
    patch_size_hr = camera_pixelsize/upsampling_factor # nm
    
    #analogon funkcji z matlaba, nie zaimplemetowana wprost w Pythonie
    def matlab_style_gauss2D(shape=(7,7),sigma=gaussian_sigma):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    
    psfHeatmap = matlab_style_gauss2D()
    ntrain = min(numImages*num_patches,maxExamples)
    
    #Zainicjowanie obrazków treningowych oraz odpowidających im obrazków-celów
    patches = np.zeros((ntrain, patch_size,patch_size))
    heatmaps = np.zeros((ntrain, patch_size,patch_size))
    spikes = np.full((ntrain ,patch_size,patch_size), False)
    
    #Import współrzędnych z pliku CSV
    Activations = pd.read_csv(csv) 
    Data = Activations.values 
    col_names = Activations.columns 
    
    #Sprawdzenie poprawności zaczytanych działań
    if len(col_names) < 7:
        print('Number of columns in the ThunderSTORM csv file is less than 7!')
    
    def nnz(array):
        array=np.array(array)
        num = len(array[array!=0])
        return num
    
    k = 1
    skip_counter = 0
    
    for frmNum in range (numImages):
        #zamiana obazka z formatu .tiff na np.array
        y = ImageStack[frmNum,:,:] #np iloc?
        y = Image.fromarray(y)
        # Powiększenie obrazka za pomocą algortmu najbliższych sąsiadów
        yus = y.resize((Nhr, Mhr), resample = Image.NEAREST)
        yus = np.array(yus)
        
        #Dopasowanie współrzędnych do obecnego powiększonego obrazka
        DataFrame = Data[Data[:,0]==(frmNum+1),:]
    
        Chr_emitters = np.maximum(np.minimum(np.round(DataFrame[:,1]/patch_size_hr),Mhr),1)
        Rhr_emitters = np.maximum(np.minimum(np.round(DataFrame[:,2]/patch_size_hr),Nhr),1)
    
        #stworzenie map ciepła i obrazków docelowych
        SpikesImage = np.zeros([Mhr, Nhr])
        SpikesImage[Rhr_emitters.astype(int)-1, Chr_emitters.astype(int)-1] = 1 
        HeatmapImage = scipy.signal.convolve2d(SpikesImage, psfHeatmap, mode='same')
        
        #limitowanie ilości wygenerownych obrazów
        if k > ntrain:
            break
        else:
            # z powiększonego obrazu wyjściowego wybieramy losowo podobrazek
            indx, indy = ClearFromBoundary([Mhr, Nhr],np.ceil(patch_size/2),num_patches)
            floor = int(np.floor(patch_size/2))
    
            # sprawdzenie czy wylosowany obrazek zawiera odpowiednią liczbę emiterów 
            for i in range(len(indx)):
                if k > ntrain:
                    break
                if nnz(SpikesImage[indx[i]-floor+1:indx[i]+floor+1, 
                                   indy[i]-floor+1:indy[i]+floor+1]) < minEmitters:
                    skip_counter = skip_counter + 1
                    continue
                else:
                    patches[k-1,:,:] = yus[indx[i]-floor+1:indx[i]+floor+1, indy[i]-floor+1:indy[i]+floor+1]  
                    heatmaps[k-1,:,:] = HeatmapImage[indx[i]-floor+1:indx[i]+floor+1, indy[i]-floor+1:indy[i]+floor+1]
                    spikes[k-1,:,:] = SpikesImage[indx[i]-floor+1:indx[i]+floor+1, indy[i]-floor+1:indy[i]+floor+1]
                    k = k + 1 
                    
    patches = patches[:k-1,:,:]
    heatmaps = heatmaps[:k-1,:,:]
    spikes = spikes[:k-1,:,:]
    
    #Zapisanie otrzymanych rezultatów 
    f = h5py.File(savename, "w")
    dset = f.create_dataset('patches',  data = patches)
    dset2 = f.create_dataset('heatmaps',  data = heatmaps)
    dset3 = f.create_dataset('spikes',  data = spikes)
    
    f.close()
    
    #Wizualizacja przykładowego wejścia i docelowego wyjścia
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    grey = plt.cm.gray
    ax1.imshow(patches[0,:,:], cmap=grey)
    ax1.set_title('Obraz oryginalny')
    ax2.imshow(spikes[0,:,:], cmap=grey)
    ax2.set_title('Obraz docelowy')
    f.subplots_adjust(hspace=0)
    plt.show()



if __name__ == '__main__':
    
    # startujemy parser
    parser = argparse.ArgumentParser()
    
    # scieżka do obrazków wygenerowanych przez ThunderStorm
    parser.add_argument('--tiff',type=str, help="path to tiff stack for generate more examples")
    
    #ścieżka do pliku .csv 
    parser.add_argument('--csv', help="path to csv-file")
    
    #rozmiar pixeli_kamery (jeden z paremetrów używanych przy generowaniu)
    parser.add_argument('--camera_pixelsize', type=int)
    
    # ścieżka do pliku, gdzie bedą zapisywane rezultaty
    parser.add_argument('--savename', type=str, help="path for saving the examples to training")
    
    # rozmiar powiększenia 
    parser.add_argument('--upsampling_factor', type=int, default=8, help="desired upsampling factor")
    # minimalna liczba emiterów
    parser.add_argument('--emiters', type=int, default=7, help="minimalna liczba emiterów")
                        
    args = parser.parse_args()
    
    # wywołanie funkcji
    Generate_training_examples(abspath(args.tiff), abspath(args.csv), \
               args.camera_pixelsize, abspath(args.savename), \
               args.upsampling_factor, args.emiters)
