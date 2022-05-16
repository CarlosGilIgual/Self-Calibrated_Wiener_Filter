import numpy as np


# GAUSSIAN DEFINITION
def gaussian(mu, var, x):
    return np.exp(- 0.5 * (x - mu)**2 / var) / (np.sqrt(2 * var * np.pi))


# GAUSSIAN COMPOSITION
def signal_fraction(x, y):
    # CALCULATING THE RMS
    xrms = np.sqrt(np.sum(((x**2) * y) / np.sum(y)))
    # CREATING MATRIX
    Fs = np.zeros(np.shape(x)[0])
    Fs[np.where(x > xrms)] = 1
    i = 0
    Fs_prev = np.ones(np.shape(Fs))
    while(np.max(np.abs(Fs - Fs_prev)) > 1e-5):
        i += 1
        Fs_prev = np.copy(Fs)
        # ALGORITHM
        mu_signal = (np.sum(x * y * Fs)) / (np.sum(Fs * y))
        mu_noise = (np.sum(x * y * (1 - Fs))) / (np.sum((1 - Fs) * y))
        var_signal = (np.sum((x - mu_signal)**2 * Fs * y)) / (np.sum(Fs * y))
        var_noise = (np.sum((x - mu_noise)**2 * (1 - Fs) * y)) / (np.sum((1 - Fs) * y))
        # GAUSSIANS
        G_signal = gaussian(mu_signal, var_signal, x)
        G_noise = gaussian(mu_noise, var_noise, x)
        # IDENTIFYING PROBLEMATIC POINTS
        notnan = np.where((G_signal + G_noise) != 0)
        # SIGNAL FRACTION
        Fs[notnan] = G_signal[notnan] / (G_signal[notnan] + G_noise[notnan])
        Fs[np.where((G_signal + G_noise) == 0)] = 1
    Fs[:np.where(Fs == np.min(Fs))[-1][-1]] = 0
    Fs[np.where(Fs == np.max(Fs))[-1][0]:] = 1
    return Fs


# FILTER
def Wiener_filter(dat):
    
    
    # DATA DIMENSION
    dim = len(dat.shape)
    
    
    if (dim == 2):
        
        # DATA-FFT PARAMETERS (POWER AND FREQUENCY)
        y_FFT = np.fft.fft2(dat)
        power = np.abs(y_FFT)
        k_y = np.fft.fftfreq(y_FFT.shape[0], d=2)
        k_x = np.fft.fftfreq(y_FFT.shape[1], d=2)
        K2 = (k_y**2)[:,np.newaxis] + (k_x**2)[np.newaxis,:]
        
        # POWER HISTOGRAM
        n_P, bins = np.histogram(power.flatten(), bins=np.logspace(np.min(np.log10(power)), np.max(np.log10(power)), 101))
        P = (bins[:-1] + bins[1:]) / 2
        
        # FILTER BY POWER
        Fs_P = signal_fraction(P, n_P)
        
        # SORTING FREQUENCY2
        order_K2 = np.argsort(K2.flatten())
        K2_ordered = K2.flatten()[order_K2]
        Fs_K2 = np.interp(power.flatten(), P, Fs_P)[order_K2]
        
        # FREQUENCY2 HISTOGRAM
        n_K2, bins = np.histogram(K2_ordered, bins=np.logspace(np.min(np.log10(K2[1:])), np.max(np.log10(K2[1:])), 11))
        K2_average = (bins[:-1] + bins[1:]) / 2
        
        # FILTER AVERAGE WRT FREQUENCY2
        cumsum_Fs = np.cumsum(Fs_K2)
        cumsum_n = np.cumsum(n_K2)
        Fs_K2_average = np.zeros(len(n_K2))
        for i in range(len(n_K2)):
            if (n_K2[i] != 0):
                if (i == 0):
                    Fs_K2_average[i] = (cumsum_Fs[int(cumsum_n[i])] - cumsum_Fs[0]) / n_K2[i]
                else:
                    Fs_K2_average[i] = (cumsum_Fs[int(cumsum_n[i])] - cumsum_Fs[int(cumsum_n[i-1])]) / n_K2[i]
        
        # FILTERING
        Wiener = np.sqrt(np.interp(abs(y_FFT), P, Fs_P) * np.interp(K2, K2_average, Fs_K2_average))
        y_FFT_filtered = y_FFT * Wiener
        y_filtered = np.real(np.fft.ifft2(y_FFT_filtered))
        
    elif (dim == 1):
        
        # DATA-FFT PARAMETERS (POWER AND FREQUENCY)
        y_FFT = np.fft.fft(dat)
        power = np.abs(y_FFT)
        K2 = np.fft.fftfreq(y_FFT.shape[0], d=2)**2
        
        # POWER HISTOGRAM
        n_P, bins = np.histogram(power.flatten(), bins=np.logspace(np.min(np.log10(power)), np.max(np.log10(power)), 101))
        P = (bins[:-1] + bins[1:]) / 2
        
        # FILTER BY POWER
        Fs_P = signal_fraction(P, n_P)
        
        # SORTING FREQUENCY2
        order_K2 = np.argsort(K2)
        K2_ordered = K2[order_K2]
        
        # FREQUENCY2 HISTOGRAM
        n_K2, bins = np.histogram(K2_ordered, bins=np.logspace(np.min(np.log10(K2[1:])), np.max(np.log10(K2[1:])), 11))
        
        # FILTER AVERAGE WRT FREQUENCY2
        cumsum_Fs = np.cumsum(Fs_K2)
        cumsum_n = np.cumsum(n_K2)
        Fs_K2_average = np.zeros(len(n_K2))
        for i in range(len(n_K2)):
            if (n_K2[i] != 0):
                if (i == 0):
                    Fs_K2_average[i] = (cumsum_Fs[int(cumsum_n[i])] - cumsum_Fs[0]) / n_K2[i]
                else:
                    Fs_K2_average[i] = (cumsum_Fs[int(cumsum_n[i])] - cumsum_Fs[int(cumsum_n[i-1])]) / n_K2[i]
                
        # FILTERING
        Wiener = np.sqrt(np.interp(abs(y_FFT), P, Fs_P) * np.interp(K2, K2_average, Fs_K2_average))
        y_FFT_filtered = y_FFT * Wiener
        y_filtered = np.real(np.fft.ifft(y_FFT_filtered))
        
    elif (dim != 1) or (dim != 2):
        print("Warning: Size of array not supported")
    
    
    return y_filtered