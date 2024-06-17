import numpy as np
from itertools import combinations

BATCH_SIZE = 100
EPOCHS = 250
VAL_BATCH_SIZE = 30

# From Khushaba, R. N., A. H. Al-Timemy, A. Al-Ani, and A. Al-Jumaily. "A Framework of Temporal-Spatial Descriptors-Based Feature Extraction 
# for Improved Myoelectric Pattern Recognition." Ieee Transactions on Neural Systems and Rehabilitation Engineering 25, no. 10 (2017): 1821-31. 

# Copyright (c) 2017, Rami Khushaba 
# All rights reserved.

# Root squared zero order moment normalized  
# Equation (2) and (6) with lambda = 0.1
def compute_m0_rms(vector):
    m0 = np.sqrt(np.nansum(np.square(vector), axis=0))[:, np.newaxis]
    return m0 ** .1 / .1

# Root squared 2nd and 4th order moments normalized
# Equation (3), (4), (5), and (6)
def compute_m2_rms(vector, channels, samples):
    d1 = np.gradient(np.concatenate([np.zeros((1, channels)), vector[:-1]]), axis=0)
    m2 = np.sqrt(np.nansum(np.square(d1), axis=0)/samples)[:, np.newaxis]
    return m2 ** .1 / .1

def compute_m4_rms(vector, channels, samples):
    d1 = np.gradient(np.concatenate([np.zeros((1, channels)), vector[:-1]]), axis=0)
    d2 = np.gradient(np.concatenate([np.zeros((1, channels)), d1[:-1]]), axis=0)
    m4 = np.sqrt(np.nansum(np.square(d2), axis=0)/samples)[:, np.newaxis]
    return m4 ** .1 / .1

# Sparseness
# Equation (8)
def compute_sparseness(m0, m2, m4):
    S = m0 / np.sqrt(np.abs(np.multiply((m0-m2)**2, (m0-m4)**2)))
    return S

# Irregularity Factor
# Equation (9)
def compute_irregularity_factor(m0, m2, m4):
    IF = m2 / np.multiply(m0, m4)
    return IF

# Coefficient of Variation
# Equation (10)
def compute_cov(vector, samples):
    mean = np.nanmean(vector, axis=0)
    if 0 in mean:   # avoid divison by zero case 
        mean[mean==0] = 1e-10
    std = np.sqrt(np.nansum(np.square(vector - mean), axis=0)/(samples-1))
    COV = std/mean
    return COV[:, np.newaxis]

# Teager-Kaiser Energy Operator
# Equation (11)
def compute_teager_kaiser_eo(vector, channels, samples):
    d1 = np.gradient(np.concatenate([np.zeros((1, channels)), vector[:-1]]), axis=0)
    d2 = np.gradient(np.concatenate([np.zeros((1, channels)), d1[:-1]]), axis=0)
    TEA = np.nansum(d1**2 - np.multiply(vector[0:samples,:],d2),axis=0)[:,np.newaxis]
    return TEA

def getTDDfeatures_for_one_window(vector):
    
    # Get the size of the input signal
    samples, channels = vector.shape

    if channels > samples:
        vector = np.transpose(vector)
        samples, channels = channels, samples
        
    m0 = compute_m0_rms(vector)
    m2 = compute_m2_rms(vector, channels, samples)
    m4 = compute_m4_rms(vector, channels, samples)
    S = compute_sparseness(m0, m2, m4)
    IF = compute_irregularity_factor(m0, m2, m4)
    COV = compute_cov(vector, samples)
    TKEO = compute_teager_kaiser_eo(vector, channels, samples)
    
    # All features together
    # Maybe similar to Equation (11)
    STDD = np.nanstd(m0, axis=0, ddof=1)[:, np.newaxis]

    if channels > 2:
        Feat = np.concatenate((m0 / STDD, (m0 - m2) / STDD, (m0 - m4) / STDD, S, IF, COV, TKEO), axis=0)
    else:
        Feat = np.concatenate((m0, m0 - m2, m0 - m4, S, IF, COV, TKEO), axis=0)

    Feat = np.log(np.abs(Feat)).flatten()

    return Feat

def getTSD(all_channels_data_in_window):
    
    # x should be a numpy array
    all_channels_data_in_window = np.swapaxes(np.array(all_channels_data_in_window), 1, 0)
    
    # Get the size of the input signal
    samples, channels = all_channels_data_in_window.shape

    if channels > samples:
        all_channels_data_in_window = np.transpose(all_channels_data_in_window)
        samples, channels = channels, samples
    
    if len(all_channels_data_in_window.shape) == 1:
        all_channels_data_in_window = all_channels_data_in_window[:, np.newaxis]
    
    # Prepare indices of each 2 channels combination
    # NCC = Number of channels to combine
    NCC = 2
    Indx = np.array(list(combinations(range(channels), NCC)))   # (28,2)
    
    # allocate memory
    # define the number of features per channel
    NFPC = 7

    # Preallocate memory
    feat = np.zeros((Indx.shape[0] * NFPC + channels * NFPC))

    # Step1.1: Extract between-channels features
    ebp = getTDDfeatures_for_one_window(
        all_channels_data_in_window[:, Indx[:, 0]] - all_channels_data_in_window[:, Indx[:, 1]])
    efp = getTDDfeatures_for_one_window(
        np.log(
            (all_channels_data_in_window[:, Indx[:, 0]] - all_channels_data_in_window[:, Indx[:, 1]]) ** 2 + np.spacing(
                1)) ** 2)
    # Step 1.2: Correlation analysis
    num = np.multiply(efp, ebp)
    den = np.sqrt(np.multiply(efp, efp)) + np.sqrt(np.multiply(ebp, ebp))
    feat[range(Indx.shape[0] * NFPC)] = num / den

    # Step2.1: Extract within-channels features
    ebp = getTDDfeatures_for_one_window(all_channels_data_in_window)
    efp = getTDDfeatures_for_one_window(np.log((all_channels_data_in_window) ** 2 + np.spacing(1)) ** 2)
    # Step2.2: Correlation analysis
    num = np.multiply(efp, ebp)
    den = np.sqrt(np.multiply(efp, efp)) + np.sqrt(np.multiply(ebp, ebp))
    feat[np.max(range(Indx.shape[0] * NFPC)) + 1:] = num / den
    return feat

def format_examples(emg_examples, window_size=50, size_non_overlap=10):
    """ 
    Process EMG signals and then put into one window

    Args:
        emg_examples: list of emg signals, each row represent one recording of a 8 channel emg
        window_size: analysis window size
        size_non_overlap: length of non-overlap portion between each analysis window

    Returns:
        formated_examples: (252,) array including 7 features for each channel and for each two 
                            combination of channel signals
    """
    # Get the size of the input signal
    samples, channels = emg_examples.shape
    
    if channels > samples:
        emg_examples = np.transpose(emg_examples)
        samples, channels = channels, samples
        
    # Prepare indices of each 2 channels combination
    # NCC = Number of channels to combine
    NCC = 2
    Indx = np.array(list(combinations(range(channels), NCC)))   # (28,2)
    
    # allocate memory
    # define the number of features per channel
    NFPC = 7

    # Preallocate memory
    feat_num = Indx.shape[0] * NFPC + channels * NFPC
    print(f'Number of Features: {feat_num}')
    
    formated_examples = np.zeros((0, feat_num))
    win_num = 0
    examples = emg_examples
    while len(examples) >= window_size:
        window = examples[:window_size]
        # get TSD features
        if not np.sum(window) == 0:   # avoid all zero signals
            featured_example = getTSD(window)[:, np.newaxis] # (N,1)
            formated_examples = np.append(formated_examples, np.array(featured_example).transpose(), axis=0)
        else:
            formated_examples = np.append(formated_examples, np.zeros((252)))
        # Remove part of the data of the example according to the size_non_overlap variable    
        examples = examples[size_non_overlap:]
        win_num += 1
    print(f'Number of Computed Windows: {win_num}')
    return formated_examples