import sys
from glob import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import filtfilt, butter
from scipy import signal
from numpy import genfromtxt
from natsort import natsorted
import os
from shutil import copy
import csv
import datetime
import pytz
cwd = os.getcwd()

def loadTrueMetInterp(met_ss_interp):
    
    met_ss_interp_avg = np.mean(met_ss_interp,axis=0) # should be just size 5 for standing, walk, run
    met_ss_speeds = np.array([0.,1.,1.5,2.5,3.])#,3.5])
    #dt_met = 2*met_ss_interp_avg[-1] - met_ss_interp_avg[-2]
    #met_ss_interp_avg = np.append(met_ss_interp_avg, np.array(dt_met))
    
    tv1_sp = np.array([0.0, 0.0, 1.0, 1.0, 0.0])#[0.0, 1.0, 1.0, 0.0, 0.0])
    tv1_t = np.array([0., 4., 6., 16., 18.])#[0., 1., 12., 13., 24.])
    tv1_int = np.arange(0,24)
    tv1_true_sp = np.interp(tv1_int,tv1_t,tv1_sp)
    tv1_true_met = np.interp(tv1_true_sp, met_ss_speeds, met_ss_interp_avg)
    
    tv4_sp = np.array([1.0, 1.0, 3.0, 3.0, 1.0])#[1.0, 3.0, 3.0, 1.0, 1.0])
    tv4_t = np.array([0., 6., 10., 22., 26.])#[0., 4., 15., 19., 30.])
    tv_int = np.arange(0,30)
    tv4_true_sp = np.interp(tv_int,tv4_t,tv4_sp)
    tv4_true_met = np.interp(tv4_true_sp, met_ss_speeds, met_ss_interp_avg)  
    
    # add sine function for 2,3
    p = 30.0
    t = np.linspace(0,29,30)
    min_speed = 1.0
    max_speed = 1.5
    avg = (min_speed + max_speed)/2.0
    ampl = (max_speed - min_speed)/2.0
    tv_sine = avg + ampl*np.sin(2*3.14159*(t-p/4-6)/p)
    tv2_true_met = np.interp(tv_sine, met_ss_speeds, met_ss_interp_avg)
    
    min_speed = 1.0
    max_speed = 3.00
    avg = (min_speed + max_speed)/2.0
    ampl = (max_speed - min_speed)/2.0
    tv_sine = avg + ampl*np.sin(2*3.14159*(t-p/4-6)/p)
    tv3_true_met = np.interp(tv_sine, met_ss_speeds, met_ss_interp_avg)
    
    tv_mat = np.zeros((4,30))
    tv_mat[0,:24] = tv1_true_met
    tv_mat[1,:] = tv2_true_met
    tv_mat[3,:] = tv3_true_met
    tv_mat[2,:] = tv4_true_met
    
    return tv_mat

def loadRawMet(data_dir, subj, timezone):
    gen_files = os.listdir(data_dir+subj)
    for filename in gen_files:
        if len(filename) >= 4:
            if filename[-4:] == 'xlsx': # metabolics file
                met_array = pd.read_excel(data_dir+subj+'\\'+filename, names=["time","V02","VC02","HR"], skiprows=3, usecols=[9, 14, 15, 23]) # left then right insole forces   
                met_array['MET'] = (met_array["V02"]*16.48 + met_array["VC02"]*4.48)/60.
                dt = [i.hour*3600 + i.minute*60 + i.second for i in met_array["time"]]

                # load starting time
                met_raw = np.array(pd.read_excel(data_dir+subj+'\\'+filename, usecols=[4], header=None))
                starting_time = met_raw[0:2]
                test = str(starting_time[0][0]) + ' ' + str(starting_time[1][0])
                test_dt = datetime.datetime.strptime(test, '%m/%d/%Y %I:%M:%S %p')  
                test_dt = timezone.localize(test_dt)
                time_stamps = [test_dt + datetime.timedelta(seconds=i) for i in dt]
                met_array["time_stamp"] = time_stamps
                met_array["dt"] = np.array(dt) - dt[0] 
                
    return met_array, time_stamps

# gets all data between start and stop time 
def getCondData(input_data, time_stamps, start_time, cond_len, time_stamp_label='time_stamp', tz = "America/Los_Angeles"):
    stop_time = start_time + datetime.timedelta(seconds=cond_len)
    mask = (input_data[time_stamp_label] > start_time) & (input_data[time_stamp_label] < stop_time)
    try:
        last_true_ind = mask.index[mask][-1]
        mask[last_true_ind+1] = True
    except:
        print("error adding last index beyond the end of the file")
    input_subset = input_data.loc[mask]
    first_true_ind = mask.index[mask][0]
    input_subset["dt"] = np.array(input_subset["dt"]) - np.array(input_subset["dt"])[0] + (time_stamps[first_true_ind] - start_time).total_seconds()
    return input_subset

# input is list of times, iterate through and output a list of datetime objects
def convertAppleTime2DateTime(time_list):
    dt_list = []
    for i, time_str in enumerate(time_list):
        dt_str_split = time_str.split()
        dt1_date = datetime.datetime.strptime(dt_str_split[0]+" "+dt_str_split[1], "%Y-%m-%d %H:%M:%S")
        dt_list.append(dt1_date)
    return dt_list

# gets all data between start and stop time 
def getDataBetweenStartStop(input_data, start_time, stop_time, tz = "America/Los_Angeles"):
    mask = (input_data['startDate'] > start_time) & (input_data['endDate'] < stop_time)
    try:
        last_true_ind = mask.index[mask][-1]
        mask[last_true_ind+1] = True
    except:
        print("error adding last index beyond the end of the file")
    input_subset = input_data.loc[mask]
    input_subset['cumTime'] -= input_subset.iloc[0]['cumTime']
    return input_subset

# load the main data and pre-process times, and compute EE
def loadRawWatchData(watch_data_dir, file_type, kcal2watt, tz="America/Los_Angeles"):
    load_mat = pd.read_csv(watch_data_dir+file_type)
    load_mat['startDate'] = pd.to_datetime(load_mat['startDate'])
    load_mat['endDate'] = pd.to_datetime(load_mat['endDate'])
    load_mat['startDate'] = load_mat['startDate'].dt.tz_convert(tz)
    load_mat['endDate'] = load_mat['endDate'].dt.tz_convert(tz)
    # compute time interval and EE from these datetime vectors
    dt_vec_raw = load_mat['endDate'] - load_mat['startDate']
    dt_vec = [i.total_seconds() for i in dt_vec_raw]
    dt_fs_vec_raw = load_mat['startDate'] - load_mat.loc[0]['startDate']
    dt_cum = [i.total_seconds() for i in dt_fs_vec_raw]
    for i, ele in enumerate(dt_vec):
        if ele == 0: # if time stamps between energy expenditure output is 0, then make it 1 second
            dt_vec[i] = 1.0
    ee_vec = load_mat['value']*kcal2watt/(np.array(dt_vec)/3600.0)
    load_mat['time'] = dt_vec
    load_mat['cumTime'] = dt_cum
    load_mat['ee'] = ee_vec
    return load_mat

# take processed subsets of data and compute ee per second
# return array of [time (s), ee]
def interpolateWatchData(input_data, start_time, hr = False, cond_len_s=300, kcal2watt=1.16279):
    interp_vec = -1*np.ones(cond_len_s)
    skip_inds = []
    for i, ele in input_data.iterrows():
        if len(skip_inds) > 0:
            if len(skip_inds) == 1:
                skip_inds = []
            else:
                skip_inds = skip_inds[1:]
            continue
        dt = ele['startDate'] - start_time
        start_idx = dt.total_seconds()
        stop_idx = (ele['endDate'] - start_time).total_seconds()
        if start_idx == stop_idx:
            stop_idx += 1
        idx_rng = np.arange(np.maximum(int(start_idx),0), np.minimum(int(stop_idx), cond_len_s))
        # check if future indeces have the same value and lump them all together
        if (len(idx_rng) > 0) and not hr: # some data is within the range and compute EE
            new_ind_cnt = 1
            check_next_pt = True
            while(check_next_pt): # if next data point is in the 
                next_val_ind = i+new_ind_cnt
                if next_val_ind > input_data.last_valid_index():
                    break #break_flag = True
                else:
                    try:
                        next_val = input_data['value'].loc[next_val_ind]
                    except:
                        print('Past max "reachable" index... ', input_data.last_valid_index(), next_val_ind)
                        break
                if (ele['value'] == next_val): # they are the same
                    skip_inds.append(next_val_ind)
                    new_ind_cnt += 1
                else: # next index is different
                    check_next_pt = False
                    if len(skip_inds) > 0: # some data points to append to idx_rng
                        stop_idx = (input_data['endDate'].loc[next_val_ind-1] - start_time).total_seconds()
                        idx_rng = np.arange(np.maximum(int(start_idx),0), np.minimum(int(stop_idx), cond_len_s))
                        # compute new EE estimate rather than from each timestamp
                        new_dt = (input_data['endDate'].loc[next_val_ind-1] - ele['startDate']).total_seconds()
                        if new_dt == 0:
                            print("0 in the new_dt")
                            new_dt = 1
                        new_ee = ele['value']*kcal2watt*(new_ind_cnt)/(new_dt/3600.0)
                        interp_vec[idx_rng] = new_ee
                    else:
                        interp_vec[idx_rng] = ele['ee']
        elif hr:
            interp_vec[idx_rng] = ele['value']
        else:
            interp_vec[idx_rng] = ele['ee']
    # then correct for any missing seconds by applying the previous ee value that is not -1
    for i, ele in enumerate(interp_vec):
        if ele == -1:
            index = np.where(interp_vec != -1.0)
            if i == 0:
                if len(index[0]) == 0:
                    #print("No watch measurements found for this condition. Taking initial value.")
                    if hr:
                        interp_vec = input_data['value'].iloc[0]*np.ones(cond_len_s)
                    else:
                        interp_vec = input_data['ee'].iloc[0]*np.ones(cond_len_s)
                    break
                else:
                    interp_vec[i] = interp_vec[index[0][0]]
            else:
                interp_vec[i] = interp_vec[i-1]
    return interp_vec

# combine interpolated active and basal energy expenditure
def combineActiveBasal(active_e, passive_e, start_time, hr = False, cond_len_s = 300):
    active_interp = interpolateWatchData(active_e, start_time, hr, cond_len_s)
    passive_interp = interpolateWatchData(passive_e, start_time, hr, cond_len_s)
    return active_interp + passive_interp

def watchValidationPlot(ee_interp, hr_interp, ee_met, hr, dd):
    fig, ax1 = plt.subplots()
    plt.title('Apple watch data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Energy expenditure (W)')
    ax1.plot(np.arange(len(ee_interp)),ee_interp, ls = '--', color = 'k', label='Watch EE estimate')
    ax1.plot(np.arange(len(ee_met)), ee_met, color='k', alpha=0.7, label='Respirometry per breath')
    ax1.plot(np.arange(len(dd)), dd, color='b', label='Data driven')
    ax1.legend(loc='lower right')
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(np.arange(len(hr_interp)),hr_interp, color = color, ls = '--', label='Watch heart rate')
    ax2.plot(np.arange(len(hr)),hr, color = color, label='Heart rate monitor')
    ax2.set_ylabel('Heart rate (bpm)', color = color)
    ax2.tick_params(axis='y',labelcolor = color)
    ax2.legend(loc='lower left')
    fig.tight_layout()
    plt.show()

# takes in input data in [time steps x feats] and computes heel strikes over sliding window
# returns processed matrix of processed gait cycles, stacked [gaits x binned feats]
def simRealStrikes(input_data, weight, height, shift_ind, stride_detect_window, detect_window, peak_height_thresh, peak_min_dist, shank_gyro_z_ind, b, a, deg2rad, old_data = False, data_rate = 100.0):
    gait_data = []
    t_steps, feats = input_data.shape
    watching_heelstrike = True
    watch_strike_cnt = 0 
    time_list = []
    time_of_gait = []
    for k in range(t_steps - stride_detect_window):
        stride_window = input_data[k:k+stride_detect_window]
        if watching_heelstrike: # if looking for heelstrike
            peak_list = checkPeaks(stride_window[:,shank_gyro_z_ind], b, a, peak_height_thresh, peak_min_dist, deg2rad, old_data) # check for peaks
            if len(peak_list) > 1: # checking if a new heel strike has occured
                if (stride_detect_window - peak_list[-1]) < detect_window: # peak has occured in last detect_window of data
                    watching_heelstrike = False # now wait a bit to detect heelstrikes again
                    new_gait_cycle = processRawGait(stride_window, peak_list[-2], peak_list[-1], shift_ind, b, a, weight, height, deg2rad, old_data) # process most recent gait data
                    #time_stamp = round(time.time() - init_time,3) add later by sample cnt?
                    new_gait_cycle = np.expand_dims(new_gait_cycle, axis=0)
                    time_list.append(peak_list[-1] - peak_list[-2])
                    time_of_gait.append((k + peak_list[-1])/data_rate)
                    if len(gait_data) == 0:
                        gait_data = new_gait_cycle
                    else:
                        gait_data = np.concatenate((gait_data, new_gait_cycle), axis=0)
                    
        else: # count until the peak has cleared the recent window
            if watch_strike_cnt > detect_window:
                watching_heelstrike = True
                watch_strike_cnt = 0
            else:
                watch_strike_cnt += 1        
    return gait_data, time_list, time_of_gait

# for saved IMU data, make it into the same format as the real-time estimates
def computeEstimatesFromIMU(real_time_est, subj, cond, timezone, estimate_file_name, mass, height, shift_ind, stride_detect_window, detect_window, peak_height_thresh, peak_min_dist, shank_gyro_z_ind, b, a, deg2rad, model_weights, basal_rate, file_len_in_s = 5.0):
    subj_cond_dir = real_time_est + subj + '\\' + cond + '\\'
    basal_rate = round(float(basal_rate), 3)
    data_files = os.listdir(subj_cond_dir)
    data_files = natsorted(data_files)
    sample_data = np.array([])
    with open(subj_cond_dir+data_files[-1], 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            init_ts = datetime.datetime.strptime(row[0],"%d/%m/%Y %H:%M:%S")
            break
    f.close()
    init_ts = timezone.localize(init_ts)
    for l, file in enumerate(data_files[:-1]):
        if (file == estimate_file_name) or (file[0:2] == 'TV'): # skip this file
            continue
        else:
            file_data = np.load(subj_cond_dir+file)
            if len(sample_data) == 0:
                sample_data = file_data
            else:
                sample_data = np.concatenate((sample_data, file_data), axis=0)
    gait_cycles,_,time_of_gait = simRealStrikes(sample_data[:,:-1], mass, height, shift_ind, stride_detect_window, detect_window, peak_height_thresh, peak_min_dist, shank_gyro_z_ind, b, a, deg2rad) 
    num_gaits,_ = gait_cycles.shape
    gait_cycles = np.concatenate((np.ones((num_gaits,1)), gait_cycles), 1)
    estimates = np.round(np.dot(gait_cycles,model_weights),3)
    dt_stamps = [init_ts + datetime.timedelta(seconds=(i-file_len_in_s)) for i in time_of_gait]
    basal_t_thresh = 8.0
    offset_t = 1.0
    with open(subj_cond_dir+estimate_file_name,'w') as f:
        for p in range(num_gaits):
            if (p > 0) and ((dt_stamps[p]-dt_stamps[p-1]).total_seconds() >= basal_t_thresh): # large enough gap detected
                # adding first basal point after last gait
                f.write("{},{},{},{}".format(time_of_gait[p-1]+offset_t, dt_stamps[p-1] + datetime.timedelta(seconds=offset_t), basal_rate, basal_rate))
                f.write("\n")                
                f.write("{},{},{},{}".format(time_of_gait[p]-offset_t, dt_stamps[p]- datetime.timedelta(seconds=offset_t), basal_rate, basal_rate))
                f.write("\n")
            if estimates[p] < basal_rate:
                f.write("{},{},{},{}".format(time_of_gait[p], dt_stamps[p], basal_rate, basal_rate))
            else:
                f.write("{},{},{},{}".format(time_of_gait[p], dt_stamps[p], estimates[p], estimates[p]))
            f.write("\n")
    f.close()    
    
### Computing first order estimates for watch, hr, metabolics
def computeFirstOrderEstimates(t_mat, subjects, conditions, metabolics_real, met_mat, watch_int, hr_int, fo_time, watch_fo_int, hr_fo_int):
    # convert t_mat to all be in seconds
    for i in range(len(t_mat)):
        for j in range(len(t_mat[i])):
            offset = 3600*t_mat[i][j][0].hour + 60*t_mat[i][j][0].minute + t_mat[i][j][0].second
            for k in range(len(t_mat[i][j])):
                t_mat[i][j][k] = 3600*t_mat[i][j][k].hour + 60*t_mat[i][j][k].minute + t_mat[i][j][k].second - offset + 1

    conditions = conditions[1:]
    # compute first order estimates of met, hr, watch
    for i, subj in enumerate(subjects):
        for j, cond in enumerate(conditions):
            # for watch add pt whenever there is a change in the estimate
            watch_est_list = [watch_int[i,j,0]]
            tvec = [1]
            for k in range(1,fo_time+2):
                if watch_int[i,j,k] != watch_est_list[-1]: # next point different
                    watch_est_list.append(watch_int[i,j,k])
                    tvec.append(k+1)
                if k >= 2:
                    watch_fo_int[i,j,k-2],_,_ = metabolic_rate_estimation(tvec, watch_est_list)
            # for hr add a pt for each interp value
            for k in range(2,fo_time+2):
                tvec = np.arange(1,k+1)
                hr_fo_int[i,j,k-2],_,_ = metabolic_rate_estimation(tvec, hr_int[i,j,:k])
    # for met do it like previous method for each breath
    tmin = 2
    tmax = 121
    t = np.arange(tmin,tmax)
    met_fo_int = np.zeros(len(t))
    met_ordering = np.zeros(len(t))
    met_est_full = np.zeros((len(subjects),len(conditions),tmax-tmin))
    for z,time_horizon in enumerate(t):
        metabolics_est = np.zeros((len(subjects),len(conditions)))
        for i in range(len(t_mat)):
            for j in range(1,len(t_mat[i])):
                end_ind = 0
                for k in range(len(t_mat[i][j])):
                    if t_mat[i][j][k] >= time_horizon: # save index and break
                        end_ind = k
                        break
                metabolics_est[i,j-1], y_bar, mean_squared_err = metabolic_rate_estimation(t_mat[i][j][:end_ind], met_mat[i][j][:end_ind])
        met_est_full[:,:,z] = metabolics_est
        met_fo_int[z], met_ordering[z] = compute_2min_met_errors(metabolics_real, metabolics_est)

    return hr_fo_int, met_fo_int, watch_fo_int, met_est_full
    
# compute data-driven estimates interpolated at 1-second intervals
def computeDDinter(real_time_est, est_col_ind, subj, cond, subj_cond_dir, estimate_file_name, timezone, loc_cond_timestamp, cond_time_s, basal_flag=False, basal_rate = 0.0):
    basal_t_thresh = 8.0 # number of second gap between gait cycles to estimate adjusted standing rate
    subjcond_time = []
    subjcond_dt = []
    subjcond_est = []
    with open(subj_cond_dir+estimate_file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row[1]) == 25:
                new_dt = timezone.localize(datetime.datetime.strptime(row[1][:-6],"%Y-%m-%d %H:%M:%S")) #.%f-%Z"))
            elif len(row[1]) == 26:
                #print(len(row[1]), row[1][:-7])
                new_dt = timezone.localize(datetime.datetime.strptime(row[1][:-7],"%Y-%m-%d %H:%M:%S")) #.%f-%Z"))
            else:
                new_dt = timezone.localize(datetime.datetime.strptime(row[1][:-13],"%Y-%m-%d %H:%M:%S")) #.%f-%Z"))
            subjcond_time.append(float(row[0]))
            subjcond_dt.append(new_dt)
            subjcond_est.append(float(row[est_col_ind]))
    f.close()
    # compute time difference between start datetime and first gait cycle
    time_diff = (loc_cond_timestamp - subjcond_dt[0]).total_seconds()
    # from time vector subjtract initial gait value and time difference
    subjcond_time = np.array(subjcond_time)
    subjcond_est = np.array(subjcond_est)
    subjcond_time = subjcond_time - (time_diff + subjcond_time[0])
    # find indeces of gaits occuring after the start of the condition (time == 0) and less than end of condition (time == 300)
    prev_gaits = subjcond_time >= 0.0
    subjcond_time = subjcond_time[prev_gaits]
    subjcond_est = subjcond_est[prev_gaits]
    prev_gaits2 = subjcond_time < cond_time_s
    subjcond_time = subjcond_time[prev_gaits2]
    subjcond_est = subjcond_est[prev_gaits2]
    tmin = 0
    tmax = cond_time_s
    t = np.arange(tmin,tmax)
    dd_int = np.interp(t, subjcond_time, subjcond_est)
    return dd_int, subjcond_est, subjcond_time
    
# take in shank IMU vector, filter, look for thresholding & min_distance, return strike indeces
def checkPeaks(strike_vec, b, a, peak_height_thresh, peak_min_dist, deg2rad, old_data = False):
    if old_data:
        strike_vec_filt = signal.filtfilt(b,a,strike_vec*(-1.0/deg2rad))
    else:
        strike_vec_filt = signal.filtfilt(b,a,strike_vec)
    peak_list = signal.find_peaks(strike_vec_filt, height=peak_height_thresh, distance=peak_min_dist)
    return peak_list[0]
    
# takes data and moves the first half of the data to the second half
# assumes data_array is size [time steps x feature]
def convertBetweenLegs(data_array):
    array_copy = np.copy(data_array)
    convert_array = np.zeros(data_array.shape)
    gait_time_steps,_ = data_array.shape
    half_gait = int(gait_time_steps/2)
    convert_array[:half_gait,:] = array_copy[half_gait:,:]
    convert_array[half_gait:,:] = array_copy[:half_gait,:]
    return convert_array
    
# downsample the data into a discrete number of bins
def binData(data_array, num_bins=30):
    return signal.resample(data_array, num_bins) # resamples along axis = 0 by default

# shift the data in the binned matrix to the left by the given shift_ind
def shiftBinData(data_array, shift_ind, num_bins=30):
    array_copy = np.copy(data_array)
    convert_array = np.zeros(data_array.shape)
    convert_array[:-shift_ind, :] = array_copy[shift_ind:,:]
    convert_array[-shift_ind:, :] = array_copy[:shift_ind,:]
    return convert_array

# pass in array of data to process [time_samples x num_features] into [num_bins x num_features]
# start_ind, end_ind are indeces of start/end of gait cycle
# shift_ind is correction term for timing diff between heel strike and IMU thresh
# b, a are filter parameters
def processRawGait(data_array, start_ind, end_ind, shift_ind, b, a, weight, height, deg2rad, old_data = False, num_bins=30):
    gait_data = data_array[start_ind:end_ind, :] # crop to the gait cycle
    if not old_data:
        gait_data = gait_data*np.array([deg2rad,-deg2rad,-deg2rad,deg2rad,-deg2rad,-deg2rad,1,-1,-1,1,-1,-1]) # flip y & z, convert to rad/s
    filt_gait_data = signal.filtfilt(b,a,gait_data, axis=0) # low-pass filter
    bin_gait = binData(filt_gait_data) # discretize data into bins
    shift_flip_bin_gait = bin_gait.transpose() # get in shape of [feats x bins] for correct flattening
    model_input = shift_flip_bin_gait.flatten()
    model_input = np.insert(model_input, 0, [weight, height]) # adding a 1 for the bias term at start
    return model_input

# vec1 here is true vec, vec 2 is the estimate
def pairwise_similarity(vec1, vec2, threshold = 0.042):
    # figure out how many pairs without repeats
    pairwise_len = 0
    for i in range(0,len(vec1)-1):
        pairwise_len += (i+1)
    # initialize pairways datastructures    
    pairwise_est = np.zeros(pairwise_len)
    pairwise_real = np.zeros(pairwise_len)
    # go through pairs to determine if the first entry is less than (0), within the threshold (1), or greater than (2)
    # the second entry. 
    counter = 0
    for i in range(len(vec1)-1):
        for j in range(i+1, len(vec1)):
            if mapd(vec1[i],vec1[j]) <= threshold: # they are equal, set element to 1
                pairwise_real[counter] = 1
            elif vec1[i] < vec1[j]:
                pairwise_real[counter] = 0
            else:
                pairwise_real[counter] = 2
                
            if mapd(vec2[i],vec2[j]) <= threshold: # they are equal, set element to 1
                pairwise_est[counter] = 1
            elif vec2[i] < vec2[j]:
                pairwise_est[counter] = 0
            else:
                pairwise_est[counter] = 2
                
            counter += 1
    # sum the number of same elements between real and est
    match = np.equal(pairwise_est, pairwise_real)
    correct = sum(match)
    return correct, correct, pairwise_len

def mapd(vec1, vec2): # mean absolute percent difference
    return np.abs(vec1 - vec2) / ((vec1 + vec2)/2)

def compute_2min_met_errors(metabolics_real, met_2min_est):
    mae_mat = (np.abs(metabolics_real-met_2min_est)/metabolics_real)
    met_mape = np.mean(mae_mat)
    subjs_acc = np.zeros(metabolics_real.shape[0])
    for i in range(metabolics_real.shape[0]): # loop thru subjects
        correct, correct, pairwise_len = pairwise_similarity(metabolics_real[i,:], met_2min_est[i,:])
        subjs_acc[i] = correct/pairwise_len        
    ordering = np.mean(subjs_acc)
    return met_mape, ordering

def compute_2min_met_errors_subj(metabolics_real, met_2min_est):
    mae_mat = (np.abs(metabolics_real-met_2min_est)/metabolics_real)
    met_mape = np.mean(mae_mat)
    subjs_acc = np.zeros(metabolics_real.shape[0])
    for i in range(metabolics_real.shape[0]): # loop thru subjects
        correct, correct, pairwise_len = pairwise_similarity(metabolics_real[i,:], met_2min_est[i,:])
        subjs_acc[i] = correct/pairwise_len        
        
    ordering = np.mean(subjs_acc)
    return met_mape, ordering, subjs_acc 

def metabolic_rate_estimation(t, y_meas, tau=42):
    n_samp = len(t)
    A = np.zeros((n_samp,2))
    A[0,:] = [1,0]
    for i in range(1,n_samp):
        for j in range(2):
            dt = t[i] - t[i-1]
            if j == 0:
                A[i,j] = A[i-1,j]*(1-dt/tau)
            else:
                A[i,j] = A[i-1,j]*(1-dt/tau) + (dt/tau)
    x_star = np.dot(np.linalg.pinv(A),y_meas)
    y_bar = np.dot(A,x_star)
    mean_squared_err = np.dot(np.transpose(y_bar-y_meas),(y_bar-y_meas))/n_samp
    met_est = x_star[1]
    
    return met_est, y_bar, mean_squared_err

def print_error_maps(debug_values):
    num_plots, num_subj, num_cond = debug_values.shape
    title_list = ['corr_max_angles', 'corr_diff_angles', 'mape_angles', 'corr_max_force', 'corr_diff_force', 'mape_force', 'zero_counter']
    for i in range(num_plots):
        plt.figure()
        plt.matshow(debug_values[i,:,:])
        plt.xlabel('Conditions')
        plt.ylabel('Subjects')
        plt.title(title_list[i])
        plt.colorbar()
        plt.show()
    # printing final values
    print("Average MAPE for forces: ", np.mean(np.mean(debug_values[5,:,:])))
    print("Average MAPE across angles: ", np.mean(np.mean(debug_values[2,:,:])))

# a is the longer time series on both ends
# b is shorter on both sides
# returns a,b vectors overlapping and corr_ind of a
def trim_corr(a,b):
    corr = np.correlate(a,b,'valid')
    corr_ind = np.argmax(corr)
    new_a = a[corr_ind:(len(b)+corr_ind)]
    return new_a, b, corr_ind
    
# a is the longer time series on both ends
# b is shorter on both sides
# returns a,b vectors overlapping and corr_ind of a
def trim_corr_arrays(a,b,a_ind,b_ind,mean=True):
    if mean:
        corr = np.correlate(a[:,a_ind]-np.mean(a[:,a_ind]),b[:,b_ind]-np.mean(b[:,b_ind]),'valid')
    else:
        corr = np.correlate(a[:,a_ind],b[:,b_ind],'valid')
    corr_ind = np.argmax(corr)
    new_a = a[corr_ind:(len(b)+corr_ind),:]
    range_new_a = float(np.abs(np.max(new_a[:,a_ind]) - np.min(new_a[:,a_ind]))/2.0)
    corr_max = max(corr)/(1.0*len(new_a)*range_new_a**2)
    return new_a, b, corr_ind, corr_max, corr

# pull metabolics from the metabolics .csv file and the subject height/weight from the other .csv
def calc_metabolics(data_dir, subj, visualize = False, add_heartrate = False, cond_len=14, len_cond_s = 300, end_time_len = 180):
    gen_files = os.listdir(data_dir+subj)
    t_mat = []
    met_mat = []
    hr_mat = []
    for filename in gen_files:
        if len(filename) >= 4:
            if filename[-4:] == 'xlsx': # metabolics file
                if add_heartrate:
                    met_array = np.array(pd.read_excel(data_dir+subj+'\\'+filename, skiprows=3, usecols=[9, 14, 15, 23, 37])) # left then right insole forces          
                else:
                    met_array = np.array(pd.read_excel(data_dir+subj+'\\'+filename, skiprows=3, usecols=[9, 14, 15, 37])) # left then right insole forces        
                # load starting time
                met_raw = np.array(pd.read_excel(data_dir+subj+'\\'+filename, usecols=[4], header=None))
                starting_time = met_raw[0:2]
                test = str(starting_time[0][0]) + ' ' + str(starting_time[1][0])
                test_dt = datetime.datetime.strptime(test, '%m/%d/%Y %I:%M:%S %p')    
                met_len, cols = met_array.shape
                cond_indeces = list(np.arange(1,cond_len+1)) #[1,2,3,4,5,6,7,8,9,10,11,12]
                start_inds = ['start 01','start 02','start 03','start 04','start 05','start 06','start 07','start 08','start 09','start 10','start 11','start 12','start 13', 'start 14', 'start 15', 'start 16']
                start_inds = start_inds[:cond_len]
                hr_int_mat = np.zeros((cond_len, len_cond_s))
                met_int_mat = np.zeros((cond_len, len_cond_s))
                start_stamps = []
                met_inds = []
                met_start_inds = np.zeros(cond_len, dtype=int)
                hr = np.zeros(cond_len) # first value in each col is length to use of this placeholder vector
                met_2mins = np.zeros(cond_len)
                met_vals = np.zeros((cond_len,1))
                for cnd in range(cond_len): # adding to check for conditions out of order
                    for i in range(met_len):
                        if met_array[i,-1] == cond_indeces[cnd]:#in cond_indeces:
                            met_inds.append(i)
                        if met_array[i,-1] == start_inds[cnd]:#in start_inds:
                            start_ind = start_inds.index(met_array[i,-1])
                            met_start_inds[start_ind] = i
                        
                # take the average of vco2 and vo2 over the num of avg_breaths, then find W from met eq
                for i, ind in enumerate(met_inds):
                    end_time = met_array[ind,0]
                    start_ind = ind
                    start_time = end_time
                    #while (start_time.minute != end_time.minute-1) or (start_time.second > end_time.second): # want 1 minute less than end_time
                    while (start_time.hour*3600 + start_time.minute*60 + start_time.second + end_time_len) > (end_time.hour*3600 + end_time.minute*60 + end_time.second):
                        start_ind -= 1 # decrement
                        start_time = met_array[start_ind,0]                    
                    vo2 = np.mean(met_array[start_ind:ind,1])
                    vco2 = np.mean(met_array[start_ind:ind,2])
                    met_vals[i] = (vo2*16.48 + vco2*4.48)/60.0 # BROCKWAY
                    
                    if add_heartrate: # store values in the hr
                        hr[i] = np.mean(met_array[start_ind:ind,-2])
                        
                for i, ind in enumerate(met_start_inds):
                    start_time = met_array[ind,0]
                    end_time = start_time
                    end_ind = ind
                    while((start_time.hour*3600 + start_time.minute*60 + start_time.second + 120) > (end_time.hour*3600 + end_time.minute*60 + end_time.second)): # til end _time is 2 mins more than start
                        end_ind += 1 # increment
                        end_time = met_array[end_ind,0]
                    # get the c02, v02 values for those times, compute met
                    vo2 = met_array[ind:end_ind,1]
                    vco2 = met_array[ind:end_ind,2]
                    met_vec = (vo2*16.48 + vco2*4.48)/60.0 # BROCKWAY
                    tvec = np.zeros(end_ind-ind)
                    for cnt,j in enumerate(range(ind,end_ind)):
                        time_stamp = met_array[j,0]
                        tvec[cnt] = time_stamp.hour*3600 + time_stamp.minute*60 + time_stamp.second
                    tvec = tvec - tvec[0] + 1 # offset the time so starts at 1
                    met_est, y_bar, mean_squared_err = metabolic_rate_estimation(tvec, met_vec)
                    met_2mins[i] = met_est
                    
                    # computing the vectors for each 6 minute interval of interpolated estimates
                    end_time2 = start_time
                    dt_s = start_time.hour*3600 + start_time.minute*60 + start_time.second
                    start_stamps.append(test_dt + datetime.timedelta(0, dt_s)) # adding time to datetime starting time
                    end_ind2 = ind
                    while((start_time.hour*3600 + start_time.minute*60 + start_time.second + len_cond_s) > (end_time2.hour*3600 + end_time2.minute*60 + end_time2.second)): # til end _time is X mins more than start
                        end_ind2 += 1 # increment
                        end_time2 = met_array[end_ind2,0]
                    # get the c02, v02 values for those times, compute met
                    vo2 = met_array[ind:end_ind2,1]
                    vco2 = met_array[ind:end_ind2,2]
                    hr_vec5 = met_array[ind:end_ind2,3]
                    met_vec5 = (vo2*16.48 + vco2*4.48)/60.0 # BROCKWAY
                    tvec5 = np.zeros(end_ind2-ind)
                    for cnt,j in enumerate(range(ind,end_ind2)):
                        time_stamp = met_array[j,0]
                        tvec5[cnt] = time_stamp.hour*3600 + time_stamp.minute*60 + time_stamp.second
                    tvec5 = tvec5 - tvec5[0] + 1 # offset the time so starts at 1
                    tvec5_int = np.arange(1,len_cond_s+1)
                    met_vec5_int = np.interp(np.array(tvec5_int,dtype='float64'), np.array(tvec5,dtype='float64'),np.array(met_vec5,dtype='float64'))
                    hr_vec5_int = np.interp(np.array(tvec5_int,dtype='float64'), np.array(tvec5,dtype='float64'),np.array(hr_vec5,dtype='float64'))
                    hr_int_mat[i,:] = hr_vec5_int
                    met_int_mat[i,:] = met_vec5_int
                    
                for i,ind in enumerate(met_start_inds):
                    stop_ind = met_inds[i]
                    t_mat.append(met_array[ind:stop_ind,0])
                    met_mat_temp = (met_array[ind:stop_ind,1]*16.48 + met_array[ind:stop_ind,2]*4.48)/60.0
                    met_mat.append(met_mat_temp)
                    hr_mat.append(met_array[ind:stop_ind,-2])
    if visualize:
        print('Metabolic values (Watts):', met_vals)
        print('2 min. Met values: ',met_2mins)
        if add_heartrate:
            print('HR: ', hr)
    return met_vals, hr, met_2mins, hr_int_mat, met_int_mat, start_stamps, t_mat, met_mat, hr_mat

def load_constants(cur_dir, subjects):
    code_files = os.listdir(cur_dir)
    for fnm in code_files:
        if len(fnm) >= 6:
            if fnm[-6:] == 'rt.csv':
                subj_data = np.array(pd.read_csv(cur_dir+'\\'+fnm, sep=",", skiprows=0, usecols=[1,2]))
                num_subj, cols = subj_data.shape
                if len(subjects) <= num_subj:
                    masses = subj_data[:len(subjects),1]
                    heights = subj_data[:len(subjects),0]
                    return masses, heights
                else:
                    print("Trying to pull more subjects of data from code/subjects.csv than are there...")
                    return
    print("No subjects.cvs file in the code folder...")
    
# split the data into a subset of features based on text input and a list of features    
def loadSignals(signals, data_list, num_bins = 30):
    labels = data_list[0]
    signal_list = signals.split() # create individual list of features to include
    num_constants = data_list[6]
    if "time" in labels: # weird convention I'm using...
        num_constants += 1

    signal_ind = [] # store indeces of signals here
    new_labels = [] # store included signal labels here
    for signal in signal_list:
        for i, label in enumerate(labels): # go through labels looking for partial matches
            if len(signal) <= len(label): # can check for subset without error
                if signal == label[:len(signal)]: # includes subset and add indeces
                    new_labels.append(label)
                    if i < num_constants: # include column number
                        signal_ind.append(i)
                    else:
                        start_ind = num_constants + (i - num_constants) * num_bins
                        end_ind = num_constants + (i - num_constants + 1) * num_bins
                        for ind in range(start_ind, end_ind):
                            signal_ind.append(ind)
    signal_index_final = list(signal_ind)

    return signal_index_final, new_labels

def readDataCSV(csvfile):
    holdout_cond_raw = [] # init in case not defined
    with open(csvfile, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            if i == 0:
                features_raw = line
            elif i == 1:
                seed_list_raw = line
            elif i == 2:
                sym_list_raw = line
            elif i == 3:
                new_dir_raw = line
            elif i == 4:
                subjs_raw = line
            elif i == 5:
                conds_raw = line
            elif i == 6:
                const_raw = line
            elif i == 7:
                norm_raw = line
            elif i == 8:
                dataset_raw = line
            elif i == 9:
                freq_raw = line
            elif i == 10:
                subj_raw = line
            elif i == 11:
                cond_raw = line
            elif i == 12:
                holdout_cond_raw = line
                
    features = features_raw[0].split(',')
    seed_list = seed_list_raw[0].split(',')
    sym_list = sym_list_raw[0].split(',')
    new_dir_temp = new_dir_raw[0].split(',')
    new_dir = new_dir_temp[1]
    subjs_temp = subjs_raw[0].split(',')
    subjs = int(subjs_temp[1])
    conds_temp = conds_raw[0].split(',')
    conds = int(conds_temp[1])
    const_temp = const_raw[0].split(',')
    const = int(const_temp[1])
    norm_temp = norm_raw[0].split(',')
    norm = int(norm_temp[1])
    dataset_temp = dataset_raw[0].split(',')
    dataset = dataset_temp[0]
    freq = freq_raw[0].split(',')
    subj_list = subj_raw[0].split(',')
    cond_list = cond_raw[0].split(',')
    try:
        if holdout_cond_raw != []:
            holdout_cond = holdout_cond_raw[0].split(',')
            while '' in holdout_cond:
                holdout_cond.remove('')
        else:
            holdout_cond = holdout_cond_raw
    except:
        holdout_cond = []

    while '' in features:
        features.remove('')
    while '' in seed_list:
        seed_list.remove('')
    while '' in sym_list:
        sym_list.remove('')
    while '' in freq:
        freq.remove('')
    while '' in subj_list:
        subj_list.remove('')
    while '' in cond_list:
        cond_list.remove('')
    
    data_list = [features, seed_list, sym_list, new_dir, subjs, conds, const, norm, dataset, freq, subj_list, cond_list, holdout_cond]

    return data_list
    
def loadFiles(path, avg_num_steps=1):
    os.chdir(path)
    x_data = genfromtxt('x.csv', delimiter=',')
    y_data = genfromtxt('y.csv', delimiter=',')
    #print(y_data, type(y_data))
    if y_data.size != 1: # added this line in!!!
        y_data = np.reshape(y_data,(y_data.shape[0],-1)) #(num,) -> (num,1) so vstack doesn't throw error
    if avg_num_steps != 1: # average N steps and chop off remainder
        x_data = groupedAvg(x_data,avg_num_steps)
        y_data = groupedAvg(y_data,avg_num_steps)
    return x_data, y_data

# downsample or upsample (by randomly redrawing samples) number of gaits from each condition until all the same
def fixSamples(x, number_gaits, first_cycles):
    num_cycles = x.shape[0]
    if num_cycles == number_gaits:
        return x
    elif num_cycles < number_gaits: # randomly sample from gaits
        num_needed = number_gaits - num_cycles
        s_range = range(0,num_cycles)
        sample_inds = np.random.choice(s_range, num_needed, True)
        for ind in sample_inds:
            x = np.append(x, [x[ind,:]], axis=0)
        
    else: # take the last X number of gaits
        if first_cycles:
            x = x[:number_gaits,:]
        else:
            x = x[num_cycles-number_gaits:,:]
    return x

# find std across all bins of all relevant params and eliminate cycles that don't lie within threshold params            
def elimOutliers(x, run_data, labels, outlier_std, num_pts_outside, outlier_feat_list, num_constants, num_bins=30):
    if "time" in labels:
        num_constants += 1 # skip one more space to not filter time
        
    cycles = x.shape[0]
    cycles_to_elim = [] # list of indeces to elim
    
    for i, param in enumerate(outlier_feat_list):
        if param in labels:
            ind = labels.index(param) - num_constants
            start = num_constants+ind*num_bins
            stop = start+num_bins
            std_profile = np.std(x[:,start:stop], axis=0) # std across all bins
            mean_profile = np.mean(x[:,start:stop], axis=0)
            
            for j in range(cycles): # go through each gait cycle and check if within threshold
                below_thresh = (x[j,start:stop] <= (mean_profile - outlier_std*std_profile)) 
                above_thresh = (x[j,start:stop] >= (mean_profile + outlier_std*std_profile))
                #print(below_thresh)
                num_outside_thresh = sum(below_thresh) + sum(above_thresh)
    # elim these number cycles
    cycles_to_elim.sort()
    x = np.delete(x, cycles_to_elim, 0)
                  
    return x

# find closest X number of gait cycles to average measurement for gyro_shank_z_L data
def simpleElimOutliers(x, gait_times, num_gait_to_keep, time_dif = 0.7):
    cycles_to_elim = []
    mean_time = np.mean(gait_times)
    for i, time in enumerate(gait_times):
        if (time < time_dif*mean_time) or (time > (1 + (1-time_dif))*mean_time):
            cycles_to_elim.append(i)
    cycles_to_elim.sort()
    x = np.delete(x, cycles_to_elim, 0)  
    if len(cycles_to_elim) > 0:
        print("Removed ", len(cycles_to_elim)," gait cycles for being too short or long.")
    
    num_gaits,_ = x.shape
    x_feat = x[:,62:92]
    x_feat_mean = np.mean(x_feat, axis=0)
    x_sum_abs_diff = np.sum(abs(x_feat - np.expand_dims(x_feat_mean,axis=0)), axis=1)
    if num_gaits > num_gait_to_keep:
        x_sum_abs_diff.sort()
        x_thresh = x_sum_abs_diff[num_gait_to_keep-1]
        cycles_to_elim = []
        for i in range(num_gaits):
            if x_sum_abs_diff[i] > x_thresh:
                cycles_to_elim.append(i)

        cycles_to_elim.sort()
        x = np.delete(x, cycles_to_elim, 0)
        return x
    else:
        print("Not enough gait cycles to remove any...")
        return x

# input data is (examples x features*bins), plotting multiple binned examples
# not for plotting the constants features
def plotBinDataRetVec(labels_to_plot, start_ex, num_examples, data, labels, num_bins, num_constants):
    if "time" in labels:
        num_constants += 1 # skip one more space to not filter time
    for label in labels_to_plot:
        for example in range(num_examples):
            ind = labels.index(label)
            bin_ind = (ind - num_constants)*num_bins + num_constants
            plt.plot(data[start_ex + example, bin_ind:bin_ind+num_bins])
    plt.xlabel('Percent of gait cycle')
    plt.ylabel('Feature value')
    plt.xticks([0, 14, 29], ['0','50','100'])
    plt.show()
    return np.mean(data[start_ex:start_ex+num_examples, bin_ind:bin_ind+num_bins],axis=0)