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
    num_subjs = met_ss_interp.shape[0]
    tv_mat = np.zeros((num_subjs, 4,30))
    del_list = []
    # make a list of each condition with non-zero vals and append them all then take the mean of that
    
    for i in range(num_subjs):
        met_ss_interp_avg = met_ss_interp[i,:]#np.mean(met_ss_interp,axis=0) # should be just size 5 for standing, walk, run
        if met_ss_interp[i,-1] != 0.0:
            del_list.append(i)
        met_ss_speeds = np.array([0.,1.,1.5,2.5,3.])
        tv1_sp = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        tv1_t = np.array([0., 4., 6., 16., 18.])
        tv1_int = np.arange(0,24)
        tv1_true_sp = np.interp(tv1_int,tv1_t,tv1_sp)
        tv1_true_met = np.interp(tv1_true_sp, met_ss_speeds, met_ss_interp_avg)

        tv4_sp = np.array([1.0, 1.0, 3.0, 3.0, 1.0])
        tv4_t = np.array([0., 6., 10., 22., 26.])
        tv_int = np.arange(0,30)
        tv4_true_sp = np.interp(tv_int,tv4_t,tv4_sp)
        tv4_true_met = np.interp(tv4_true_sp, met_ss_speeds, met_ss_interp_avg)  

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

        tv_mat[i,0,:24] = tv1_true_met
        tv_mat[i,1,:] = tv2_true_met
        tv_mat[i,3,:] = tv3_true_met
        tv_mat[i,2,:] = tv4_true_met
    
    tv_avg_mat = np.zeros((4,30))
    true_inds = np.zeros((4, num_subjs))
    for j in range(4):
        if j >= 2:
            tv_avg_mat[j,:] = np.mean(tv_mat[del_list,j,:],axis=0)
            true_inds[j,del_list] = 1

        else:
            tv_avg_mat[j,:] = np.mean(tv_mat[:,j,:], axis=0)
            true_inds[j,:] = 1
            
    return tv_mat, tv_avg_mat, true_inds

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
        print("Error adding last index beyond the end of the file")
    input_subset = input_data.loc[mask]
    first_true_ind = mask.index[mask][0]
    input_subset["dt"] = np.array(input_subset["dt"]) - np.array(input_subset["dt"])[0] + (time_stamps[first_true_ind] - start_time).total_seconds()
    return input_subset


def watchValidationPlot(ee_interp, hr_interp, ee_met, hr, dd):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Energy expenditure (W)')
    ax1.plot(np.arange(len(ee_met)), ee_met, color='k', alpha=0.7, label='Respirometry per breath')
    ax1.plot(np.arange(len(dd)), dd, color='b', label='Data driven')
    ax1.legend(loc='lower right')
    fig.tight_layout()
    plt.show()

# takes in input data in [time steps x feats] and computes heel strikes over sliding window
# returns processed matrix of processed gait cycles, stacked [gaits x binned feats]
def simRealStrikes(input_data, weight, height, shift_ind, stride_detect_window, detect_window, peak_height_thresh, peak_min_dist, shank_gyro_z_ind, b, a, deg2rad, old_data = False, data_rate = 100.0, addTime = False):
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
                    new_gait_cycle = processRawGait(stride_window, peak_list[-2], peak_list[-1], shift_ind, b, a, weight, height, deg2rad, old_data, data_rate, addTime) # process most recent gait data
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
def computeEstimatesFromIMU(real_time_est, subj, cond, timezone, estimate_file_name, mass, height, shift_ind, stride_detect_window, detect_window, peak_height_thresh, peak_min_dist, shank_gyro_z_ind, b, a, deg2rad, model_weights, basal_rate, file_len_in_s = 5.0, save_gait_data=False, met_val = 0.0, overwrite = False):
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
            try:
                file_data = np.load(subj_cond_dir+file)
                if len(sample_data) == 0:
                    sample_data = file_data
                else:
                    sample_data = np.concatenate((sample_data, file_data), axis=0)
            except:
                pass
    gait_cycles,_,time_of_gait = simRealStrikes(sample_data[:,:-1], mass, height, shift_ind, stride_detect_window, detect_window, peak_height_thresh, peak_min_dist, shank_gyro_z_ind, b, a, deg2rad) 
    num_gaits,_ = gait_cycles.shape
    gait_cycles = np.concatenate((np.ones((num_gaits,1)), gait_cycles), 1)
    if save_gait_data:
        save_dir = 'C:\\Users\\patty\\Desktop\\EEE\\RealTimeTesting\\results\\data\\' + subj + '\\' + cond + '\\'
        try:
            os.makedirs(save_dir)
        except:
            pass
        np.savetxt(save_dir+'x.csv', gait_cycles, delimiter=',')
        np.savetxt(save_dir+'y.csv', np.ones((num_gaits,1))*met_val, delimiter=',')
        
    estimates = np.round(np.dot(gait_cycles,model_weights),3)
    dt_stamps = [init_ts + datetime.timedelta(seconds=(i-file_len_in_s)) for i in time_of_gait]
    basal_t_thresh = 8.0
    offset_t = 1.0
    if overwrite:
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

# compute data-driven estimates interpolated at 1-second intervals
def computeDDinter(real_time_est, est_col_ind, subj, cond, subj_cond_dir, estimate_file_name, timezone, loc_cond_timestamp, cond_time_s, basal_flag=False, basal_rate = 0.0):
    basal_t_thresh = 8.0 # number of second gap between gait cycles to estimate adjusted standing rate
    subjcond_time = []
    subjcond_dt = []
    subjcond_est = []
    with open(subj_cond_dir+estimate_file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        try:
            for row in reader:
                if len(row[1]) == 25:
                    new_dt = timezone.localize(datetime.datetime.strptime(row[1][:-6],"%Y-%m-%d %H:%M:%S"))
                elif len(row[1]) == 26:
                    new_dt = timezone.localize(datetime.datetime.strptime(row[1][:-7],"%Y-%m-%d %H:%M:%S"))
                else:
                    new_dt = timezone.localize(datetime.datetime.strptime(row[1][:-13],"%Y-%m-%d %H:%M:%S"))
 
                subjcond_time.append(float(row[0]))
                subjcond_dt.append(new_dt)
                subjcond_est.append(float(row[est_col_ind]))
        except:
            pass
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
def processRawGait(data_array, start_ind, end_ind, shift_ind, b, a, weight, height, deg2rad, old_data = False, num_bins=30, addTime = False):
    gait_data = data_array[start_ind:end_ind, :] # crop to the gait cycle
    if not old_data:
        gait_data = gait_data*np.array([deg2rad,-deg2rad,-deg2rad,deg2rad,-deg2rad,-deg2rad,1,-1,-1,1,-1,-1]) # flip y & z, convert to rad/s
    filt_gait_data = signal.filtfilt(b,a,gait_data, axis=0) # low-pass filter
    bin_gait = binData(filt_gait_data) # discretize data into bins
    shift_flip_bin_gait = bin_gait.transpose() # get in shape of [feats x bins] for correct flattening
    model_input = shift_flip_bin_gait.flatten()
    if addTime:
        model_input = np.insert(model_input, 0, [weight, height, (end_ind-start_ind)*0.01]) # adding a 1 for the bias term at start
    else:
        model_input = np.insert(model_input, 0, [weight, height]) # adding a 1 for the bias term at start
    return model_input


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

# pull metabolics from the metabolics .csv file and the subject height/weight from the other .csv
def calc_metabolics2(data_dir, subj, visualize = False, add_heartrate = False, cond_len=14, len_cond_s = 300, end_time_len = 180):
    gen_files = os.listdir(data_dir+subj)
    t_mat = []
    met_mat = []
    hr_mat = []
    for filename in gen_files:
        if len(filename) >= 4:
            if filename[-4:] == 'xlsx': # metabolics file
                if add_heartrate:
                    met_array = np.array(pd.read_excel(data_dir+subj+'\\'+filename, skiprows=3, usecols=[9, 14, 15, 23, 35])) # left then right insole forces          
                else:
                    met_array = np.array(pd.read_excel(data_dir+subj+'\\'+filename, skiprows=3, usecols=[9, 14, 15, 35])) # left then right insole forces        
                # load starting time
                met_raw = np.array(pd.read_excel(data_dir+subj+'\\'+filename, usecols=[4], header=None))
                starting_time = met_raw[0:2]
                test = str(starting_time[0][0]) + ' ' + str(starting_time[1][0])
                test_dt = datetime.datetime.strptime(test, '%m/%d/%Y %I:%M:%S %p')    
                met_len, cols = met_array.shape
                cond_indeces = list(np.arange(1,cond_len+1))
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
                        if met_array[i,-1] == cond_indeces[cnd]:
                            met_inds.append(i)
                            break
                        if met_array[i,-1] == start_inds[cnd]:
                            start_ind = start_inds.index(met_array[i,-1])
                            met_start_inds[start_ind] = i
                    if i == met_len-1: # didn't find the condition
                        met_inds.append(-1)
                        met_start_inds[len(met_inds)-1] = -1
                        
                # take the average of vco2 and vo2 over the num of avg_breaths, then find W from met eq
                for i, ind in enumerate(met_inds):
                    if ind != -1: # there exists this conditions for this subject
                        end_time = met_array[ind,0]
                        start_ind = ind
                        start_time = end_time

                        while (start_time.hour*3600 + start_time.minute*60 + start_time.second + end_time_len) > (end_time.hour*3600 + end_time.minute*60 + end_time.second):
                            start_ind -= 1 # decrement
                            start_time = met_array[start_ind,0]                    
                        vo2 = np.mean(met_array[start_ind:ind,1])
                        vco2 = np.mean(met_array[start_ind:ind,2])
                        met_vals[i] = (vo2*16.48 + vco2*4.48)/60.0 # BROCKWAY

                        if add_heartrate: # store values in the hr
                            hr[i] = np.mean(met_array[start_ind:ind,-2])
                        
                for i, ind in enumerate(met_start_inds):
                    if ind != -1: # there exists this conditions for this subject
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
                cond_cnt = -1
                for i,ind in enumerate(met_start_inds):
                    if ind != -1: # there exists this conditions for this subject
                        cond_cnt += 1
                        stop_ind = met_inds[cond_cnt]
                        t_mat.append(met_array[ind:stop_ind,0])
                        met_mat_temp = (met_array[ind:stop_ind,1]*16.48 + met_array[ind:stop_ind,2]*4.48)/60.0
                        met_mat.append(met_mat_temp)
                        hr_mat.append(met_array[ind:stop_ind,-2])
                    else:
                        t_mat.append([-1])
                        met_mat.append([-1])
                        hr_mat.append([-1])

                    
    return met_vals, hr, met_2mins, hr_int_mat, met_int_mat, start_stamps, t_mat, met_mat, hr_mat

def load_constants2(cur_dir, subjects):
    code_files = os.listdir(cur_dir)
    for fnm in code_files:
        if len(fnm) >= 7:
            if fnm[-7:] == 'rt2.csv':
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