# Estimate energy expenditure per gait cycle
# arguments: height, weight, gender (M/F), age, subject_number, cond_number, time_to_record
import time
from datetime import datetime
from datetime import timedelta
import numpy as np
import board
import busio
import digitalio
import adafruit_fxos8700
import adafruit_fxas21002c
import adafruit_tca9548a
import os
import sys
from scipy import signal

printing = False
height = float(sys.argv[1])
weight = float(sys.argv[2])
gender = sys.argv[3]
age = float(sys.argv[4])
subj_dir = sys.argv[5]
cond_dir = sys.argv[6]
if len(sys.argv) > 7:
    rec_time = int(sys.argv[7])
else:
    rec_time = 400

# connecting sensors
i2c = busio.I2C(board.SCL, board.SDA)
tca = adafruit_tca9548a.TCA9548A(i2c)
accel_shank_L = adafruit_fxos8700.FXOS8700(tca[1], accel_range = adafruit_fxos8700.ACCEL_RANGE_8G)
gyro_shank_L = adafruit_fxas21002c.FXAS21002C(tca[1], gyro_range = adafruit_fxas21002c.GYRO_RANGE_2000DPS)
accel_thigh_L = adafruit_fxos8700.FXOS8700(tca[0], accel_range = adafruit_fxos8700.ACCEL_RANGE_8G)
gyro_thigh_L = adafruit_fxas21002c.FXAS21002C(tca[0], gyro_range = adafruit_fxas21002c.GYRO_RANGE_2000DPS)

# setting up sampling params
sample_freq = 100 # in hz
samples_per_msg = 5*sample_freq # size of files to save data periodically
saving = False
stride_detect_window = 4*sample_freq # size of recent data to store when looking for strides
detect_window = int(0.25*sample_freq) # previous time window to look for a new strike to use model
watching_heelstrike = True # flag for detecting new heelstrikes in the detect_window
watch_strike_cnt = 0 # count to a certain value to wait until detect_window has been cleared
basal_t_thresh = 8.0 # minimum time between estimates to assume standing
offset_t = 1.0 # offset from previous estimates to add the standing estimates

# file saving params
time_interval = 1.0/sample_freq
time_start = time.time()
cur_time = time_start
init_time = time_start
cnt = 0
data_signals = 13
data_array = np.zeros((samples_per_msg, data_signals))
stride_window = np.zeros((stride_detect_window, data_signals))
msg_cnt = 0
stand_aug_fact = 1.41

# processing params
deg2rad = 0.0174533
shank_gyro_z_ind = 2
shift_ind = 0 # offset between IMU heelstrike detection and the previous GRF based method
wn = 6 # crossover frequency for low-pass filter
filt_order = 4
peak_height_thresh = 70 # minimum value of the shank IMU gyro-Z reading in deg/s
peak_min_dist = int(0.6*sample_freq) # min number of samples between peaks
b,a = signal.butter(filt_order, wn, fs=sample_freq) # params for low-pass filter

# loading model weights
model_dir = 'real_time_models/full_fold_aug/' #full_sm_all/'
model_weights = np.loadtxt(model_dir + 'weights.csv', delimiter=',') # model weight vector

estimate_file_name = 'energy_exp_estimates.csv'
file_ts_name = "file_timestamp.csv"
main_save_dir = 'save'
tot_dir = main_save_dir+'/'+subj_dir+'/'+cond_dir+'/'

if not os.path.exists(tot_dir):
    os.makedirs(tot_dir)
else:
    print("Overwriting dir: ", tot_dir)
    
if not online: # load file if not getting data online
    offline_file = 'save/test_imu_collect_7.csv' # walking test
    offline_data = np.loadtxt(offline_file, delimiter=',')

### PROCESSING FUNCTIONS
# take in shank IMU vector, filter, look for thresholding & min_distance, return strike indeces
def checkPeaks(strike_vec, b, a, peak_height_thresh, peak_min_dist):
    strike_vec_filt = signal.filtfilt(b,a,strike_vec)
    peak_list = signal.find_peaks(strike_vec_filt, height=peak_height_thresh, distance=peak_min_dist)
    return peak_list[0]
    
# downsample the data into a discrete number of bins
def binData(data_array, num_bins=30):
    return signal.resample(data_array, num_bins) # resamples along axis = 0 by default

# pass in array of data to process [time_samples x num_features] into [num_bins x num_features]
# start_ind, end_ind are indeces of start/end of gait cycle
# b, a are filter parameters
def processRawGait(data_array, start_ind, end_ind, shift_ind, b, a, weight, height, num_bins=30):
    gait_data = data_array[start_ind:end_ind, :] # crop to the gait cycle
    gait_data = gait_data*np.array([deg2rad,-deg2rad,-deg2rad,deg2rad,-deg2rad,-deg2rad,1,-1,-1,1,-1,-1]) # flip y & z, convert to rad/s
    filt_gait_data = signal.filtfilt(b,a,gait_data, axis=0) # low-pass filter
    bin_gait = binData(filt_gait_data) # discretize data into bins
    shift_flip_bin_gait = bin_gait.transpose() # get in shape of [feats x bins] for correct flattening
    model_input = shift_flip_bin_gait.flatten()
    model_input = np.insert(model_input, 0, [1.0, weight, height]) # adding a 1 for the bias term at start
    return model_input

def basalEst(height, weight, age, gender, stand_aug_fact, kcalPerDay2Watt = 0.048426):
    if gender == 'M':
        offset = 5
    elif gender == 'F':
        offset = -161
    basal_rate = (10.0*weight + 625.0*height - 5.0*age + offset)*kcalPerDay2Watt*stand_aug_fact
    return basal_rate

basal_rate = basalEst(height, weight, age, gender, stand_aug_fact)

np.savetxt(main_save_dir + '/' + subj_dir + '/basal_rate.txt', [basal_rate], delimiter=',')
print("Basal rate: ",basal_rate)
basal_rate = round(basal_rate,3)

### REAL TIME EXECUTION LOOP
print("Starting real-time execution...")
init_datetime = datetime.now()
time_start = init_datetime
time_stamp = round(time.time() - init_time,3)
prev_time_stamp = time_stamp
dt_stamp = init_datetime
prev_dt_stamp = init_datetime

while(not online or (time.time() - init_time < rec_time)):
    cur_time = datetime.now()#time.time()
    if (cur_time - time_start).total_seconds() > time_interval: # loop @ sample rate
        time_start = cur_time
        try:
            data_array[cnt,:3] = gyro_shank_L.gyroscope
            data_array[cnt,3:6] = gyro_thigh_L.gyroscope 
            data_array[cnt,6:9] = accel_thigh_L.accelerometer
            data_array[cnt,9:12] = accel_shank_L.accelerometer
            data_array[cnt,12] = 0
        except: # error catch for i2c
            print("i2c error")
            if cnt != 0:
                tmp_cnt = cnt - 1
            else:
                tmp_cnt = samples_per_msg - 1
            data_array[cnt,:] = data_array[tmp_cnt,:]
            data_array[cnt,12] = 1 # error flag for stored data
        
        if watching_heelstrike: # if looking for heelstrike
            peak_list = checkPeaks(stride_window[:,shank_gyro_z_ind], b, a, peak_height_thresh, peak_min_dist) # check for peaks
            if len(peak_list) > 1: # checking if a new heel strike has occured
                if (stride_detect_window - peak_list[-1]) < detect_window: # peak has occured in last detect_window of data
                    watching_heelstrike = False # now wait before detecting heelstrikes again
                    model_input = processRawGait(stride_window[:,:-1], peak_list[-2], peak_list[-1], shift_ind, b, a, weight, height) # process most recent gait data
                    estimate = round(max(np.dot(model_input, model_weights), basal_rate),3)
                    model_input_short = np.concatenate((model_input[:93], model_input[123:]))
                    time_stamp = round(time.time() - init_time,3)
                    dt_stamp = datetime.now()
                    
                    ### code to add scaled basal estimates when gaps of offset_t seconds or more occur
                    if (dt_stamp-prev_dt_stamp).total_seconds() >= basal_t_thresh:
                        with open(tot_dir+estimate_file_name,'a') as f:
                            f.write("{},{},{},{}".format(prev_time_stamp+offset_t, prev_dt_stamp+timedelta(seconds=offset_t), basal_rate, basal_rate))
                            f.write("\n")
                            f.write("{},{},{},{}".format(time_stamp-offset_t, dt_stamp-timedelta(seconds=offset_t), basal_rate, basal_rate))
                            f.write("\n")
                    
                    if printing:
                        print("Estimates (W): ", int(estimate), int(estimate))
                    
                    # saving estimate and time stamp
                    with open(tot_dir+estimate_file_name,'a') as f:
                        f.write("{},{},{},{}".format(time_stamp, dt_stamp, estimate, estimate))
                        f.write("\n")
                    f.close()
                    prev_time_stamp = time_stamp # update the previous times for future use
                    prev_dt_stamp = dt_stamp
        else: # count until the peak has cleared the recent window
            if watch_strike_cnt > detect_window:
                watching_heelstrike = True
                watch_strike_cnt = 0
            else:
                watch_strike_cnt += 1
        cnt += 1
        
    if (cnt == samples_per_msg) and online: # save raw data periodically
        cnt = 0
        msg_cnt += 1
        np.save(tot_dir+'file'+str(msg_cnt)+".npy", data_array)
        cur_datetime = datetime.now()
        cur_datetime_str = cur_datetime.strftime("%d/%m/%Y %H:%M:%S")
        with open(tot_dir+file_ts_name,'a') as f:
            f.write(cur_datetime_str)
            f.write("\n")
        f.close()
print("Script finished.")