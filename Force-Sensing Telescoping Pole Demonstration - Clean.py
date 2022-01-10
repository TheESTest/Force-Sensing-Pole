# -*- coding: utf-8 -*-
r"""
Elation Sports Technologies LLC
27 Oct 2021

Force-Sensing Telscoping Pole

This script parses, plots and animates the strain gauge data from a
telescoping pole.


"""

import csv,time
import matplotlib.pyplot as plt
import serial, io, datetime
from serial import Serial
import numpy as np
from matplotlib import animation
import matplotlib.cm as cm


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

plotAlpha = 0.25

#The extension length of EACH of the two telescoping segments of the pole
#during the test.
extension_length_inch = 25

animation_folder = r'C:\Users\(username)\Desktop\' #Replace (username) with your local username


#Read in the calibration data obtained used the Python script referenced above.
calib_folder = r'C:\Users\(username)\Desktop\'
calib_file = calib_folder + '\\' + 'Calibration_Data.csv'
calib_data = []
calib_dict = {}
with open(calib_file) as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            calib_data.append(row)

for i in range(0,len(calib_data)):
    if calib_data[i][0] == 'A_avg':
        calib_dict['A_avg'] = float(calib_data[i][1])
    if calib_data[i][0] == 'B_avg':
        calib_dict['B_avg'] = float(calib_data[i][1])
    if calib_data[i][0] == 'D_avg':
        calib_dict['D_avg'] = float(calib_data[i][1])
        
        
#https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
#x is the data you want to take the rolling average for.
#w is the size of the window to take the average
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

rolling_avg_window = 10

plot_individual_files_boolean = True

timestr = time.strftime("%d%b%Y_%H%M%p")

folder_path = animation_folder

file_type = '.csv'

file_names_to_process = []

file_names_to_process.append('Log_29Oct2021_1310PM')

file_labels = []

output_suffix = 'parsed'

data_all = []

for k in range(0,len(file_names_to_process)):
    
    file_name_to_process = file_names_to_process[k]

    file_path_to_process = folder_path + '\\' + file_name_to_process + file_type
    file_path_output= folder_path + '\\' + file_name_to_process + '_' + output_suffix + file_type
    
    print('Reading data from file: ' + file_name_to_process + file_type)
    
    raw_data = []
    
    with open(file_path_to_process) as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            for row in reader:
                raw_data.append(row)
    
    #Convert from strings to values if necessary, and remove header if necessary
    raw_data_2 = []
    start_row_index = 15 #Set to 0 to collect from the beginning of the data
    # end_row_index = 498#Set to -1 to collect to the end of the data
    
    #Strain gauge data sets
    time_data = []
    data_stream_1 = []
    data_stream_2 = []
        
    for i in range(start_row_index, len(raw_data)):
        
        temp_list = raw_data[i][0].split(',')
        temp_list_2 = []
        
        time_data.append(float(temp_list[0]))
        
        data_stream_1.append(float(temp_list[1][2:]))
        data_stream_2.append(float(temp_list[2][3:]))
        
        temp_list_2.append(time_data[-1])
        temp_list_2.append(data_stream_1[-1])
        temp_list_2.append(data_stream_2[-1])
        
        raw_data_2.append(temp_list_2)
    
    raw_data_2 = np.array(raw_data_2)
    
    time_data = np.array(time_data)
    time_data = time_data - time_data[0]
    
    fig,ax = plt.subplots()
    x_label_string = 'Time [sec]'
    y_label_string = 'Data'
    #ax.axis('equal')
    plt.grid(True,alpha=plotAlpha)
    plt.xlabel(x_label_string)
    plt.ylabel(y_label_string)
    
    plt.plot(time_data,data_stream_1,'-',color='tab:blue',label='Strain Gauge 1')
    plt.plot(time_data,data_stream_2,'-',color='tab:orange',label='Strain Gauge 2')
    
    plt.legend()
    
    plotTitle = file_name_to_process + file_type + '\n' + 'Raw Data'
    plt.title(plotTitle)
    
    plt.savefig(animation_folder + '\\' + file_name_to_process + '_RawData' + '.png', dpi=200)

    #With the raw data collected, use the calibrated values and formula
    #to convert the raw data into force,theta data.
    
    def calc_theta_and_F(A,B,D,L,f1,f2):
        theta_solved = -np.arctan2(f1*D,f2*A) - B #Added negative sign onto arctan for +/- 90 degree phase difference
        F_solved = f1/(A * L * np.sin(theta_solved + B)) #Recall that negative F is correct, b/c it points away from the theta direction.
        return theta_solved,F_solved

    #For a given f1,f2,L: calculate theta and F.
    
    f1_zero_weight = 234.6
    f2_zero_weight = 194
    
    raw_data_2[:,1] = raw_data_2[:,1] - f1_zero_weight
    raw_data_2[:,2] = raw_data_2[:,2] - f2_zero_weight
    data_to_process = raw_data_2
    
    A_avg = calib_dict['A_avg']
    B_avg = calib_dict['B_avg']
    D_avg = calib_dict['D_avg']
    
    theta_solved_list = []
    F_solved_list = []
    for i in range(0,len(data_to_process)):
        f1_curr = data_to_process[i,1]
        f2_curr = data_to_process[i,2]
        L_curr = extension_length_inch
        
        theta_curr, F_curr = calc_theta_and_F(A_avg,B_avg,D_avg,L_curr,f1_curr,f2_curr)
        
        theta_solved_list.append(theta_curr)
        F_solved_list.append(F_curr)
    
    F_solved_list = np.array(F_solved_list)
    theta_solved_list_rad = np.array(theta_solved_list)
    theta_solved_list_deg = np.rad2deg(theta_solved_list_rad)
    
    print()
    print('F_solved_list:')
    print(F_solved_list)
    
    print()
    print('theta_solved_list_deg:')
    print(theta_solved_list_deg)
    
    F_norm = -np.copy(F_solved_list)
    
    #Plot the force magnitude and theta values over time.
    fig,ax = plt.subplots()
    x_label_string = 'Time [sec]'
    y_label_string = 'Force Magnitude [lbf]'
    #ax.axis('equal')
    plt.grid(True,alpha=plotAlpha)
    plt.xlabel(x_label_string)
    plt.ylabel(y_label_string)
    plt.plot(time_data,F_norm,'-',color='tab:blue')
    plotTitle = file_name_to_process + file_type + '\n' + 'Force Magnitude'
    plt.title(plotTitle)
    plt.savefig(animation_folder + '\\' + file_name_to_process + '_ForceMagnitude' + '.png', dpi=200)
    
    fig,ax = plt.subplots()
    x_label_string = 'Time [sec]'
    y_label_string = 'Theta [degrees]'
    #ax.axis('equal')
    plt.grid(True,alpha=plotAlpha)
    plt.xlabel(x_label_string)
    plt.ylabel(y_label_string)
    plt.plot(time_data,theta_solved_list_deg,'-',color='tab:red')
    plotTitle = file_name_to_process + file_type + '\n' + 'Force Direction'
    plt.title(plotTitle)
    plt.savefig(animation_folder + '\\' + file_name_to_process + '_Theta' + '.png', dpi=200)
        
    
    #Linear map function
    #https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
    def linear_map(value, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
    
        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)
    
        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)
    
    #Have the line change color as the force magnitude changes.
    val_min = min(F_norm)
    val_max = max(F_norm)
    color_val_min = 0
    color_val_max = 1
    color_vals = []
    for i in range(0,len(F_norm)):
        val_curr = F_norm[i]
        color_vals.append(linear_map(val_curr,val_min,val_max,color_val_min,color_val_max))
    
    color_vals = np.array(color_vals)
    #https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib
    colors = cm.rainbow(color_vals)
    
    
    #Create a "functional" animation instead of a frames/artist-based one
    #https://github.com/lukepolson/youtube_channel/blob/main/Python%20Metaphysics%20Series/vid7.ipynb
    def update(i):
        ln1.set_data([theta_solved_list_rad[i],theta_solved_list_rad[i]], [0,F_norm[i]])
        ln1.set_color(colors[i])
        ln1.set_linewidth(5)
        time_string_curr = format(time_data[i], '.2f')
        ann1.set_text('t = ' + time_string_curr + ' sec')
        
        return ln1, ann1
    
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(-1)  # theta increasing clockwise
    r_limit = max(F_norm)
    ax.set_ylim(0,r_limit)
    #ax.set_facecolor('k')
    #ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
    #ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks
    ln1, = plt.plot([], [], 'b-', lw=3, markersize=8)
    ann1 = plt.annotate('test', xy=(1.3,r_limit), color = 'k', bbox=dict(boxstyle="square", alpha=1, facecolor='w'))
    # ax.set_ylim(-4,4)
    # ax.set_xlim(-4,4)
    
    print()
    print('Creating animation in a Figure...')
    #Interval is the delay between frames in msec. It is an integer.
    ani = animation.FuncAnimation(fig, update, frames=len(F_norm), interval=1)
    
    print()
    print('Saving MP4...')
    
    #You must have FFmpeg installed. Enter the filepath for ffmpeg.exe here:
    plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\(username)\...\ffmpeg.exe'
    writervideo = animation.FFMpegWriter(fps=24) 
    ani.save(animation_folder + '\\' + file_name_to_process + '_' + 'animation' + '.mp4', writer=writervideo)
    print('MP4 saved!')
    

print()
print('Script concluded.')


    