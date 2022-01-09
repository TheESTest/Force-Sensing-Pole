r"""
Elation Sports Technologies LLC
22 Oct 2021

Force-Sensing Telescoping Pole

Calibration Processing Script


"""

import matplotlib.pyplot as plt
import numpy as np
import time,csv
import matplotlib.cm as cm

plt.close('all')

currTimeString = time.strftime('%d%b%Y_%I%M%p')

data_path = r'C:\Users\(username)\Desktop' #Change (username) to the local username for the PC
folder_path = data_path

plot_alpha = 0.4

#Next, process and plot the strain gauge calibration data.
data_file_name = r'Sensing Telescoping Pole Calibration Example - Clean.csv'
data_file_path = data_path + '\\' + data_file_name

start_row = 3
raw_data = []

nonzero_weight_lbf = 2.5
center_upper_hand_pos_inch = 21.75
center_lower_hand_pos_inch = 3.25
center_strain_gauges_pos_inch = 34.5

with open(data_file_path, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        raw_data.append(row)

raw_data_2 = []
for i in range(start_row,len(raw_data)):
    row_curr = raw_data[i]
    temp_list = []
    for  j in range(0,6):    
        temp_list.append(float(row_curr[j]))
    
    raw_data_2.append(temp_list)

raw_data_2 = np.array(raw_data_2)
weight_boundary_index = (np.where(raw_data_2[:,3]==0.0))[0][-1]

data_zero_weight = raw_data_2[0:weight_boundary_index+1]
data_nonzero_weight = raw_data_2[weight_boundary_index+1:]

#Tare/zero the readings by subtracting the zero-extra-weight data from the
#nonzero-weight data.
data_tared = np.copy(data_zero_weight)
data_tared[:,4] = data_nonzero_weight[:,4] - data_zero_weight[:,4]
data_tared[:,5] = data_nonzero_weight[:,5] - data_zero_weight[:,5]


#Plot the data with respect to theta angle
fig,ax = plt.subplots()
plt.grid(True,alpha=plot_alpha)
plt.ylabel('Reading')
plt.xlabel('Theta [degrees]')
plt.title('Net Strain Gauge Readings for 2.5lbf Weight w.r.t. Theta\nVarying Extension Length')

for i in range(0,len(data_tared),4):
    theta_data = data_tared[i:i+4,2]
    gauge_data_t_1 = data_tared[i:i+4,4]
    gauge_data_t_2 = data_tared[i:i+4,5]
    if i == 0:
        plt.plot(theta_data,gauge_data_t_1,'-o',color='tab:blue',markersize=3,label='Gauge 1')
        plt.plot(theta_data,gauge_data_t_2,'-o',color='tab:orange',markersize=3,label='Gauge 2')
    else:
        plt.plot(theta_data,gauge_data_t_1,'-o',color='tab:blue',markersize=3)
        plt.plot(theta_data,gauge_data_t_2,'-o',color='tab:orange',markersize=3)

plt.legend()
plt.savefig(folder_path + '\\' + 'Strain_Gauge_Readings_wrt_Theta' + '.png', dpi=200)


#Fit each set of 4 readings in theta to a sinusoid.
#https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
import scipy.optimize
def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

res = fit_sin(data_tared[0:4,2], data_tared[0:4,5])

#Fit the data when it is repeated a few times, to get a better fit.
#Calculate the average phase shift for the first strain gauge pair, and for the
#second strain gauge pair. They should be about 90 degrees (pi/2 rad) out of phase.

num_repeats = 2
amp_1_list = [] #List of amplitudes of sine fits for strain gauge pair #1
amp_2_list = [] #List of amplitudes of sine fits for strain gauge pair #2
phase_1_list = [] #List of phase shifts of sine fits for strain gauge pair #1
phase_2_list = [] #List of phase shifts of sine fits for strain gauge pair #2
offset_1_list = [] #List of offsets of sine fits for strain gauge pair #1
offset_2_list = [] #List of offsets of sine fits for strain gauge pair #2

#Phase shift #1 is B --> Calculate B_avg
#Phase shift #2 is E --> Calculate E_avg

for i in range(0,len(data_tared),4):
    t_temp = data_tared[i:i+4,2] #Theta values
    s1_temp = data_tared[i:i+4,4] #Strain gauge pair #1 values
    s2_temp = data_tared[i:i+4,5] #Strain gauge pair #2 values

    for j in range(1,num_repeats):
        t_temp = np.concatenate((t_temp,t_temp + 360 * j))
        s1_temp = np.concatenate((s1_temp,s1_temp))
        s2_temp = np.concatenate((s2_temp,s2_temp))

    res1_temp = fit_sin(t_temp, s1_temp)
    res2_temp = fit_sin(t_temp, s2_temp)
    
    amp_1_list.append(res1_temp['amp'])
    amp_2_list.append(res2_temp['amp'])
    phase_1_list.append(res1_temp['phase'])
    phase_2_list.append(res2_temp['phase'])
    offset_1_list.append(res1_temp['offset'])
    offset_2_list.append(res2_temp['offset'])
    
    #Make a plot of the first result in the series, as a spot check
    t_many = np.linspace(0,max(t_temp),1000)
    s1_sine = res1_temp['amp'] * np.sin(np.deg2rad(t_many) + res1_temp['phase']) + res1_temp['offset']
    s2_sine = res2_temp['amp'] * np.sin(np.deg2rad(t_many) + res2_temp['phase']) + res2_temp['offset']
    
    if i == 0:
        fig,ax = plt.subplots()
        plt.xlabel('Theta [deg]')
        plt.ylabel('Strain Gauge Readings')
        plt.title('Sine Function Fit Test for Extension Length [inch]: ' + str(data_tared[i,0]))
        plt.plot(t_temp,s1_temp,'-o',color='tab:blue',label='Gauge 1 Data')
        plt.plot(t_temp,s2_temp,'-o',color='tab:orange',label='Gauge 2 Data')
        plt.plot(t_many,s1_sine,'--',color='tab:blue',label='Gauge 1 Fit')
        plt.plot(t_many,s2_sine,'--',color='tab:orange',label='Gauge 2 Fit')
        plt.legend()
        plt.savefig(folder_path + '\\' + 'Sine_Fit_Test_Smallest_Extension' + '.png', dpi=200)

B_avg = np.mean(phase_1_list)
E_avg = np.mean(phase_2_list)


#Next, perform linear fit for each set of data w.r.t. extension length to determine
#the slope (passing through zero.)
#Divide those slopes by the extension length and calculate the average
#to determine A_avg and D_avg for strain gauge pair #1 and strain gauge pair #2, respectively.

#Plot the data

slopes_1_list = []
slopes_2_list = []

fig,ax = plt.subplots()
plt.grid(True,alpha=plot_alpha)
plt.ylabel('Reading')
plt.xlabel('Extension Length [inch]')
plt.title('Net Strain Gauge Readings for 2.5lbf Weight w.r.t. Extension Length')#\nVarying Theta')

for i in range(0,4):
    
    ext_data = data_tared[i:len(data_tared):4,0]
    gauge_data_e_1 = np.abs(data_tared[i:len(data_tared):4,4])
    gauge_data_e_2 = np.abs(data_tared[i:len(data_tared):4,5])
    
    #Don't use the extension = 2 inch data point, to improve the linear fit.
    ext_data = ext_data[1:]
    gauge_data_e_1 = gauge_data_e_1[1:]
    gauge_data_e_2 = gauge_data_e_2[1:]
    
    #Force the linear fit to pass through zero
    #https://stackoverflow.com/questions/46164012/how-to-use-numpy-polyfit-to-force-scatter-points-linear-fit-pass-through-zero
    m1 = np.linalg.lstsq(ext_data.reshape(-1,1), gauge_data_e_1, rcond=None)[0][0]
    m2 = np.linalg.lstsq(ext_data.reshape(-1,1), gauge_data_e_2, rcond=None)[0][0]
    
    slopes_1_list.append(m1)
    slopes_2_list.append(m2)
    
    ext_data_fit = np.array([0,max(ext_data)])
    gauge_data_fit_1 = ext_data_fit * m1
    gauge_data_fit_2 = ext_data_fit * m2
    
    
    if i == 0:
        plt.plot(ext_data,gauge_data_e_1,'-o',color='tab:blue',markersize=3,label='Gauge 1 Data')
        plt.plot(ext_data,gauge_data_e_2,'-o',color='tab:orange',markersize=3,label='Gauge 2 Data')
        plt.plot(ext_data_fit,gauge_data_fit_1,'--',color='tab:blue',markersize=3,label='Gauge 1 Fit')
        plt.plot(ext_data_fit,gauge_data_fit_2,'--',color='tab:orange',markersize=3,label='Gauge 2 Fit')
    else:
        plt.plot(ext_data,gauge_data_e_1,'-o',color='tab:blue',markersize=3)
        plt.plot(ext_data,gauge_data_e_2,'-o',color='tab:orange',markersize=3)
        plt.plot(ext_data_fit,gauge_data_fit_1,'--',color='tab:blue',markersize=3)
        plt.plot(ext_data_fit,gauge_data_fit_2,'--',color='tab:orange',markersize=3)

plt.legend()
plt.savefig(folder_path + '\\' + 'Strain_Gauge_Readings_wrt_Extension' + '.png', dpi=200)

A_avg = np.mean(slopes_1_list)
D_avg = np.mean(slopes_2_list)

#Write the calibration data to a CSV file.
with open(folder_path + '\\' + 'Calibration_Data' + '.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['A_avg'] + [A_avg])
    spamwriter.writerow(['B_avg'] + [B_avg])
    spamwriter.writerow(['D_avg'] + [D_avg])

#The goal is to know theta and F (i.e. the direction and magnitude of the force
#acting on the tip of the pole) given the 2 x strain gauge readings (f1,f2) and the
#length of the pole (which could also be measured using a sensor or manually).

#With A_avg, B_avg, D_avg, E_avg defined, we can calculate F and theta.
#Note that E = B (+/-) pi/2 --> 90 degrees out of phase, so E can be eliminated

def calc_theta_and_F(A,B,D,L,f1,f2):
    theta_solved = -np.arctan2(f1*D,f2*A) - B #Added negative sign onto arctan for +/- 90 degree phase difference
    F_solved = f1/(A * L * np.sin(theta_solved + B)) #Recall that negative F is correct, b/c it points away from the theta direction.
    return theta_solved,F_solved

#Test the function using the calibation data.
#For a given f1,f2,L: calculate theta and F.

theta_solved_list = []
F_solved_list = []
data_to_process = data_tared
for i in range(0,len(data_to_process)):
    f1_curr = data_to_process[i,4]
    f2_curr = data_to_process[i,5]
    L_curr = data_to_process[i,0]
    
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

print()
print('Script concluded.')


