# -*- coding: cp1252 -*-
from well_coefficient import *

# There should be no empty value in the dataset
input_file = 'data-ht-7.csv'  # 1 - 2 - 5 - 6 - 7
timestamp, galery, atm = read_file(input_file)
num_galery, MINLEN, DELTA = parameters(input_file)
cycles, ht, label_cycles, start, times = water_heigth(timestamp, galery, atm)

if write_csv:
	csv_output += "{},".format(num_galery)

#MINLEN = 0

# SAAS
coef = caracteristic_time(cycles, start, DELTA, MINLEN)
coefMerged, coefMeans, coefStd = agreggate_tau(coef)

# SLOPES
fig, ax = plt.subplots()
ax.set_prop_cycle('color', colors)
slopes, std_errors, intercepts, ln_cycles, len_cycles, date_strings, times_ln, valid_cycles = compute_slopes(cycles, start, DELTA, MINLEN, label_cycles, times)

# FILTER 
slopes = filter_slopes(slopes, std_errors, intercepts, ln_cycles, len_cycles, date_strings, times_ln, valid_cycles, num_galery, NB_STD)

# PRINT
if plot:
    plt.figure(1)
    plt.plot(np.array(range(1, STEP_TIME*len(ht)+1, STEP_TIME)), ht) # the whole dataset
    plt.title("Evolution de la hauteur d'eau au sein du puits {}".format(str(num_galery)))
    plt.ylabel("Hauteur d'eau (cm H2O)")
    plt.xlabel("Temps (min)")
    #for i in range(0,len(cycles)) : 
    #    plt.plot(times1[i], cycles[i], label="H(t)")

# mean of all the coef obtained 
if print_data: 
	print("SAAS method: {}+/-{}".format(np.mean(coefMeans), np.std(coefMerged)))
	print("Cycles length: {}".format(len_cycles))
	print("Slopes: {}".format(slopes))
	print("Mean slope: {}".format(-1*np.mean(slopes)))
	print("Std error: {}".format(np.std(slopes)))
	print("Std/Mean: {}".format(-1*np.std(slopes)/np.mean(slopes)))
	print("Min slope: {}".format(min(slopes)))
	print("Max slope: {}".format(max(slopes)))

if write_csv :
	# characteristic time 
	csv_output += "{}, {},".format(np.mean(coefMeans), np.std(coefMerged))
	# slope
	csv_output += "{}, {}, {}, {},".format(-1*np.mean(slopes), np.std(slopes), min(slopes), max(slopes))
	with open("coef_output.csv", 'w') as f:
		f.write(csv_output)

if plot: 
	plt.legend()
	plt.show()

# TODO
# Calculate tau based on the absolute delta max (obtained by going throught the whole data) and not on the value of the cycle (method does not work if cycles are not complete -> 63% of the delta of that cycle does not correspond to the 63% of the well if the cycle is not complete). Adding this feature would remove the need to only consider some cycles for the method using the charactéristic time and thus remove the need of DELTA and MINLEN
# Should also check that it's the right beginning of the cycle
# Add linear approx when concatenating (instead of deleting c2, change its value with a linear approx between last elem of c1 and first elem of c3)