from well_coefficient import *

# There should be no empty value in the dataset
input_file = 'data-ht-7.csv'  # 1 - 2 - 5 - 6 - 7
timestamp, galery, atm, diff = read_file(input_file)
num_galery, MINLEN, DELTA = parameters(input_file)
cycles, ht, label_cycles, start, times = water_heigth(timestamp, galery, atm)


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
	csv_output += "{}, {}, {}, {}".format(np.std(slopes), -1*np.std(slopes)/np.mean(slopes), min(slopes), max(slopes))
	csv_output += "{}, {},".format(np.mean(coefMeans), np.std(coefMerged))
	with open("coef_output.csv", 'w') as f:
		f.write(csv_output)

if plot: 
	plt.legend()
	plt.show()

# TODO
# CSV output with galery number
# Calculate tau based on the real delta max and not on the value of the cycle (method does not work if cycles are not complete)
# bug linearisation