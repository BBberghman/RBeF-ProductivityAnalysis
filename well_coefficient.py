# -*- coding: cp1252 -*-
from __future__ import division # http://stackoverflow.com/questions/1267869/how-can-i-force-division-to-be-floating-point-division-keeps-rounding-down-to-0
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import linregress
from matplotlib.dates import date2num
from datetime import datetime, date

# Coded for Python 2.7

# Output parameters
print_help = 0      # 1 = print help and result, 0 = print only result
plot = 0            # 1 = print graph
print_data = 0      # 1 = print data
write_csv = 0       # 1 = write in CSV

# Parameters
STEP_TIME = 1       # Minutes between each sample, depend on the parameter used in the data logger

# Global variables
NB_MERGE = 8        # Max nb of elements in the cycle delelted (c2) when considering the concatenation of c1 and c3
EPS = 1e-7          # EPS value added when calculating the ln to avoid having ln(0)
NB_MAX = 10         # Nb of elements to calculate the max of the cycle (mean of the NB_MAX last elements)
NB_MIN = 5          # Nb of elements to calculate the min (mean of the NB_MIN first elements)
NB_STD = 2          # Nb of std deviation beyond which cycles are discarded

# colors when ploting the cycles
colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]

#if write_csv : 
csv_output = "Galery, 95 mean, 95 std, 87 mean, 87 std, 63 mean, 63 std, SAAS mean, SAAS std, slope mean, slope std, slope min, slope max\n" # CSV string output

# Parse data
def date2int(date_str):
    try:
        date = datetime.strptime(date_str,  '%d-%m-%y %H:%M')
    except ValueError:
        date = datetime.strptime(date_str,  '%Y/%m/%d %H:%M:%S')
    return date

def read_file(input_file):
    timestamp = []
    galery = []
    atm = []
    diff = []
    with open(input_file, 'r') as f:
        line = f.readline().strip()
        while line:
            # print line
            lineSplit = line.split(',')

            timestamp.append(date2int(lineSplit[0]))
            galery.append(float(lineSplit[1]))
            atm.append(float(lineSplit[2]))
            #diff.append(float(lineSplit[3]))
            line = f.readline().strip()
    return timestamp, galery, atm

# Get the correct parameters if name of input file has the number of the galery in it
# MINLEN = minimal length of the cycle to be considered an analysable cycle
# DELTA = minimal height difference between hf and h0 to be considered an analysable cycle
def parameters(input_file):
    if '1' in input_file:
        num_galery = 1
        MINLEN = 80
        DELTA = 150
    elif '2' in input_file:
        num_galery = 2
        MINLEN = 80
        DELTA = 150
    elif '5' in input_file:
        num_galery = 5
        MINLEN = 100
        DELTA = 100
    elif '6' in input_file:
        num_galery = 6
        MINLEN = 100
        DELTA = 100
    elif '7' in input_file:
        num_galery = 7
        MINLEN = 100
        DELTA = 150
    else:
        num_galery = -1    
        MINLEN = 100
        DELTA = 50
    return num_galery, MINLEN, DELTA

def max_cycle(cycle):
    return np.sum(cycle[-NB_MAX:])/NB_MAX

def min_cycle(cycle):
    return np.sum(cycle[:NB_MIN])/NB_MIN

def delta(cycle):
    deltaProb = max_cycle(cycle)-min_cycle(cycle) 
    if deltaProb > 0:       # as deltaProb is not the absolute delta (see fun of min and max_cycle), it might happen that it's < 0
        return deltaProb
    else:
        return max(cycle) - min(cycle)      # absolute diff

# Two criterias are used to define a complete cycle: a min length and a min difference of total height, defined for each galery at the beginning
def complete_cycle(cycle, deltaH, minLen):
    return delta(cycle) > deltaH and len(cycle) > minLen


########      ###     ########    #######   ########   ##      ##   #######
##     ##    ## ##    ##     ##  ##     ##     ##      ###     ##  ##
##     ##   ##   ##   ##     ##  ##            ##      ## ##   ##  ##
#######    ##     ##  ########    #######      ##      ##  ##  ##  ##   ####
##         #########  ##   ##           ##     ##      ##   ## ##  ##     ##
##         ##     ##  ##    ##   ##     ##     ##      ##     ###  ##     ##
##         ##     ##  ##     ##   #######   ########   ##      ##  ########

def merge(cycle, ls):
    for elem in cycle:
        ls[-1].append(elem)

def add_cycles(cycle, cycles, label, labels):
    cycles.append(cycle)
    labels.append(label)

# Order data to get the production and pumping cycles

# When we have 3 cycles one after the other such as c1, c2, c3
# If cycle c2 is smaller than a certain length (NB_MERGE), we don't consider it. It is thus discarded and c1 and c3 are concatenated
def merge_cycles(cycles, labels, ):
    new_cycles = [cycles[0]]
    new_labels = [labels[0]]
    i = 1
    while i < len(cycles) - 1: 
        if len(cycles[i]) < NB_MERGE :
            if print_help: print("Deleting the following cycle (c2): {}".format(cycles[i]))
            merge(cycles[i+1], new_cycles)      # c3
            i = i + 2
        else:
            add_cycles(cycles[i], new_cycles, labels[i], new_labels)
            i = i + 1 
    return new_cycles, new_labels

def same_cycle(value, ref, asc):
    return (value >= 0.999*ref and asc) or (value <= 1.001*ref and not asc)
      
def cut_data(galery, timestamp):
    galery_cycles = []
    label_cycles = [timestamp[0]]
    ref = galery[0]     
    cycle = []

    # determine if the first cycle is going up or down
    asc = True if galery[1] > galery[0] else False
    start = 0 if asc == True else 1
    
    for i in range (0, len(galery)) :
        value = galery[i]
        
        # Check if we're still in the same cycle; relaxed rules are used to take small variation into account
        if same_cycle(value, ref, asc): cycle.append(value)
        else :
            if cycle is not [] : add_cycles(cycle, galery_cycles, timestamp[i], label_cycles)
            cycle = [value]
            asc = not asc
        # update ref only if it's better than before (because of the relaxed rules used before)
        if (asc and ref < value) or (not asc and ref > value) : ref = value

    # when we're done, need to add the last cycle
    if cycle is not []: add_cycles(cycle, galery_cycles, timestamp[-1], label_cycles)
    cycles_concat, label_concat = merge_cycles(galery_cycles, label_cycles)
    return cycles_concat, label_concat, start

def time_of(cycles):
    return [ np.array(range(0, STEP_TIME*len(cycles[i]),STEP_TIME)) for i in range(0, len(cycles)) ]

def water_heigth(timestamp, galery, atm):
    ht = np.array(galery) - np.array(atm)
    cycles, label_cycles, start = cut_data(ht, timestamp)
    times = time_of(cycles)
    return cycles, ht, label_cycles, start, times


 #######      ###        ###      #######   
##     ##    ## ##      ## ##    ##     ##  
##          ##   ##    ##   ##   ##         
 #######   ##     ##  ##     ##   #######   
       ##  #########  #########         ##  
##     ##  ##     ##  ##     ##  ##     ##  
 #######   ##     ##  ##     ##   #######   

# Control theory method

def getDeltaHGalery(cycles, start):
    deltaH_list = []
    for i in range (start, len(cycles),2):
        deltaH_list.append(delta(cycles[i]));
    return max(deltaH_list) 

def considered_cycle(cycle):
    return delta(cycle) > 40

def caracteristic_time(cycles, start, deltaH, minLen):
    coef1 = (1 - 1/math.exp(1))*100
    coef2 = (1 - 1/math.exp(2))*100
    coef3 = (1 - 1/math.exp(3))*100

    percentages = {coef3:3, coef2:2, coef1:1}
    coef = {coef3:[], coef2:[], coef1:[]}


    for i in range(start,len(cycles),2) :
        cycle = cycles[i]

        if complete_cycle(cycle, deltaH, minLen) :
            for p in percentages:
                ratio = percentages[p]
                minCycle = min_cycle(cycle)
                # First, find the threshold at 95, 86.5 or 63.2%
                threshold = minCycle + p * delta(cycle) / 100
                if print_help: print("expected: {}, exact: {}".format(threshold, min(cycle, key=lambda x:abs(threshold-x))))
                
                # Then the lambda function will find the closest actual value in the list.
                # Its index gives us the abscissa of the corresponding value. Since all cycles begin
                # at zero, this abscissa is also the T, 2T or 3T of our system.
                idx = cycle.index(min(cycle, key=lambda x:abs(threshold - x)))   # minimize (threshold - value of cycle) to get the index closer to the threshold
               
               # linear approximation because the data is discrete and we might not find the value corresponding to the threshold
                if cycle[idx] > threshold:
                    idx = idx +  (idx-(idx-1)) * (cycle[idx] - threshold) / (cycle[idx] - cycle[idx-1])
                elif cycle[idx] < threshold:
                    idx = idx +  ((idx+1)-idx) * (threshold - cycle[idx]) / (cycle[idx+1] - cycle[idx])
                
                time_carac = idx * STEP_TIME / ratio

                try:
                    coef[p].append(1/time_carac)
                except ZeroDivisionError:
                    if print_help: print("ERROR. Cycle from {} has a characteristic time of {} for percentage {}".format(label_cycles[i], time_carac, p))
    return coef


def agreggate_tau(coef):
#compute the mean and std for each percentage to aggregate the data of each cycle
    coefMerged = []
    coefMeans = []
    coefStd = []
    for c in coef:
        coefMeans.append(np.mean(coef[c]))
        coefStd.append(np.std(coef[c]))
        if print_data: print("{} mean: {}+/-{}".format(c, coefMeans[-1], coefStd[-1]))
        if write_csv : csv_output += "{}, {},".format(coefMeans[-1], coefStd[-1])
        for i in coef[c]:
            coefMerged.append(i)
    return coefMerged, coefMeans, coefStd



 #######   ##           #####    ########   #########   #######   
##     ##  ##         ##     ##  ##     ##  ##         ##     ##  
##         ##         ##     ##  ##     ##  ##         ##         
 #######   ##         ##     ##  #######    ######      #######   
       ##  ##         ##     ##  ##         ##                ##  
##     ##  ##         ##     ##  ##         ##         ##     ##  
 #######   #########    #####    ##         #########   #######   


# Compute the ln
def compute_ln(ht):
    hf = max_cycle(ht)
    h0 = min_cycle(ht)
    return np.log(abs((ht-hf + EPS)/(h0 - hf)))

def compute_slopes(cycles, start, deltaH, minLen, label_cycles, times):
    ln_cycles = []
    len_cycles = []
    times_ln = []
    slopes = []
    intercepts = []
    std_errors = []
    date_strings = []

    # Non-discarded cycles
    valid_cycles = []
    for i in range(start, len(cycles), 2) :
        cycle = cycles[i]
        
        #verify we have a complete cycle
        if complete_cycle(cycle, deltaH, minLen) :
            date_string = label_cycles[i].strftime('%d-%m-%y %H:%M')
            time_ln = times[i]
            ln = compute_ln(cycle)
           
           # look for the part of the slope we have to consider to get a R > 0.99
            r_value = 0
            offset = 0
            while abs(r_value) < 0.99 and offset < len(ln) - 1:
                offset += 1
                slope, intercept, r_value, p_value, std_err = linregress(time_ln[:-offset], ln[:-offset])

            if print_help: print("Search is done --------------------")
            time_ln = time_ln[:-offset]
            ln = ln[:-offset]
            cycle = cycle[:-offset]

            # only considering the value found if it's still a complete cycle
            if complete_cycle(cycle, deltaH, minLen):

                # then the linear regression is saved
                slope, intercept, r_value, p_value, std_err = linregress(time_ln, ln)

                if print_help:
                    print("slope R value: {}".format(r_value))
                    print("Dataset length: {}".format(len(ln)))
                    print("Slope: {}, date: {}".format(slope, date_string))

                #save in lists
                valid_cycles.append(cycle)
                ln_cycles.append(ln)
                len_cycles.append(len(ln))
                times_ln.append(time_ln)
                
                slopes.append(slope)
                std_errors.append(std_err)
                intercepts.append(intercept)
                date_strings.append(date_string)
            else:
                if print_help: print("Not possible to find a R-value > 0.99 and keep a complete cycle. Discarding cycle from {}, delta too low: {}/{}".format(date_string, max(cycle) - min(cycle), deltaH))

        else:
            if print_help: print("Cycle is not complete. Dropping a cycle of length {} and delta {}".format(len(cycle), max(cycle) - min(cycle)))

    return slopes, std_errors, intercepts, ln_cycles, len_cycles, date_strings, times_ln, valid_cycles;


#########  ########   ##         ##########  #########  ########   
##            ##      ##             ##      ##         ##     ##  
##            ##      ##             ##      ##         ##     ##  
######        ##      ##             ##      ######     ########   
##            ##      ##             ##      ##         ##   ##    
##            ##      ##             ##      ##         ##    ##   
##         ########   #########      ##      #########  ##     ##  

# Now that everything has been processed, look for abnormalities and discard them.
def filter_slopes(slopes, std_errors, intercepts, ln_cycles, len_cycles, date_strings, times_ln, valid_cycles, num_galery, nb_std):
    slopeMean = np.mean(slopes)
    slopeStd = np.std(slopes)

    if print_help: print("Discarding abnormalities. Original mean and std: {}, {}".format(slopeMean, slopeStd))

    # If a slope is further than twice the stdev from the mean, throw it away.
    slopesCpy = slopes[:]
    idxSlopes = 0
    for i in range(len(slopesCpy)):
        if abs(slopeMean - slopesCpy[i]) > nb_std * slopeStd:
            if print_help: print("Slope is too far from the mean. Slope: {} from {}, mean: {}, stdev: {}".format(slopesCpy[i], date_strings[idxSlopes], slopeMean, slopeStd))
            del slopes[idxSlopes]
            del std_errors[idxSlopes]
            del intercepts[idxSlopes]
            del ln_cycles[idxSlopes]
            del len_cycles[idxSlopes]
            del date_strings[idxSlopes]
        else:
            if plot:
                plt.figure(2)
                plt.plot(times_ln[idxSlopes], slopes[idxSlopes] * times_ln[idxSlopes] + intercepts[idxSlopes], linestyle='--', label="Cycle {}".format(str(date_strings[idxSlopes])))
                plt.title("Approximation lineaire du logarithme - puits {}".format(str(num_galery)))
                #plt.ylabel("Hauteur d'eau (cm H2O)")
                plt.xlabel("Temps (min)")
                plt.legend()
                plt.figure(num=3)
                plt.plot(times_ln[idxSlopes], valid_cycles[idxSlopes], label="Cycle {}".format(str(date_strings[idxSlopes])))
                plt.title("Cycles de production d'eau du puits {}".format(str(num_galery)))
                plt.ylabel("Hauteur d'eau (cm H2O)")
                plt.xlabel("Temps (min)")
                plt.axis("tight")
            # If we do not delete an element from the lists,
            # we can increment the index. Otherwise we don't.
            idxSlopes += 1
    return slopes
