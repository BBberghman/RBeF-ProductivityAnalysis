# Riverbed Filtration System 
## Automatic computation of the productivity coefficient

This program computes automaticaly the value of the productivity coefficient for a dataset given. This is done by two methods: 
- numerically, directly by solving the equation
- using the properties of the characteristic time of a first-order system

More information can be found in "Berghmen Erica, Analyse et modélisation d'un système de production d'eau potable par filtration sous lit de rivière, Juin 2018."

## How to use it?
### Data
The data to analyze should be saved in a CSV containing the number of the galery.
This CSV should contain 3 columns :
- The timestamp in one of these 2 formats : d-m-y H:M, Y/m/d H:M:S
- The water height above the sensor
- The atmospheric pressure

A data should be given for each of these columns and for each line. No cell can be empty. If no data of the atmospheric pressure exists, it is recommended to do a linear approximation between the data known. It is not automaticaly done by this program.

### Parameters
You should at least verify one parameter depending on the value you have chosen in the datalogger.
- STEP_TIME: time between each of your sample in min (not been tested for < 1)

Some parameters have been chosen, you can change them:
- NB_MERGE : Used when concatenating two cycles. It represents the maximum number of elements in the cycle between the 2 concatenated.
- EPS: EPS value added when calculating the ln to avoid ln(0)
- NB_MAX: Number of elements to calculate the max of the cycle (mean of the NB_MAX last elements)
- NB_MIN: Number of elements to calculate the min of the cycle (mean of the NB_MIN first elements)
- NB_STD: Number of standard deviation beyond which a cycle is discarded

For each galery, two other parameters have been chosen as well:
- MINLEN: Minimal length of the cycle to consider the cycle
- DELTA: Minimal water height difference between the min and the max of the cycle to consider the cycle

### Run
This program has been coded under Python 2.7. 
To run it, just add your CSV files in your repository and launch the script. Change the name of the input file in the script such that it corresponds to your CSV file.

### Output
There is 4 outputs parameters :
- plot: if active, 3 plots are generated: the whole dataset, the linear approximation of the log of the cycle considered and the cycles considered.
- print_data: output the values of the slopes for both methods in terminal
- print_help: some help to understand what is going on in the program (i.e. which cycles it is discarding)
- write_csv: write the data (values of the slopes) in a CSV file


## Troubleshooting
### DecodeError
In case of error: "DecodeError: 'ascii' codec can't decode byte", make sure there is no special character or accent in the path.