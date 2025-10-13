# Normalize value for a column of the dataframe
# values is the list of values we will get data from and normalized
# num_bins is the alpha from the formula --> it also generate the values in the range [0, 1] with num_bins bins
# column_name is the name of the column that we take values list from
# This will return a list of normalized value
def normalize_values(values, num_bins):   
    max_value = values.max()
    min_value = values.min()
    
    if max_value == min_value:
        return values

    return (1 + ((values - min_value) / (max_value - min_value)) * num_bins)**-1


deltaT_max = 19599912.0*1.5


def normalize_deltaT(values, num_bins):   
    global deltaT_max

    if values > deltaT_max:
        deltaT_max = values*1.5

    deltaT_min = 15000
    

    return (1 + ((values - deltaT_min) / (values - deltaT_max)) * num_bins)**-1