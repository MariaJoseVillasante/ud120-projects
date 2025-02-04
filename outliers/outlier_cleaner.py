#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    errors = abs(predictions - net_worths)
    data = list(zip(ages,net_worths,errors))
    data.sort(key=lambda tup: tup[2])
    cleaned_data = data[0:80]#[age,net_worth,error in ages,net_worths,errors]


    
    return cleaned_data

