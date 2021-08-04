#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

print('How many data points (people) are in the dataset?:',len(enron_data))

#features = enron_data.values()
#p = len(features)
p = len(enron_data["SKILLING JEFFREY K"])
print('For each person, how many features are available?', p)#[""][""])

#In other words, count the number of entries in the dictionary where
#data[person_name]["poi"]==1
pois = [person_name for person_name, key in enron_data.items() if enron_data[person_name]["poi"]==1]
print('How many POIs are there in the E+F dataset?:', len(pois))

#totpois = joblib.load(open("../final_project/poi_names.txt", "rb"))
#print('How many POI’s were there total?', totpois)

JP = enron_data["PRENTICE JAMES"]["total_stock_value"]
print('What is the total value of the stock belonging to James Prentice?', JP)

fWCtp = enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print('How many email messages do we have from Wesley Colwell to persons of interest?', fWCtp)

vosJKS = enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print('What’s the value of stock options exercised by Jeffrey K Skilling?', vosJKS)

money1 = enron_data["SKILLING JEFFREY K"]["total_payments"]
money2 = enron_data["LAY KENNETH L"]["total_payments"]
money3 = enron_data["FASTOW ANDREW S"]["total_payments"]
print('Skilling took {}, Lay took {}, Fastow took {}'.format(money1,money2,money3))

sal = [person_name for person_name, key in enron_data.items() if enron_data[person_name]["salary"]!='NaN']
ema = [person_name for person_name, key in enron_data.items() if enron_data[person_name]["email_address"]!='NaN']

print('How many folks in this dataset have a quantified salary?', len(sal))
print('What about a known email address?', len(ema))
'''
How many data points (people) are in the dataset?: 146
For each person, how many features are available? 21
How many POIs are there in the E+F dataset?: 18
How many POI’s were there total? 35
What is the total value of the stock belonging to James Prentice? 1095040
How many email messages do we have from Wesley Colwell to persons of interest? 11
What’s the value of stock options exercised by Jeffrey K Skilling? 19250000
Of these three individuals (Lay, Skilling and Fastow),
...who took home the most money (largest value of “total_payments” feature)?
Skilling took 8682716, Lay took 103559793, Fastow took 2424083
How many folks in this dataset have a quantified salary? 95
What about a known email address? 111
'''
