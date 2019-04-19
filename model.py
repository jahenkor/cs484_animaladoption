import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def main():


    animal_intake, animal_outcome = LoadData()
    PreprocessData(animal_outcome,animal_intake)



    return 0

def LoadData():

    animal_intake = pd.read_csv("dataset/Austin_Animal_Center_Intakes.csv")
    animal_outcome = pd.read_csv("dataset/Austin_Animal_Center_Outcomes.csv")

    #removing spaces in fieldnames so their easier to work with
    animal_intake.columns = [x.replace(' ', '') for x in animal_intake.columns]
    animal_outcome.columns = [x.replace(' ','') for x in animal_outcome.columns]


    print(animal_outcome)
    print(animal_intake)
    print(animal_intake.columns)

    trainI, testI = train_test_split(animal_intake, train_size = 0.1,shuffle=False)

    return trainI, animal_outcome

def PreprocessData(animal_outcome,animal_intake):

    #Add/Remove/update fields


    dropList = ['FoundLocation']
    animal_intake.drop(columns=dropList,inplace=True)

    #Cats,Birds,Dogs only
    animal_intake = animal_intake[animal_intake.AnimalType != "Other"]
    print(animal_intake)



    #Name Field
    #Boolean
    '''
    animal_intake['Name'].dropna(axis=0, how='any', inplace=True)
    animal_intake['Name'].fillna('No')
    for i in range(len(animal_intake['Name'])):
        print(i)
        if animal_intake['Name'].iloc[i] != 'No':
            animal_intake['Name'].iloc[i] = 'Yes'

    print(animal_intake['Name'])
    #End Name conversion
'''


    #Map down colors into color_list
'''    colorset=[]
    color_list = ['Brown','White','Red','Blue','Black','Orange','Yellow']
    print("Pre-Update color column %s "% animal_intake['Color'])
    for i in animal_intake['Color']:
        if i in colorset:
            continue
        else:
            colorset.append(i)

    colors = animal_intake['Color']
    print(len(colors))
    for i in range(len(colors)):
        print(i)

        if any(elem in colors.iloc[i] for elem in color_list):
            for j in color_list:
                if j in colors.iloc[i]:
                    colors.iloc[i] = j
        #else:
            #print("nope %s" % colors.iloc[i])
    print("Updated color column %s" % colors)
    #End change colors snippet
'''







    #Join Tables by AnimalID



main()
