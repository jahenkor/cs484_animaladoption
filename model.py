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

    return animal_intake, animal_outcome

def PreprocessData(animal_outcome,animal_intake):

    #Add/Remove/update fields


    dropList = ['FoundLocation']
    animal_intake.drop(columns=dropList,inplace=True)
    print(animal_intake)



    #Join Tables by AnimalID



main()
