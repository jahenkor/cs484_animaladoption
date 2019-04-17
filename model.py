import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def main():


    animal_intake, animal_outcome = loadData()



    return 0

def loadData():

    animal_intake = pd.read_csv("dataset/Austin_Animal_Center_Intakes.csv")
    animal_outcome = pd.read_csv("dataset/Austin_Animal_Center_Outcomes.csv")


    print(animal_outcome)
    print(animal_intake)

    return animal_intake, animal_outcome

main()
