import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import date


#To Do: Visualize table join using SQL

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

    #Train/test split
    trainI, testI = train_test_split(animal_intake, train_size = 0.5,shuffle=False)
    trainO, testO = train_test_split(animal_outcome,train_size=0.5, shuffle=False)

    return animal_intake, animal_outcome

def BreakDates(animal_intake):

# Break dates, include Day of Week, Time
    animal_intake['Date'], animal_intake['Time'] = animal_intake['DateTime'].str.split(' ',1).str
    animal_intake['DayOfWeek'] = animal_intake['Date'].copy()
    animal_intake.drop(['DateTime'],axis=1, inplace=True)


    j = 0
    for i in animal_intake['DayOfWeek']:
        print(j)
        j += 1
        dateSplit = i.split('/')
        #print(dateSplit)
        real_date = date(int(dateSplit[2]),int(dateSplit[0]),int(dateSplit[1]))
        #print(real_date.weekday())
        animal_intake['DayOfWeek'] = real_date.weekday()
    print(animal_intake['DayOfWeek'])


 #       #Remove backslashes from Date
   # for i in range(len(animal_intake['Date'])):
    #    print(i)
     #   animal_intake['Date'].iloc[i] = animal_intake['Date'].iloc[i].replace("/","")



    return animal_intake



#Takes a loooong time
def PreprocessData(animal_outcome,animal_intake):
    print(animal_outcome.columns)
    print(animal_outcome['AnimalID'])
    #ids = animal_outcome['AnimalID']
    #print(animal_outcome[ids.isin(ids[ids.duplicated()])].sort_values("AnimalID"))
    #print(animal_outcome.groupby("AnimalID").count())


    #Add/Remove/update fields

    dropList = ['FoundLocation']
    animal_intake.drop(columns=dropList,inplace=True)

    #Cats,Birds,Dogs only
    animal_intake = animal_intake[animal_intake.AnimalType != "Other"]
    animal_outcome = animal_outcome[animal_outcome.AnimalType != "Other"]


    isAggressive = []
    isMix = []
    for i in range(len(animal_intake['Breed'])):
        print(i)
        if("Pit Bull" in animal_intake['Breed'].iloc[i]):
            isAggressive.append("Yes")
        else:
            isAggressive.append("No")
        if("Mix" in animal_intake['Breed'].iloc[i]):
            isMix.append("Yes")
        else:
            isMix.append("No")
    animal_intake['isAggressive'] = isAggressive
    animal_intake['isMix'] = isMix
    print(animal_intake['isMix'])
    print(animal_intake['isAggressive'])
    exit(0)
    print("removedup intake")
    print(animal_intake)



    #Name Frequency
    '''
    animal_intake['NameFreq'] = np.ones((animal_intake.shape)[0])



    #Aggregate Name Frequency
   # animal_intake['NameFreq'] = animal_intake.groupby('AnimalID')['NameFreq'].count()

    #Gets last entry for animalID
    print("PreDup")
    animal_outcome = animal_outcome.sort_values('DateTime').drop_duplicates('AnimalID',keep='last')
    animal_intake = animal_intake.sort_values('DateTime').drop_duplicates('AnimalID',keep='last')
    print("PostDup  intake")
    print(animal_intake)
    print(len(animal_intake))

    #animal_intake['NameFreq'] = animal_intake.groupby('AnimalID')['NameFreq']
    print(animal_intake['AnimalID'])
    print(animal_outcome['AnimalID'])
    common = pd.concat([animal_intake['AnimalID'],animal_outcome['AnimalID']]).drop_duplicates(keep=False)
    print(common)

    exit(0)

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #print(animal_outcome[:10])
        #print(animal_intake[:10])


    #Join fields
    dataset = pd.merge(animal_outcome,animal_intake,on=["AnimalID","Breed","Color","AnimalType","Name"], how="outer")
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #print(dataset[:10])

    #print(name_freq)
    #dataset['NameFreq'] = name_freq
    #print(dataset)
    #print(dataset.columns)


    BreakDates(animal_outcome)
    BreakDates(animal_intake)









    for i in range(5):

        print(dataset.iloc[i])

    print(dataset)
    animalID = dataset.groupby(['AnimalID'])
    print((animalID.count())['NameFreq'])
    print(dataset.iloc[0])
'''
    #Add/Remove/update fields
    '''

    dropList = ['FoundLocation']
    animal_intake.drop(columns=dropList,inplace=True)

    #Cats,Birds,Dogs only
    animal_intake = animal_intake[animal_intake.AnimalType != "Other"]
    print(animal_intake)
    '''



    #Fix Age (Normalize values to age in days)
    '''
    for i in range(len(animal_intake['AgeuponIntake'])):
        if "months" in animal_intake['AgeuponIntake'].iloc[i] or "month" in animal_intake['AgeuponIntake'].iloc[i]:
            print(i)
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 30.417
            animal_intake['AgeuponIntake'].iloc[i] = str(fixed_time) + " days"
            continue
        if "years" in animal_intake['AgeuponIntake'].iloc[i] or "year" in animal_intake['AgeuponIntake'].iloc[i]:
            print(i)
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 365
            animal_intake['AgeuponIntake'].iloc[i] = str(fixed_time) + " days"
            continue

        if "weeks" in animal_intake['AgeuponIntake'].iloc[i]:
            print(i)
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 7
            animal_intake['AgeuponIntake'].iloc[i] = str(fixed_time) + " days"
            continue

    print(animal_intake['AgeuponIntake'])
'''





#Split sex upon intake as Gender, and intactness
'''
    animal_intake['Gender'], animal_intake['Intactness'] = animal_intake['SexuponIntake'].str.split(' ',1).str
    animal_intake.drop(['SexuponIntake'],axis=1,inplace=True)
    print(animal_intake.columns)
    #End break up SexuponIntake into Gender and Intactness
'''

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
