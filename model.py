import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import date, datetime
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
import os
import math


#Remove file path from disk, to make new processed dataset
PROC_DATASET= 'dataset/procdataset.csv'

def main():

    #Check if preprocessed dataset exists, if not perform preprocessing steps
    #Don't waste time on preproc step
    processed_dataset = 0
    exists = os.path.isfile(PROC_DATASET)
    if not(exists):

        # Load Data
        animal_intake, animal_outcome = LoadData(False)
        # Preprocessed dataset (Pandas dataframe)
        processed_dataset = PreprocessData(animal_outcome, animal_intake)
        print(processed_dataset)
        SaveProcDatasetToDisk(processed_dataset)
    else:
        processed_dataset = LoadData(True)


    features = np.array(processed_dataset.columns.values)
    features = np.delete(features, 0, 0)

    '''
    for x in features:
        print(x)
        printGraphics(processed_dataset,x)
'''

    train, test = train_test_split(processed_dataset, train_size =0.33, shuffle=False)

    return 0


# Load Data
def LoadData(isProcessed):
    if(isProcessed):
        animal_intake = pd.read_csv(PROC_DATASET)
        animal_intake.columns = [x.replace(' ', '') for x in animal_intake.columns]
        animal_intake = animal_intake.sort_values('AnimalID')

        return animal_intake


    animal_intake = pd.read_csv("dataset/Austin_Animal_Center_Intakes.csv")
    animal_outcome = pd.read_csv("dataset/Austin_Animal_Center_Outcomes.csv")

    # removing spaces in fieldnames so their easier to work with
    animal_intake.columns = [x.replace(' ', '') for x in animal_intake.columns]
    animal_outcome.columns = [x.replace(' ', '') for x in animal_outcome.columns]

    # Sort both datasets
    animal_intake = animal_intake.sort_values('AnimalID')
    animal_outcome = animal_outcome.sort_values('AnimalID')


    return animal_intake, animal_outcome


def BreakDates(animal_intake):
    fmt = '%m/%d/%Y %I:%M:%S %p'
    # Break dates, include Day of Week, Time
    animal_intake['Date_Outcomes'], animal_intake['Time_Outcomes'] = animal_intake['DateTime_x'].str.split(' ', 1).str
    animal_intake['Date_Intakes'], animal_intake['Time_Intakes'] = animal_intake['DateTime_y'].str.split(' ', 1).str

    animal_intake['Time_Intakes'], time1, time2 = animal_intake['Time_Intakes'].str.split(':', 3).str
    animal_intake['Time_Outcomes'], time1, time2 = animal_intake['Time_Outcomes'].str.split(':', 3).str

    animal_intake['DayOfWeek_Outcomes'] = animal_intake['Date_Outcomes'].copy()
    animal_intake['DayOfWeek_Intakes'] = animal_intake['Date_Intakes'].copy()

    print(datetime.strptime(animal_intake['DateTime_y'].iloc[1],fmt))

    LengthOfStay = []
    k = 0
    for i in animal_intake['DateTime_y']:
        d1 = datetime.strptime(i,fmt)
        d2 = datetime.strptime(animal_intake['DateTime_x'].iloc[k],fmt)
        diff = (d1-d2).days
        minutesDiff = diff*24*60
        LengthOfStay.append(math.sqrt(minutesDiff**2)) #L2 norm
        k+=1
    animal_intake['LengthOfStay'] = LengthOfStay


    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(animal_intake[:10])


#    exit(0)

    animal_intake.drop(['DateTime_y'], axis=1, inplace=True)
    animal_intake.drop(['DateTime_x'], axis=1, inplace=True)
    animal_intake.drop(['Date_Outcomes'], axis=1, inplace=True)
    animal_intake.drop(['Date_Intakes'], axis=1, inplace=True)

    print(animal_intake['DayOfWeek_Outcomes'])


    dayOfWeek = []
    j = 0
    for i in animal_intake['DayOfWeek_Outcomes']:
        if i == "nan":
            continue
        j += 1
        dateSplit = i.split('/')
        real_date = date(int(dateSplit[2]), int(dateSplit[0]), int(dateSplit[1]))
        dayOfWeek.append(real_date.weekday())
    animal_intake['DayOfWeek_Outcomes'] = dayOfWeek
    j = 0
    dayOfWeek = []
    for i in animal_intake['DayOfWeek_Intakes']:
        j += 1
        dateSplit = i.split('/')
        # print(dateSplit)
        real_date = date(int(dateSplit[2]), int(dateSplit[0]), int(dateSplit[1]))
        dayOfWeek.append(real_date.weekday())
    animal_intake['DayOfWeek_Intakes'] = dayOfWeek
    exit(0)

    #       #Remove backslashes from Date
    # for i in range(len(animal_intake['Date'])):
    #    print(i)
    #   animal_intake['Date'].iloc[i] = animal_intake['Date'].iloc[i].replace("/","")

    return animal_intake


# V1: Takes a loooong time
# V2: Gooooes faster now
def PreprocessData(animal_outcome, animal_intake):
    # Cats,Birds,Dogs only
    animal_intake = animal_intake[animal_intake.AnimalType != "Other"]
    animal_outcome = animal_outcome[animal_outcome.AnimalType != "Other"]

    # Name Field
    # Boolean

    animal_intake['Name'].fillna('No')
    names = []
    for i in range(len(animal_intake['Name'])):
        print("Changing Name: %d" % i)
        if animal_intake['Name'].iloc[i] != 'No':
            names.append("Yes")
        else:
            names.append("No")
    animal_intake['Name'] = names

    # End Name conversion

    # Add/Remove/update fields

    dropListIntake = ['FoundLocation', 'MonthYear']
    # remove redundant features from outcome dataset
    dropListOutcome = ['DateofBirth', 'MonthYear', 'OutcomeSubtype', 'Name', 'AnimalType', "Color", "Breed"]
    animal_intake = animal_intake.drop(columns=dropListIntake)
    animal_outcome = animal_outcome.drop(columns=dropListOutcome)
    animal_outcome = animal_outcome.dropna(inplace=False)
    animal_intake = animal_intake.dropna(inplace=False)

    # Remove missing values
    animal_intake = animal_intake.dropna(inplace=False)
    animal_outcome = animal_outcome.dropna(inplace=False)

    isAggressive = []
    Mixed = []
    for i in range(len(animal_intake['Breed'])):
        print(i)
        if ("Pit Bull" in animal_intake['Breed'].iloc[i]):
            isAggressive.append("Yes")
        else:
            isAggressive.append("No")
        if ("Mix" in animal_intake['Breed'].iloc[i]):
            Mixed.append("Yes")
        else:
            Mixed.append("No")
    animal_intake['Aggressive'] = isAggressive
    animal_intake['Mixed'] = Mixed
    print(animal_intake['Mixed'])
    print(animal_intake['Aggressive'])
    animal_intake = animal_intake.drop(['Breed'], axis=1, inplace=False)

    # Name Frequency
    animal_intake['NameRecurrence'] = np.ones((animal_intake.shape)[0])

    # Aggregate Name Frequency
    name_rec = animal_intake.groupby('AnimalID')['NameRecurrence'].count()
    print(name_rec)

    # Gets last entry for animalID
    animal_outcome = animal_outcome.sort_values('DateTime').drop_duplicates('AnimalID', keep='last')
    animal_intake = animal_intake.sort_values('DateTime').drop_duplicates('AnimalID', keep='last')
    animal_intake = animal_intake.merge(name_rec, on=['AnimalID'], how='right')
    animal_intake = animal_intake.rename(index=str, columns={"NameRecurrence_y": "NumberOfAdmits"})
    animal_intake = animal_intake.drop(['NameRecurrence_x'], axis=1)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(animal_intake[:10])

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    # print(animal_outcome[:10])
    # print(animal_intake[:10])

    # Join fields
    dataset = pd.merge(animal_outcome, animal_intake, on=["AnimalID"], how="right").drop_duplicates()
    # dataset = pd.merge(dataset, name_freq, on=["AnimalID"], how="inner")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dataset[:10])
    dataset = dataset.dropna(inplace=False)

    BreakDates(dataset)
    #    dataset = dataset.sort_values('Date_x').drop_duplicates('AnimalID',keep='last')

    # dataset = dataset[['AnimalID','DateTime_x']].drop_duplicates(keep='last'))
    # print(dataset[['AnimalID','DateTime_y']].drop_duplicates(keep='last'))

    # dataset[['AnimalID','Date_x','Time_x']] = dataset[['AnimalID','Date_x','Time_x']].drop_duplicates(keep='last')

    # dataset[['AnimalID','Date_y','Time_y']] = dataset[['AnimalID','Date_y','Time_y']].drop_duplicates(keep='last')

    # dataset.dropna(inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dataset[:10])

    #    print(dataset['MonthYear_x'].drop_duplicates(keep='last'))
    # print(name_freq)
    # dataset['NameFreq'] = name_freq
    # print(dataset)
    # print(dataset.columns)

    # To fit below code
    animal_intake = dataset

    # Fix Age (Normalize values to age in days)
    # Could make neater by using reg expressions
    ageUponIntake = []
    for i in range(len(animal_intake['AgeuponIntake'])):
        if "months" in animal_intake['AgeuponIntake'].iloc[i] or "month" in animal_intake['AgeuponIntake'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 30.417
            ageUponIntake.append(str(fixed_time))
            continue
        if "years" in animal_intake['AgeuponIntake'].iloc[i] or "year" in animal_intake['AgeuponIntake'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 365
            ageUponIntake.append(str(fixed_time))
            continue

        if "weeks" in animal_intake['AgeuponIntake'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 7
            ageUponIntake.append(str(fixed_time))
            continue

        if "week" in animal_intake['AgeuponIntake'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 7
            ageUponIntake.append(str(fixed_time))
            continue

        if "days" in animal_intake['AgeuponIntake'].iloc[i]:
            ageUponIntake.append(animal_intake['AgeuponIntake'].iloc[i])
            continue

        if "day" in animal_intake['AgeuponIntake'].iloc[i]:
            ageUponIntake.append(animal_intake['AgeuponIntake'].iloc[i])
            continue

    ageUponOutcome = []
    for i in range(len(animal_intake['AgeuponOutcome'])):

        if "months" in animal_intake['AgeuponOutcome'].iloc[i] or "month" in animal_intake['AgeuponOutcome'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponOutcome'].iloc[i])[0]) * 30.417
            ageUponOutcome.append(str(fixed_time))
            continue
        if "years" in animal_intake['AgeuponOutcome'].iloc[i] or "year" in animal_intake['AgeuponOutcome'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponOutcome'].iloc[i])[0]) * 365
            ageUponOutcome.append(str(fixed_time))
            continue

        if "weeks" in animal_intake['AgeuponOutcome'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponOutcome'].iloc[i])[0]) * 7
            ageUponOutcome.append(str(fixed_time))
            continue

        if "week" in animal_intake['AgeuponOutcome'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponOutcome'].iloc[i])[0]) * 7
            ageUponOutcome.append(str(fixed_time))
            continue

        if "days" in animal_intake['AgeuponOutcome'].iloc[i]:
            ageUponOutcome.append(animal_intake['AgeuponOutcome'].iloc[i])
            continue

        if "day" in animal_intake['AgeuponOutcome'].iloc[i]:
            ageUponOutcome.append(animal_intake['AgeuponOutcome'].iloc[i])
            continue

    animal_intake['AgeuponOutcome'] = ageUponOutcome
    animal_intake['AgeuponIntake'] = ageUponIntake
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dataset[:10])

    # Split sex upon intake as Gender, and intactness
    animal_intake['Intactness'], animal_intake['Gender'] = animal_intake['SexuponIntake'].str.split(' ', 1).str
    animal_intake.drop(['SexuponIntake'], axis=1, inplace=True)
    # End break up SexuponIntake into Gender and Intactness

    # Map down colors into color_list
    colorset = []
    colorsForList = []
    color_list = {'Brown': 'Brown', 'White': 'White', 'Red': 'Red', 'Blue': 'Blue', 'Black': 'Black',
                  'Orange': 'Orange', 'Yellow': 'Yellow', 'Tan': 'Brown', 'Tortie': "Brown/Black", 'Tricolor': 'Mixed',
                  'Chocolate': 'Brown', "Calico": "Brown/White", "Gold": 'Yellow', 'Cream': 'Brown', 'Gray': 'Black',
                  'Gray': 'Gray', 'Agouti': 'Brown', 'Apricot': 'Orange', 'Buff': 'Tan', 'Fawn': 'Tan',
                  'Flame Point': 'White', 'Green': 'Green', 'Silver': 'Gray', 'Seal': 'Black/White', 'Pink': 'Red',
                  'Lilac Point': 'White', 'Liver': 'Red', 'Lynx Point': 'Black/White', 'Sable': 'Black',
                  'Torbie': 'Gray'}

    # for i in animal_intake['Color']:
    #   if i in colorset:
    #      continue
    # else:
    #            colorset.append(i)

    colors = animal_intake['Color']

    nope = []
    print(len(colors))
    for i in range(len(colors)):
        print(i)

        if any(elem in colors.iloc[i] for elem in color_list):
            # Choose a unique color for element
            colorsChosen = False
            for x, y in color_list.items():
                if x in colors.iloc[i]:
                    if (not (colorsChosen)):
                        colorsForList.append(y)
                        colorsChosen = True

        else:
            print("nope %s" % colors.iloc[i])
            nope.append(colors.iloc[i])
            colorsForList.append("Other")

    animal_intake['Color'] = colorsForList
    # End change colors snippet

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(animal_intake[:10])



    return animal_intake

def SaveProcDatasetToDisk(dataset):
    dataset.to_csv(path_or_buf=PROC_DATASET)
    return 0

def PredictOutcome():
    RandomTreesClassifier

    return 0


def RandForestRegr(train, labels,test):


    regr = RandomForestRegressor(max_depth = 2, n_estimators=100)
    regr.fit(train,labels)
    print("Regr feature importances")
    print(regr.feature_importances_)
    print(regr.predict(test))

    return 0


def printGraphics(animal_dataset, feature):
    ScanFeature = "OutcomeType"
    OutcomeTypes = animal_dataset.OutcomeType.unique()
    values = np.zeros(OutcomeTypes.shape)
    for i in animal_dataset[ScanFeature]:
        outcome = np.where(OutcomeTypes == i)
        print(outcome[0])
        print(i)
        print(values[outcome])
        values[outcome] += 1;

    print(OutcomeTypes)
    print(values)

    y = np.arange(len(OutcomeTypes))


    plt.barh(y, values, align='center', alpha=0.5)
    plt.yticks(y, OutcomeTypes)
    plt.xlabel('Number of Animals')
    plt.title('OutcomeType v Count')

    plt.show()

    return 0;

main()
