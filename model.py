import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import date



def main():


    #Load Data
    animal_intake, animal_outcome = LoadData()

    #Preprocessed dataset (Pandas dataframe)
    processed_dataset = PreprocessData(animal_outcome,animal_intake)



    return 0

#Load Data
def LoadData():

    animal_intake = pd.read_csv("dataset/Austin_Animal_Center_Intakes.csv")
    animal_outcome = pd.read_csv("dataset/Austin_Animal_Center_Outcomes.csv")

    #removing spaces in fieldnames so their easier to work with
    animal_intake.columns = [x.replace(' ', '') for x in animal_intake.columns]
    animal_outcome.columns = [x.replace(' ','') for x in animal_outcome.columns]



    #Sort both datasets
    animal_intake = animal_intake.sort_values('AnimalID')
    animal_outcome = animal_outcome.sort_values('AnimalID')


    #Train/test split
    trainI, testI = train_test_split(animal_intake, train_size = 0.01,shuffle=False)
    trainO, testO = train_test_split(animal_outcome,train_size=0.01, shuffle=False)

    return animal_intake, animal_outcome

def BreakDates(animal_intake):

# Break dates, include Day of Week, Time
    animal_intake['Date_Outcomes'], animal_intake['Time_Outcomes'] = animal_intake['DateTime_x'].str.split(' ',1).str
    animal_intake['Date_Intakes'], animal_intake['Time_Intakes'] = animal_intake['DateTime_y'].str.split(' ',1).str

    animal_intake['DayOfWeek_Outcomes'] = animal_intake['Date_Outcomes'].copy()
    animal_intake['DayOfWeek_Intakes'] = animal_intake['Date_Intakes'].copy()

    animal_intake.drop(['DateTime_y'],axis=1, inplace=True)
    animal_intake.drop(['DateTime_x'],axis=1, inplace=True)

    print(animal_intake['Date_Outcomes'])
    print(animal_intake['DayOfWeek_Outcomes'])


    dayOfWeek = []
    j = 0
    for i in animal_intake['DayOfWeek_Outcomes']:
        if i == "nan":
            continue
        j += 1
        dateSplit = i.split('/')
        real_date = date(int(dateSplit[2]),int(dateSplit[0]),int(dateSplit[1]))
        dayOfWeek.append(real_date.weekday())
    animal_intake['DayOfWeek_Outcomes'] = dayOfWeek
    j = 0
    dayOfWeek = []
    for i in animal_intake['DayOfWeek_Intakes']:
        j += 1
        dateSplit = i.split('/')
        #print(dateSplit)
        real_date = date(int(dateSplit[2]),int(dateSplit[0]),int(dateSplit[1]))
        dayOfWeek.append(real_date.weekday())
    animal_intake['DayOfWeek_Intakes'] = dayOfWeek





 #       #Remove backslashes from Date
   # for i in range(len(animal_intake['Date'])):
    #    print(i)
     #   animal_intake['Date'].iloc[i] = animal_intake['Date'].iloc[i].replace("/","")



    return animal_intake



#V1: Takes a loooong time
#V2: Gooooes faster now
def PreprocessData(animal_outcome,animal_intake):


    #Cats,Birds,Dogs only
    animal_intake = animal_intake[animal_intake.AnimalType != "Other"]
    animal_outcome = animal_outcome[animal_outcome.AnimalType != "Other"]

   #Name Field
    #Boolean

    animal_intake['Name'].fillna('No')
    names = []
    for i in range(len(animal_intake['Name'])):
        print("Changing Name: %d" % i)
        if animal_intake['Name'].iloc[i] != 'No':
            names.append("Yes")
        else:
            names.append("No")
    animal_intake['Name'] = names



    #End Name conversion




    #Add/Remove/update fields





    dropListIntake = ['FoundLocation','MonthYear']
    #remove redundant features from outcome dataset
    dropListOutcome = ['DateofBirth','MonthYear','OutcomeSubtype','Name','AnimalType',"Color","Breed"]
    animal_intake = animal_intake.drop(columns=dropListIntake)
    animal_outcome = animal_outcome.drop(columns=dropListOutcome)
    animal_outcome.dropna(inplace=True)
    animal_intake.dropna(inplace=True)

    #Remove missing values
    animal_intake.dropna(inplace=True)
    animal_outcome.dropna(inplace=True)




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


    #Name Frequency
    animal_intake['NameFreq'] = np.ones((animal_intake.shape)[0])


    #Aggregate Name Frequency
    name_freq = animal_intake.groupby('AnimalID')['NameFreq'].count()
    print(name_freq)


    #Gets last entry for animalID
    animal_outcome = animal_outcome.sort_values('DateTime').drop_duplicates('AnimalID',keep='last')
    animal_intake = animal_intake.sort_values('DateTime').drop_duplicates('AnimalID',keep='last')
    animal_intake = animal_intake.merge(name_freq,on=['AnimalID'],how='right')
    animal_intake = animal_intake.rename(index=str, columns={"NameFreq_y":"NumberOfAdmits"})
    animal_intake
    animal_intake = animal_intake.drop(['NameFreq_x'],axis=1)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(animal_intake[:10])


    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #print(animal_outcome[:10])
        #print(animal_intake[:10])

    #Join fields
    dataset = pd.merge(animal_outcome,animal_intake,on=["AnimalID"], how="right").drop_duplicates()
    #dataset = pd.merge(dataset, name_freq, on=["AnimalID"], how="inner")


    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dataset[:10])
    dataset.dropna(inplace=True)






    BreakDates(dataset)
#    dataset = dataset.sort_values('Date_x').drop_duplicates('AnimalID',keep='last')




    #dataset = dataset[['AnimalID','DateTime_x']].drop_duplicates(keep='last'))
    #print(dataset[['AnimalID','DateTime_y']].drop_duplicates(keep='last'))




    #dataset[['AnimalID','Date_x','Time_x']] = dataset[['AnimalID','Date_x','Time_x']].drop_duplicates(keep='last')


    #dataset[['AnimalID','Date_y','Time_y']] = dataset[['AnimalID','Date_y','Time_y']].drop_duplicates(keep='last')

    #dataset.dropna(inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dataset[:10])



#    print(dataset['MonthYear_x'].drop_duplicates(keep='last'))
    #print(name_freq)
    #dataset['NameFreq'] = name_freq
    #print(dataset)
    #print(dataset.columns)





#To fit below code
    animal_intake = dataset






    #Fix Age (Normalize values to age in days)
    #Could make neater by using reg expressions
    ageUponIntake = []
    for i in range(len(animal_intake['AgeuponIntake'])):
        if "months" in animal_intake['AgeuponIntake'].iloc[i] or "month" in animal_intake['AgeuponIntake'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 30.417
            ageUponIntake.append(str(fixed_time) + " days")
            continue
        if "years" in animal_intake['AgeuponIntake'].iloc[i] or "year" in animal_intake['AgeuponIntake'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 365
            ageUponIntake.append(str(fixed_time) + " days")
            continue

        if "weeks" in animal_intake['AgeuponIntake'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 7
            ageUponIntake.append(str(fixed_time) + " days")
            continue

        if "week" in animal_intake['AgeuponIntake'].iloc[i]:
            fixed_time = int((animal_intake['AgeuponIntake'].iloc[i])[0]) * 7
            ageUponIntake.append(str(fixed_time) + " days")
            continue

        if "days" in animal_intake['AgeuponIntake'].iloc[i]:
            ageUponIntake.append(animal_intake['AgeuponIntake'].iloc[i])
            continue

        if "day" in animal_intake['AgeuponIntake'].iloc[i]:
            ageUponIntake.append(animal_intake['AgeuponIntake'].iloc[i])
            continue

        print(animal_intake['AgeuponIntake'].iloc[i])







    animal_intake['AgeuponIntake'] = ageUponIntake
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dataset[:10])








#Split sex upon intake as Gender, and intactness
    animal_intake['Intactness'], animal_intake['Gender'] = animal_intake['SexuponIntake'].str.split(' ',1).str
    animal_intake.drop(['SexuponIntake'],axis=1,inplace=True)
    #End break up SexuponIntake into Gender and Intactness


    #Map down colors into color_list
    colorset=[]
    colorsForList = []
    color_list = {'Brown':'Brown','White':'White','Red':'Red','Blue':'Blue','Black':'Black','Orange':'Orange','Yellow':'Yellow','Tan':'Brown','Tortie':"Brown/Black",'Tricolor':'Mixed','Chocolate':'Brown',"Calico":"Brown/White","Gold":'Yellow','Cream':'Brown','Gray':'Black','Gray':'Gray','Agouti':'Brown','Apricot':'Orange','Buff':'Tan','Fawn':'Tan','Flame Point':'White','Green':'Green','Silver':'Gray','Seal':'Black/White','Pink':'Red','Lilac Point':'White','Liver':'Red','Lynx Point':'Black/White','Sable':'Black','Torbie':'Gray'}

    #for i in animal_intake['Color']:
     #   if i in colorset:
      #      continue
       # else:
#            colorset.append(i)

    colors = animal_intake['Color']

    nope=[]
    print(len(colors))
    for i in range(len(colors)):
        print(i)

        if any(elem in colors.iloc[i] for elem in color_list):
            #Choose a unique color for element
            colorsChosen = False
            for x,y in color_list.items():
                if x in colors.iloc[i]:
                    if(not(colorsChosen)):
                        colorsForList.append(y)
                        colorsChosen=True

        else:
            print("nope %s" % colors.iloc[i])
            nope.append(colors.iloc[i])
            colorsForList.append("Other")


    animal_intake['Color'] = colorsForList
    #End change colors snippet






    #Map down Breeds into breed_list - Lots of breeds!
    breedsForList = []
    breed_list = {'Pit Bull':'Working','German Shepherd':'Herding','Hound':'Hound','Chihuahua':'Companion','Domestic Shorthair':'Common','Labrador':'Sporting'}
    breeds = animal_intake['Breed']

    nope=[]
    for i in range(len(breeds)):
        print(i)

        if any(elem in breeds.iloc[i] for elem in breed_list):
            #Choose a unique color for element
            breedsChosen = False
            for x,y in breed_list.items():
                if x in breeds.iloc[i]:
                    if(not(breedsChosen)):
                        breedsForList.append(y)
                        breedsChosen=True

        else:
            print("nope %s" % breeds.iloc[i])
            nope.append(breeds.iloc[i])
            breedsForList.append("Other")

    print("nope breeds %s"%nope)

    animal_intake['Breed'] = breedsForList
    print(animal_intake.Breed.unique())
    #End change colors snippet




    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(animal_intake[:10])

    return animal_intake



def PredictOutcome():

    RandomTreesClassifier

    return 0



################### HELPER FUNCTIONS and Data Structures########################################

main()
