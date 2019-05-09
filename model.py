import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from datetime import date, datetime
from sklearn.metrics import r2_score, accuracy_score, log_loss
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from statistics import mean
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
import math
import time

#Remove file path from disk, to make new processed dataset
PROC_DATASET= 'dataset/procdataset.csv'

def main():

    #Check if preprocessed dataset exists, if not perform preprocessing steps
    #Don't waste time on preproc step
    processed_dataset = 0
    exists = os.path.isfile(PROC_DATASET)
    print(exists)
    if not(exists):

        # Load Data
        animal_intake, animal_outcome = LoadData(False)
        # Preprocessed dataset (Pandas dataframe)
        processed_dataset = PreprocessData(animal_outcome, animal_intake)
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
    
    #takes a while to run, but prints elbow plot for finding best k
    #FindBestk(processed_dataset)
    
    #print(processed_dataset.head())
    print(processed_dataset['Gender'].unique())
    #second value is k value for knn 
    PredictClassification(processed_dataset, 4)
    
    #print(train.columns)
    #print(train.head())
    #RandForestRegr(train,test)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #    print(processed_dataset[:10])



    return 0


# Load Data
def LoadData(isProcessed):
    if(isProcessed):
        animal_intake = pd.read_csv(PROC_DATASET)
        animal_intake.columns = [x.replace(' ', '') for x in animal_intake.columns]

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
            ageUponIntake.append(animal_intake['AgeuponIntake'].iloc[i][0])
            continue

        if "day" in animal_intake['AgeuponIntake'].iloc[i]:
            ageUponIntake.append(animal_intake['AgeuponIntake'].iloc[i][0])
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
            ageUponOutcome.append(animal_intake['AgeuponOutcome'].iloc[i][0])
            continue

        if "day" in animal_intake['AgeuponOutcome'].iloc[i]:
            ageUponOutcome.append(animal_intake['AgeuponOutcome'].iloc[i][0])
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

    #Remove Animal ID
    animal_intake = animal_intake.drop(columns=['AnimalID'])

    #Change categorical to numerical values
    columnsEncode = ['OutcomeType','SexuponOutcome','Name','IntakeType','IntakeCondition','AnimalType', 'Color', 'Aggressive', 'Mixed', 'Intactness', 'Gender']
    le = LabelEncoder()
    animal_intake = animal_intake.dropna(inplace=False)


    #animal_intake[columnsEncode].apply(le.fit_transform))
    map = []
    for i in columnsEncode:
        animal_intake[i] = animal_intake.apply(le.fit_transform)
    #Encode each categorical
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(animal_intake[:10])



    return animal_intake

def SaveProcDatasetToDisk(dataset):
    dataset.to_csv(path_or_buf=PROC_DATASET,index=False)
    return 0

def PredictOutcome():
    RandomTreesClassifier

    return 0


def RandForestRegr(train,test):


    labels = train['LengthOfStay'].copy(deep=True)

    test = test.drop(columns=['LengthOfStay'])

    train = train.drop(columns=['LengthOfStay'])

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #    print(train[:10])



    regr = RandomForestRegressor(max_depth = 15, n_estimators=100)
    regr.fit(train,labels)
    #print("Regr feature importances")
    #print(regr.feature_importances_)
    #print(labels)
    #print(regr.predict(test))

    return 0

# ----- CLASSIFICATION -----#
 
def FindBestk(dataset):
    #reference: https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
    mms = MinMaxScaler()
    mms.fit(dataset)
    data_transformed = mms.transform(dataset)
    squared_dist = []
    K = range(1,31)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_transformed)
        squared_dist.append(km.inertia_)
    plt.plot(K, squared_dist, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum squared distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def PredictClassification(dataset, k):
    #reference: https://machinelearningmastery.com/k-fold-cross-validation/
    #for basic Kfold cross validation
    labels = dataset['OutcomeType'].copy(deep=True)
    features = dataset.drop(columns=['OutcomeType'])
    
    model1 = GaussianNB()
    model2 = KNeighborsClassifier(n_neighbors = 4)
    model3 = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
    model4 = AdaBoostClassifier(n_estimators=200, learning_rate=1.5)
    estimators = [('GB', model1), ('KNN', model2), ('DT', model3)] #for voting classifier
    
    kf = KFold(n_splits=10, shuffle = True)
    
    whole_time = time.time()
    NaiveBayesClass(labels, features, kf)
    KNNClass(labels, features, kf, k)
    DecisionTreeClass(labels, features, kf)
    Adaboost(labels, features, kf)
    VotingClass(labels, features, kf, estimators)
    RandomForestClass(labels, features, kf)
    OneVRest(labels, features, kf, k)
    print("TOTAL TIME: ", time.time() - whole_time)
    

def NaiveBayesClass(labels, features, kf):  
    nb = GaussianNB()
    accuracy_scores = []
    log_loss_scores = []
    start = time.time()
    for train, test in kf.split(features):
         X_train, X_test = features.iloc[train], features.iloc[test]
         y_train, y_test = labels.iloc[train], labels.iloc[test]
         nb.fit(X_train, y_train)
         
         pred_logloss = nb.predict_proba(X_test)
         log_loss_scores.append(log_loss(y_test, pred_logloss, labels = labels))
         
         pred_acc = nb.predict(X_test)
         accuracy_scores.append(accuracy_score(y_test, pred_acc))
         
    print("NAIVE BAYES MEAN LOG LOSS: ", mean(log_loss_scores))
    print("NAIVE BAYES MEAN ACCURACY: %", mean(accuracy_scores)*100)
    print("NB TIME: ", time.time() - start, "\n")
    
def KNNClass(labels, features, kf, k):
    knn = KNeighborsClassifier(n_neighbors = k)
    accuracy_scores = []
    log_loss_scores = []
    start = time.time()
    for train, test in kf.split(features):
         X_train, X_test = features.iloc[train], features.iloc[test]
         y_train, y_test = labels.iloc[train], labels.iloc[test]
         knn.fit(X_train, y_train)
         
         pred_logloss = knn.predict_proba(X_test)
         log_loss_scores.append(log_loss(y_test, pred_logloss, labels = labels))
         
         pred_acc = knn.predict(X_test)
         accuracy_scores.append(accuracy_score(y_test, pred_acc))
         
    print("K VALUE: ", k)
    print("KNN MEAN LOG LOSS: ", mean(log_loss_scores))
    print("KNN MEAN ACCURACY: %", mean(accuracy_scores)*100)
    print("KNN TIME: ", time.time() - start, "\n")

def DecisionTreeClass(labels, features, kf):  #ENTROPY > GINI
    crit = "entropy"
    depth = 2
    tree = DecisionTreeClassifier(criterion = crit, max_depth = depth)
    accuracy_scores = []
    log_loss_scores = []
    start = time.time()
    for train, test in kf.split(features):
         X_train, X_test = features.iloc[train], features.iloc[test]
         y_train, y_test = labels.iloc[train], labels.iloc[test]
         tree.fit(X_train, y_train)
         
         pred_logloss = tree.predict_proba(X_test)
         log_loss_scores.append(log_loss(y_test, pred_logloss, labels = labels))
         
         pred_acc = tree.predict(X_test)
         accuracy_scores.append(accuracy_score(y_test, pred_acc))
         
    print("DECISION TREE CRITERION: ", crit)
    print("DECISION TREE MAX DEPTH: ", depth)
    print("DECISION TREE MEAN LOG LOSS: ", mean(log_loss_scores))
    print("DECISION TREE MEAN ACCURACY: %", mean(accuracy_scores)*100)
    print("DECISION TREE TIME: ", time.time() - start, "\n")
    
def Adaboost(labels, features, kf): #ESTIMATOR AND LEARNING RATE
    num_est = 200
    rate = 1.5
    ada = AdaBoostClassifier(n_estimators=num_est, learning_rate=rate)
    accuracy_scores = []
    log_loss_scores = []
    start = time.time()
    for train, test in kf.split(features):
         X_train, X_test = features.iloc[train], features.iloc[test]
         y_train, y_test = labels.iloc[train], labels.iloc[test]
         ada.fit(X_train, y_train)
         
         pred_logloss = ada.predict_proba(X_test)
         log_loss_scores.append(log_loss(y_test, pred_logloss, labels = labels))
         
         pred_acc = ada.predict(X_test)
         accuracy_scores.append(accuracy_score(y_test, pred_acc))
         
    print("ADABOOST BASE ESTIMATORS: ", num_est)
    print("ADABOOST LEARNING RATE: ", rate)
    print("ADABOOST MEAN LOG LOSS: ", mean(log_loss_scores))
    print("ADABOOST MEAN ACCURACY: %", mean(accuracy_scores)*100)
    print("ADABOOST TIME: ", time.time() - start, "\n")
    
def VotingClass(labels, features, kf, estimators):
    vc = VotingClassifier(estimators = estimators, voting = 'soft')
    #lowers with addition of ADA
    accuracy_scores = []
    log_loss_scores = []
    start = time.time()
    for train, test in kf.split(features):
         X_train, X_test = features.iloc[train], features.iloc[test]
         y_train, y_test = labels.iloc[train], labels.iloc[test]
         vc.fit(X_train, y_train)
         
         pred_logloss = vc.predict_proba(X_test)
         log_loss_scores.append(log_loss(y_test, pred_logloss, labels = labels))
         
         pred_acc = vc.predict(X_test)
         accuracy_scores.append(accuracy_score(y_test, pred_acc))
         

    print("VOTING MEAN LOG LOSS: ", mean(log_loss_scores))
    print("VOTING MEAN ACCURACY: %", mean(accuracy_scores)*100)
    print("VOTING TIME: ", time.time() - start, "\n")
    

def RandomForestClass(labels, features, kf):
    crit = 'entropy'
    depth = 3
    rand = RandomForestClassifier(n_estimators = 100, criterion = crit, max_depth = depth)
    #max_depth 2 worst loss than decision tree with same max
    accuracy_scores = []
    log_loss_scores = []
    start = time.time()
    for train, test in kf.split(features):
         X_train, X_test = features.iloc[train], features.iloc[test]
         y_train, y_test = labels.iloc[train], labels.iloc[test]
         rand.fit(X_train, y_train)
         
         pred_logloss = rand.predict_proba(X_test)
         log_loss_scores.append(log_loss(y_test, pred_logloss, labels = labels))
         
         pred_acc = rand.predict(X_test)
         accuracy_scores.append(accuracy_score(y_test, pred_acc))
    
    print("RANDOM FOREST CRITERION: ", crit)
    print("RANDOM FOREST MAX DEPTH: ", depth)
    print("RANDOM FOREST MEAN LOG LOSS: ", mean(log_loss_scores))
    print("RANDOM FOREST MEAN ACCURACY: %", mean(accuracy_scores)*100)
    print("RANDOM FOREST TIME: ", time.time() - start, "\n")

def OneVRest(labels, features, kf, k):
    crit = 'entropy'
    depth = 1
    ov1 = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = k))
    ov2 = OneVsRestClassifier(GaussianNB())   
    ov3 = OneVsRestClassifier(DecisionTreeClassifier(criterion = crit, max_depth = depth))
    
    accuracy_scores1 = []
    log_loss_scores1 = []
    
    accuracy_scores2 = []
    log_loss_scores2 = []
    
    accuracy_scores3 = []
    log_loss_scores3 = []
    
    start = time.time()
    for train, test in kf.split(features):
         X_train, X_test = features.iloc[train], features.iloc[test]
         y_train, y_test = labels.iloc[train], labels.iloc[test]
         
         ov1.fit(X_train, y_train)
         pred_logloss1 = ov1.predict_proba(X_test)
         log_loss_scores1.append(log_loss(y_test, pred_logloss1, labels = labels))
         pred_acc1 = ov1.predict(X_test)
         accuracy_scores1.append(accuracy_score(y_test, pred_acc1))
    print("K VALUE: ", k)
    print("ONE VS REST (KNN) LOG LOSS: ", mean(log_loss_scores1))
    print("ONE VS REST (KNN) ACCURACY: %", mean(accuracy_scores1)*100)
    print("ONE VS REST (KNN) TIME: ", time.time() - start, "\n")
         
    start = time.time()
    for train, test in kf.split(features):
         X_train, X_test = features.iloc[train], features.iloc[test]
         y_train, y_test = labels.iloc[train], labels.iloc[test]  
         ov2.fit(X_train, y_train)
         pred_logloss2 = ov2.predict_proba(X_test)
         log_loss_scores2.append(log_loss(y_test, pred_logloss2, labels = labels))
         pred_acc2 = ov2.predict(X_test)
         accuracy_scores2.append(accuracy_score(y_test, pred_acc2))
    print("ONE VS REST (NB) LOG LOSS: ", mean(log_loss_scores2))
    print("ONE VS REST (NB) ACCURACY: %", mean(accuracy_scores2)*100)
    print("ONE VS REST (NB) TIME: ", time.time() - start, "\n")
         
    start = time.time()
    for train, test in kf.split(features):
         X_train, X_test = features.iloc[train], features.iloc[test]
         y_train, y_test = labels.iloc[train], labels.iloc[test]
         ov3.fit(X_train, y_train)
         pred_logloss3 = ov3.predict_proba(X_test)
         log_loss_scores3.append(log_loss(y_test, pred_logloss3, labels = labels))
         pred_acc3 = ov3.predict(X_test)
         accuracy_scores3.append(accuracy_score(y_test, pred_acc3))      
    print("DT CRITERION: ", crit)
    print("DT MAX DEPTH: ", depth)
    print("ONE VS REST (DT) LOG LOSS: ", mean(log_loss_scores3))
    print("ONE VS REST (DT) ACCURACY: %", mean(accuracy_scores3)*100)
    print("ONE VS REST (DT) TIME: ", time.time() - start, "\n")
    
# ----- END CLASSIFICATION -----#

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
