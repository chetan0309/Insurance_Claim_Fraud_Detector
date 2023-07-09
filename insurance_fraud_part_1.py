# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:12:47 2022

@author: CCE
"""

def predictorfunction(xnew):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sbn
    df=pd.read_csv("fraud_oracle.csv")
    print(df.head())
    print(df.columns)
    l=list(df.columns)
    for i in l:
        print(df[i].value_counts())
    print(df.info())
    print(df.isnull().sum())
    col_to_drop=["Month",
                 "WeekOfMonth",
                 "DayOfWeek",
                 "Make",
                 "AccidentArea",
                 "DayOfWeekClaimed",
                 "MonthClaimed",
                 "WeekOfMonthClaimed",
                 "Age",
                 "VehicleCategory",
                 "PolicyNumber",
                 "RepNumber",
                 "AddressChange_Claim",
                 "Year"]
    # these columns does not have effect on the prediction
    df.drop(columns=col_to_drop,inplace=True)
    List_removal=[0,0,0,0,0,0,0,0,2,4,6,6,17,18]
    for i in List_removal:
        xnew.pop(i)
    
    print(df.head())
    print(df.info())
    for i in df:
        print(df[i].value_counts())
    #Label Encoding
    from sklearn.preprocessing import LabelEncoder
    #Use different encoder for different columns
    le1=LabelEncoder()
    le2=LabelEncoder()
    df["Sex"]=le1.fit_transform(df["Sex"])
    df["AgentType"]=le2.fit_transform(df["AgentType"])
    xnew[0]=int(le1.transform([xnew[0]]))
    xnew[15]=int(le2.transform([xnew[15]]))
    print(df.head())
    #custom encoding
    df["Fault"]=df["Fault"].map({"Policy Holder":0,"Third Party":1})
    df["PoliceReportFiled"]=df["PoliceReportFiled"].map({"No":0,"Yes":1})
    df["WitnessPresent"]=df["WitnessPresent"].map({"No":0,"Yes":1})
    
    if xnew[2]=="Policy Holder":
        
        xnew[2]=0
    else:
        xnew[2]=1
        
        
    if xnew[13]=="No":
        xnew[13]=0
    else:
        xnew[13]=1
        
        
    if xnew[14]=="No":
        xnew[14]=0
    else:
        xnew[14]=1
    
    
    print(df.head())
    for i in df:
        print(df[i].value_counts())
    print(df.head())
    #Categorical encoding 
    #OneHotEncoder
    #custom categorical encoding
    df["Days_Policy_Accident"]=df["Days_Policy_Accident"].map({"more than 30":4,"none":0,"8 to 15":2,"15 to 30":3,"1 to 7":1})
    df["Days_Policy_Claim"]=df["Days_Policy_Claim"].map({"more than 30":4,"15 to 30":3,"8 to 15":2,"none":1})
    df["PastNumberOfClaims"]=df["PastNumberOfClaims"].map({"2 to 4":2,"1":1,"none":0,"more than 4":3})
    df["AgeOfVehicle"]=df["AgeOfVehicle"].map({"new":0,"2 years":1,"3 years":2,"4 years":3,"5 years":4,"6 years":5,"7 years":6,"more than 7":7})
    
    
    if xnew[8]=="more than 30":
        xnew[8]=4
    elif xnew[8]=="none":
        xnew[8]=0
    elif xnew[8]=="8 to 15":
        xnew[8]=2
    elif xnew[8]=="15 to 30":
        xnew[8]=3
    else:
        xnew[8]=1
       
        
        
    if xnew[9]=="more than 30":
        xnew[9]=4
    elif xnew[9]=="15 to 30":
        xnew[9]=3
    elif xnew[9]=="8 to 15":
        xnew[9]=2
    else:
        xnew[9]=1
    
    
    
    if xnew[10]=="2 to 4":
        xnew[10]=2
    elif xnew[10]=="none":
        xnew[10]=0
    elif xnew[10]=="1":
        xnew[10]=1
    else:
        xnew[10]=3
        
        
        
    if xnew[11]=="new":
        xnew[11]=0
    elif xnew[11]=="2 years":
        xnew[11]=1
    elif xnew[11]=="3 years":
        xnew[11]=2
    elif xnew[11]=="4 years":
        xnew[11]=3
    elif xnew[11]=="5 years":
        xnew[11]=4
    elif xnew[11]=="6 years":
        xnew[11]=5
    elif xnew[11]=="7 years":
        xnew[11]=6
    else:
         xnew[11]=7
        
    
    if xnew[12]=="16 to 17":
        xnew[12]=0
    elif xnew[12]=="18 to 20":
        xnew[12]=1
    elif xnew[12]=="21 to 25":
        xnew[12]=2
    elif xnew[12]=="26 to 30":
        xnew[12]=3
    elif xnew[12]=="31 to 35":
        xnew[12]=4
    elif xnew[12]=="36 to 40":
        xnew[12]=5
    elif xnew[12]=="41 to 50":
        xnew[12]=6
    elif xnew[12]=="51 to 65":
         xnew[12]=7
    else:
        xnew[12]=8
        
    
    
    if xnew[16]=="none":
        xnew[16]=0
    elif xnew[16]=="1 to 2":
        xnew[16]=1
    elif xnew[16]=="3 to 5":
        xnew[16]=2
    elif xnew[16]=="more than 5":
        xnew[16]=3
    
    
    
    if xnew[17]=="1 vehicle":
        xnew[17]=1
    elif xnew[17]=="2 vehicles":
        xnew[17]=2
    elif xnew[17]=="3 to 4":
        xnew[17]=3.5
    elif xnew[17]=="5 to 8":
        xnew[17]=6.5
    else:
        xnew[17]=9
        
    
    
    if xnew[4]=="less than 20000":
        xnew[4]=20000
    elif xnew[4]=="20000 to 29000":
        xnew[4]=25000
    elif xnew[4]=="30000 to 39000":
        xnew[4]=35000
    elif xnew[4]=="40000 to 59000":
        xnew[4]=52000
    elif xnew[4]=="60000 to 69000":
        xnew[4]=65000
    else:
         xnew[4]=70000
    
    
    
    
    if xnew[1]=="Married":
        #xnew.pop(1)
        xnew.extend([0,1,0,0])
    elif xnew[1]=="Single":
        #xnew.pop(1)
        xnew.extend([0,0,1,0])
    elif xnew[1]=="Divorced":
        #xnew.pop(1)
        xnew.extend([1,0,0,0])
    else:
        #xnew.pop(1)
        xnew.extend([0,0,0,1])
        
    
    if xnew[3]=="Sedan - Collision":
        #xnew.pop(3)
        xnew.extend([0,1,0,0,0,0,0,0,0])
    elif xnew[3]=="Sedan - Liability":
        #xnew.pop(3)
        xnew.extend([0,0,1,0,0,0,0,0,0])
    elif xnew[3]=="Sedan - All Perils":
        #xnew.pop(3)
        xnew.extend([1,0,0,0,0,0,0,0,0])
    elif xnew[3]=="Sport - Collision":
        #xnew.pop(3)
        xnew.extend([0,0,0,0,1,0,0,0,0])
    elif xnew[3]=="Utility - All Perils":
        #xnew.pop(3)
        xnew.extend([0,0,0,0,0,0,1,0,0])
    elif xnew[3]=="Utility - Collision":
        #xnew.pop(3)
        xnew.extend([0,0,0,0,0,0,0,1,0])
    elif xnew[3]=="Sport - All Perils":
        #xnew.pop(3)
        xnew.extend([0,0,0,1,0,0,0,0,0])
    elif xnew[3]=="Utility - Liability":
         #xnew.pop(3)
         xnew.extend([1,0,0,0,0,0,0,0,1])
    else:
        #xnew.pop(3)
        xnew.extend([0,0,0,0,0,1,0,0,0])
        
    
    
    if xnew[18]=="All Perils":
        #xnew.pop(18)
        xnew.extend([1,0,0])
    elif xnew[18]=="Collision":
        #xnew.pop(18)
        xnew.extend([0,1,0])
    else:
        #xnew.pop(18)
        xnew.extend([0,0,1])
        
    xnew.pop(1)
    xnew.pop(2)
    xnew.pop(16)
    xnew.pop(3)
    print(df.head())
    df["AgeOfPolicyHolder"]=df["AgeOfPolicyHolder"].map({"16 to 17":0,"18 to 20":1,"21 to 25":2,"26 to 30":3,"31 to 35":4,"36 to 40":5,"41 to 50":6,"51 to 65":7,"over 65":8})
    df["NumberOfSuppliments"]=df["NumberOfSuppliments"].map({"none":0,"1 to 2":1,"3 to 5":2,"more than 5":3})
    df["NumberOfCars"]=df["NumberOfCars"].map({"1 vehicle":1,"2 vehicles":2,"3 to 4":3.5,"5 to 8":6.5,"more than 8":9})
    df["VehiclePrice"]=df["VehiclePrice"].map({"less than 20000":20000,"20000 to 29000":25000,"30000 to 39000":35000,"40000 to 59000":52000,"60000 to 69000":65000,"more than 69000":70000})
    print(df.info())
    #MaritalStatus
    #PolicyType
    #BasePolicy

    df_temp1=df["MaritalStatus"]
    df_temp2=df["PolicyType"]
    df_temp3=df["BasePolicy"]
    df_temp1=pd.get_dummies(df_temp1,columns=["MaritalStatus"],prefix="MaritalStatus")
    print(df_temp1.head())
    df_temp2=pd.get_dummies(df_temp2,columns="PolicyType",prefix="PolicyType")
    print(df_temp2.head())
    df_temp3=pd.get_dummies(df_temp3,columns="BasePolicy",prefix="BasePolicy")
    print(df_temp3.head())
    print(df.head())
    col_drop=["MaritalStatus","PolicyType","BasePolicy"]
    df.drop(columns=col_drop,inplace=True)
    print(df.head())
    df=pd.concat([df,df_temp1,df_temp2,df_temp3],axis=1)
    print(df.head())
    for i in df:
        print(df[i].value_counts())
    y=df["FraudFound_P"]
    X=df.drop(columns="FraudFound_P")
    print(X.head())
    print(y.head())
    # understanding the frequency(occurrence) and distribution of values in the modified dataframe df
    plt.hist(df["VehiclePrice"])
    sbn.distplot(df["Sex"])
    #data is uniformly distributed for all categories of gender
    sbn.distplot(df["VehiclePrice"])
    #from here we can see that most of the claims are for vehicle have price in between 20000 to 30000
    sbn.distplot(df["AgeOfPolicyHolder"])
    # people in the age group of 31 to 35 claimed the most
    plt.figure(figsize=(20,15))
    sbn.heatmap(X.corr(),annot=True)
    from sklearn.cluster import KMeans
    wcss=[]
    for i in range(1,10):
        kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1,10),wcss,linestyle="solid",marker="o")
    plt.xlabel("number of clusters")
    plt.ylabel("wcss")
    plt.show()
    # the number of clusters =3
    kmeans=KMeans(n_clusters=3,init="k-means++",random_state=0)
    kmeans.fit(X)
    cluster_number=kmeans.predict(X)
    
    #cluster_number is an array storing the cluster number of each row in X
    print(np.unique(cluster_number))
    l=list(cluster_number)
    count_0=l.count(0)
    count_1=l.count(1)
    count_2=l.count(2)
    print(count_0,count_1,count_2)
    #we can see we have enough number of points in each cluster
    #l_0 stores the index of those rows in df which belong to cluster 0
    #l_1 stores the index of those rows in df which belong to cluster 1
    #l_2 stores the index of those rows in df which belong to cluster 2

    l_0=[]
    l_1=[]
    l_2=[]
    for i in range(15420):
        if l[i]==0:
            l_0.append(i)
        elif l[i]==1:
            l_1.append(i)
        else:
            l_2.append(i)
    
    df_0=df.iloc[l_0,:]
    df_1=df.iloc[l_1,:]
    df_2=df.iloc[l_2,:]
    #setting the indexs of each data frame df_0,df_1,df_2
    df_0.set_index(pd.Index(range(3994)),inplace=True)
    df_1.set_index(pd.Index(range(2251)),inplace=True)
    df_2.set_index(pd.Index(range(9175)),inplace=True)
    
    #creating X and y for each clusters (independent and dependent varialbles)
    y_0=df_0["FraudFound_P"]
    X_0=df_0.drop(columns="FraudFound_P")
    y_1=df_1["FraudFound_P"]
    X_1=df_1.drop(columns="FraudFound_P")
    y_2=df_2["FraudFound_P"]
    X_2=df_2.drop(columns="FraudFound_P")
    #Now we will define a function train test split
    from sklearn.model_selection import train_test_split
    X_0_train,X_0_test,y_0_train,y_0_test=train_test_split(X_0,y_0,test_size=0.2,random_state=42)
    X_1_train,X_1_test,y_1_train,y_1_test=train_test_split(X_1,y_1,test_size=0.2,random_state=42)
    X_2_train,X_2_test,y_2_train,y_2_test=train_test_split(X_2,y_2,test_size=0.2,random_state=42)    
    print(type(X_0_train))
    #Creating numpy arrays(ndarrays) of train and test data

    X_0_train=X_0_train.values
    X_0_test=X_0_test.values
    y_0_train=y_0_train.values
    y_0_test=y_0_test.values
    X_1_train=X_1_train.values
    X_1_test=X_1_test.values
    y_1_train=y_1_train.values
    y_1_test=y_1_test.values
    X_2_train=X_2_train.values
    X_2_test=X_2_test.values
    y_2_train=y_2_train.values
    y_2_test=y_2_test.values
    
    #feature scaling the train and test data of each cluster

    from sklearn.preprocessing import StandardScaler
    sc0=StandardScaler()
    sc1=StandardScaler()
    sc2=StandardScaler()
    X_0_train[:,2:4]=sc0.fit_transform(X_0_train[:,2:4])
    X_0_test[:,2:4]=sc0.transform(X_0_test[:,2:4])
    X_1_train[:,2:4]=sc1.fit_transform(X_1_train[:,2:4])
    X_1_test[:,2:4]=sc1.transform(X_1_test[:,2:4])
    X_2_train[:,2:4]=sc2.fit_transform(X_2_train[:,2:4])
    X_2_test[:,2:4]=sc2.transform(X_2_test[:,2:4])
    # now we will define a function to find a best model(classifier) for each cluster 
    def model_finder(xtrain,xtest,ytrain,ytest):
        from sklearn.metrics import f1_score
        #1 Logistic regression
        #2 K-Nearest Neighbors
        #3 Support Vector Machine(SVM)
        #4 Kernel SVM
        #5 Naive Bayes
        #6 Decision Tree Classifier
        #7 Random Forest Classifier
        #8 XGBoost Classifier
        cls_name=["Logistic Regression","K-Nearest Neighbors","Support Vector Machine","Kernel SVM","Naive Bayes","Decision Tree Classifier","Random Forest Classifier","XGBoost Classifier"]
        score=[]
        #1 Logistic regression
        from sklearn.linear_model import LogisticRegression
        classifier_LR=LogisticRegression(random_state=42,solver="sag",max_iter=1000)
        classifier_LR.fit(xtrain,ytrain)
        y_pred=classifier_LR.predict(xtest)
        score_LR=f1_score(ytest,y_pred)#we are keeping the default values of other parameters like average="binary",pos_label=1 are default values
        score.append(score_LR)
    
        #2 K-Nearest Neighbors
        from sklearn.neighbors import KNeighborsClassifier
        classifier_KNN=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
        classifier_KNN.fit(xtrain,ytrain)
        y_pred=classifier_KNN.predict(xtest)
        score_KNN=f1_score(ytest,y_pred)
        score.append(score_KNN)
        
        #3 Support Vector Machine(SVM)
        from sklearn.svm import SVC
        classifier_SVM=SVC(kernel="linear",random_state=42)
        classifier_SVM.fit(xtrain,ytrain)
        y_pred=classifier_SVM.predict(xtest)
        score_SVM=f1_score(ytest,y_pred)
        score.append(score_SVM)
        
        #4 Kernel SVM
        from sklearn.svm import SVC
        classifier_KSVM=SVC(kernel="rbf",random_state=42)
        classifier_KSVM.fit(xtrain,ytrain)
        y_pred=classifier_KSVM.predict(xtest)
        score_KSVM=f1_score(ytest,y_pred)
        score.append(score_KSVM)
    
        #5 Naive Bayes
        from sklearn.naive_bayes import GaussianNB
        classifier_NB=GaussianNB()
        classifier_NB.fit(xtrain,ytrain)
        y_pred=classifier_NB.predict(xtest)
        score_NB=f1_score(ytest,y_pred)
        score.append(score_NB)
    
        #6 Decision Tree Classifier
        from sklearn.tree import DecisionTreeClassifier
        classifier_DT=DecisionTreeClassifier(criterion="entropy", random_state=42)
        classifier_DT.fit(xtrain,ytrain)
        y_pred=classifier_DT.predict(xtest)
        score_DT=f1_score(ytest,y_pred)
        score.append(score_DT)
        
        #7 Random Forest Classifier
        from sklearn.ensemble import RandomForestClassifier
        classifier_RF=RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=42)
        classifier_RF.fit(xtrain,ytrain)
        y_pred=classifier_RF.predict(xtest)
        score_RF=f1_score(ytest,y_pred)
        score.append(score_RF)
        
        #8 XGBoost Classifier
        from xgboost import XGBClassifier
        classifier_xgb=XGBClassifier(use_label_encoder=False)#making it equal to false will just remove the warnings
        classifier_xgb.fit(xtrain,ytrain,eval_metric='rmse')
        y_pred=classifier_xgb.predict(xtest)
        score_xgb=f1_score(ytest,y_pred)
        score.append(score_xgb)
        
        classifier_list=[classifier_LR,classifier_KNN,classifier_SVM,classifier_KSVM,classifier_NB,classifier_DT,classifier_RF,classifier_xgb]
        max_score=max(score)
        classifier_name=cls_name[score.index(max_score)]
        classifier=classifier_list[(score.index(max_score))]
        return classifier_name,classifier,max_score
    
    
    
    
    classifier_name_0,classifier_0,max_score_0=model_finder(X_0_train,X_0_test,y_0_train,y_0_test)
    classifier_name_1,classifier_1,max_score_1=model_finder(X_1_train,X_1_test,y_1_train,y_1_test)
    classifier_name_2,classifier_2,max_score_2=model_finder(X_2_train,X_2_test,y_2_train,y_2_test)
        
    classifier_list=[classifier_0,classifier_1,classifier_2]
    classifier_name_list=[classifier_name_0,classifier_name_1,classifier_name_2]
        
    #Hyperparameters tuning

    def hyperparameter_tuning(classifier,classifier_name,xtrain,ytrain):
    
        from sklearn.model_selection import GridSearchCV #This time we'll use GridSearchCV instead of RandomizedSearchCV
        
        if classifier_name=="Logistic Regression":
            
            param_grid = [    
                    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                     'C' : np.logspace(-4, 4, 20),
                     'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
                     'max_iter' : [1000,2500, 5000]
                     }
                        ]
            
            clf = GridSearchCV(classifier, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
            clf.fit(xtrain,ytrain)
            
        elif classifier_name=="K-Nearest Neighbors":
                
            leaf_size = list(range(1,50))
            n_neighbors = list(range(1,30))
            p=[1,2]
        
            #Convert to dictionary
            hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
        
            clf = GridSearchCV(classifier, hyperparameters, cv=10)
            clf.fit(xtrain,ytrain)
           
        elif classifier_name=="Support Vector Machine":
            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          }
            
            clf = GridSearchCV(classifier, param_grid, refit = True, verbose = 3)
            clf.fit(xtrain,ytrain)
        
        elif classifier_name=="Kernel SVM":
            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          }
            
            clf = GridSearchCV(classifier, param_grid, refit = True, verbose = 3)
            
            clf.fit(xtrain,ytrain)
            
        elif classifier_name=="Naive Bayes":
            param_grid_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
        
            clf = GridSearchCV(estimator=classifier, param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
            
            clf.fit(xtrain, ytrain)
            
        elif classifier_name=="Decision Tree Classifier":
            params = { 'max_depth': [2, 3, 5, 10, 20],
                      'min_samples_leaf': [5, 10, 20, 50, 100],
                      'criterion': ["gini", "entropy"]
                      }
            clf = GridSearchCV(estimator=classifier,param_grid=params,cv=4,n_jobs=-1,verbose=1)
        
            clf.fit(xtrain, ytrain)
        
        elif classifier_name=="Random Forest Classifier":
        
        
            param_grid = {'bootstrap': [True],
                          'max_depth': [80, 90, 100, 110],
                          'max_features': [2, 3],
                          'min_samples_leaf': [3, 4, 5],
                          'min_samples_split': [8, 10, 12],
                          'n_estimators': [100, 200, 300, 1000]
                          }


            clf = GridSearchCV(estimator = classifier, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)
        
            clf.fit(xtrain,ytrain)
         
        else:#it will be XGBoost Classifier
            params = { 'max_depth': [3,6,10],
                      'learning_rate': [0.01, 0.05, 0.1],
                      'n_estimators': [100, 500, 1000],
                      'colsample_bytree': [0.3, 0.7]}
            clf = GridSearchCV(estimator=classifier,param_grid=params,verbose=1)
            clf.fit(xtrain, ytrain, eval_metric='rmse')
            
            
        return clf
    
    classifier_0=hyperparameter_tuning(classifier_0,classifier_name_0,X_0_train,y_0_train)
    classifier_1=hyperparameter_tuning(classifier_1,classifier_name_1,X_1_train,y_1_train)
    classifier_2=hyperparameter_tuning(classifier_2,classifier_name_2,X_2_train,y_2_train)
        
        
        
    #now all is done
    #we'll have the best model for each cluster
    print(xnew)
    cluster_belong=kmeans.predict([xnew])
    print("Belongs to cluster :",cluster_belong)
    if cluster_belong==0:
        ypredicted=classifier_0.predict([xnew])
    elif cluster_belong==1:
        ypredicted=classifier_1.predict([xnew])
    else:
        ypredicted=classifier_2.predict([xnew])
    
    return ypredicted    
        
    
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    