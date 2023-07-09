# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:59:23 2022

@author: CCE
"""

import insurance_fraud_part_1

while True:
    print("Enter the new values in the specified format:")
    
    L_new_data=[]
    
    #By the way Blow eight entries are not included in predictions
    
    Month=input("Enter the Month in string format e.g. as 'Jan','Feb' etc:")
    
    L_new_data.append(Month)
    
    WeekOfMonth=int(input("Enter the WeekOfMonth e.g. 1,2 etc:"))
    
    L_new_data.append(WeekOfMonth)
    
    DayOfWeek=input("Enter the day of week e.g. Wednesday ,Friday:")
    
    L_new_data.append(DayOfWeek)
    
    Make=input("Enter the Make e.g. Honda,Tyotota (first letter capital):")
    
    L_new_data.append(Make)
    
    AccidentArea=input("Enter the Accident Area Urban or Rural:")
    
    L_new_data.append(AccidentArea)
    
    DayOfWeekClaimed=input("Enter the DayOfWeekClaimed e.g. Wednesday ,Friday:")
    
    L_new_data.append(DayOfWeekClaimed)
    
    MonthClaimed=input("Enter Month Claimed e.g. Jan, Feb:")
    
    L_new_data.append(MonthClaimed)
    
    WeekOfMonthClaimed=int(input("Enter Week of Month Claimed e.g. 1,2 etc:"))
    
    L_new_data.append(WeekOfMonthClaimed)
    
    Sex=input("Enter the Gender e.g.Male or Female:")
    
    while (Sex not in ["Male","Female"]):
        
        Sex=input("Enter the Gender in correct format e.g. Male, Female:")
        
    L_new_data.append(Sex)
    
    MaritalStatus=input("Enter the Marital Status e.g. Married,Single,Divorced or Widow:")
    
    while (MaritalStatus not in ["Married","Single","Divorced","Widow"]):
        
        MaritalStatus=input("Enter the Marital Status in correct in format e.g. Married,Single,Divorced or Widow:")
        
    L_new_data.append(MaritalStatus)
    
    #By the way age is not included in predictions
    
    Age=int(input("Enter the Age e.g,. 23, 45 etc:"))
    
    L_new_data.append(Age)
    
    Fault=input("Enter the Fault -:Policy Holder or Third Party:")
    
    while (Fault not in ["Policy Holder","Third Party"]):
        
        Fault=input("Enter the Fault in correct format -:Policy Holder or Third Party:")
        
    L_new_data.append(Fault)
    
    PolicyType=input("Enter the PolicyType -:Sedan - Collision,Sedan - Liability,Sedan - All Perils,Sport - Collision,Utility - All Perils,Utility - Collision,Sport - All Perils,Utility - Liability,Sport - Liability:")
    
    while (PolicyType not in ["Sedan - Collision","Sedan - Liability","Sedan - All Perils","Sport - Collision","Utility - All Perils","Utility - Collision","Sport - All Perils","Utility - Liability","Sport - Liability"]):
        
        PolicyType=input("Enter the PolicyType in correct format-:Sedan - Collision,Sedan - Liability,Sedan - All Perils,Sport - Collision,Utility - All Perils,Utility - Collision,Sport - All Perils,Utility - Liability,Sport - Liability:")
        
    L_new_data.append(PolicyType)
    #By the wayVehicleCategory is not included in predictions
    VehicleCategory=input("Enter the Vehicle Cetegory Type -:Sedan,Sport or Utility:")
    
    L_new_data.append(VehicleCategory)
    
    VehiclePrice=input("Enter the range of vehicle price -:20000 to 29000,30000 to 39000,more than 69000,less than 20000,40000 to 59000,60000 to 69000:")
    
    while (VehiclePrice not in ["20000 to 29000","30000 to 39000","more than 69000","less than 20000","40000 to 59000","60000 to 69000"]):
        
        VehiclePrice=input("Enter the range of vehicle price in correct format-:20000 to 29000,30000 to 39000,more than 69000,less than 20000,40000 to 59000,60000 to 69000:")
    
    L_new_data.append(VehiclePrice)
    
    #By the way this estimation is not used in predictions
    
    FraudFound_P=int(input("Enter your estimation for fraud 0 or 1:"))
    
    L_new_data.append(FraudFound_P)
    
    #By the way Policy Number is not used in predictions
    
    PolicyNumber=int(input("Enter the policy number:"))
    
    L_new_data.append(PolicyNumber)
    
    #By the RepNumber is not used in predictions
    
    RepNumber=int(input("Enter the rep number interger between 1-16(both inclusive):"))
    
    L_new_data.append(RepNumber)
    
    Deductible=int(input("Enter the deductible -:400,500,700,300:"))
    
    while (Deductible not in [300,400,500,700]):
        
        Deductible=int(input("Enter the deductible correctly -:400,500,700,300:"))
        
    L_new_data.append(Deductible)
    
    DriverRating=int(input("Enter the driver rating in between 1-4 both inclusive:"))
    
    while (DriverRating not in [1,2,3,4]):
        
        DriverRating=int(input("Enter the driver rating in between 1-4 both inclusive:"))
        
    L_new_data.append(DriverRating)
    
    Days_Policy_Accident=input("Enter the days policy accident-: more than 30,none,8 to 15,15 to 30,1 to 7:")
    
    while ( Days_Policy_Accident not in ["more than 30","none","8 to 15","15 to 30","1 to 7"]):
        
        Days_Policy_Accident=input("Enter the days policy accident in correct format-: more than 30,none,8 to 15,15 to 30,1 to 7:")
    
    L_new_data.append(Days_Policy_Accident)
    
    Days_Policy_Claim=input("Enter the days policy claim -:more than 30,15 to 30,8 to 15,none:")
    
    while (Days_Policy_Claim not in ["more than 30","15 to 30","8 to 15","none"]):
        
        Days_Policy_Claim=input("Enter the days policy claim  in correct format-:more than 30,15 to 30,8 to 15,none:")
            
    L_new_data.append(Days_Policy_Claim)
    
    PastNumberOfClaims=input("Enter the past number of claims -:2 to 4,none,1,more than 4:")
    
    while (PastNumberOfClaims not in ["2 to 4","none","1","more than 4"]):
        
        PastNumberOfClaims=input("Enter the past number of claims in correct format-:2 to 4,none,1,more than 4:")
    
    L_new_data.append(PastNumberOfClaims)
    
    AgeOfVehicle=input("Enter Age of vehicle -:7 years,more than 7,6 years,5 years,new,4 years,3 years,2 years:")
    
    while (AgeOfVehicle not in ["7 years","more than 7","6 years","5 years","new","4 years","3 years","2 years"]):
        
        AgeOfVehicle=input("Enter Age of vehicle in correct format-:7 years,more than 7,6 years,5 years,new,4 years,3 years,2 years:")
        
    L_new_data.append(AgeOfVehicle)
    
    AgeOfPolicyHolder=input("Enter the Age of policy holder -:31 to 35,36 to 40,41 to 50,51 to 65,26 to 30,over 65,16 to 17,21 to 25,18 to 20:")
    
    while (AgeOfPolicyHolder not in ["31 to 35","36 to 40","41 to 50","51 to 65","26 to 30","over 65","16 to 17","21 to 25","18 to 20"]):
        
        AgeOfPolicyHolder=input("Enter the Age of policy holder in correct format-:31 to 35,36 to 40,41 to 50,51 to 65,26 to 30,over 65,16 to 17,21 to 25,18 to 20:")
        
    L_new_data.append(AgeOfPolicyHolder)
    
    PoliceReportFiled=input("Enter Yes or No whether Police Report Filed or not:")
    
    while (PoliceReportFiled not in ["Yes","No"]):
        
        PoliceReportFiled=input("Enter Yes or No whether Police Report Filed or not:")
        
    L_new_data.append(PoliceReportFiled)
    
    WitnessPresent=input("Enter Yes or No whether witness present or not:")
    
    while (WitnessPresent not in ["Yes","No"]):
        
        WitnessPresent=input("Enter Yes or No whether witness present or not:")
        
    L_new_data.append(WitnessPresent)
    
    AgentType=input("Enter the Agent Type External or Internal:")
    
    while (AgentType not in ["External","Internal"]):
        
        AgentType=input("Enter the Agent Type External or Internal:")
        
    L_new_data.append(AgentType)
    
    NumberOfSuppliments=input("Enter the Number of suppliments -:none,more than 5,1 to 2,3 to 5:")
    
    while (NumberOfSuppliments not in ["none","more than 5","1 to 2","3 to 5"]):
        
        NumberOfSuppliments=input("Enter the Number of suppliments in correct format-:none,more than 5,1 to 2,3 to 5:")
        
    L_new_data.append(NumberOfSuppliments)
    
    #By the way AddressChange_Claim is not used in predictions
    
    AddressChange_Claim=input("Enter the Address Change claim -:no change,4 to 8 years,2 to 3 years,1 year,under 6 months:")
    
    L_new_data.append(AddressChange_Claim)
    
    NumberOfCars=input("Enter the Number of cars -:1 vehicle,2 vehicles,3 to 4,5 to 8,more than 8:")
    
    while (NumberOfCars not in ["1 vehicle","2 vehicles","3 to 4","5 to 8","more than 8"]):
        
        NumberOfCars=input("Enter the Number of cars in correct format-:1 vehicle,2 vehicles,3 to 4,5 to 8,more than 8:")
        
    L_new_data.append(NumberOfCars)
    
    #By the way year is not used in predictions
    
    Year=int(input("Enter the Year e.g. 1994,1995,1996:"))
    
    L_new_data.append(Year)
    
    BasePolicy=input("Enter the Base policy either Collision,Liability,All Perils:")
    
    while (BasePolicy not in ["Collision","Liability","All Perils"]):
        
        BasePolicy=input("Enter the Base policy either Collision,Liability,All Perils:")
        
    L_new_data.append(BasePolicy)
    
    prediction=insurance_fraud_part_1.predictorfunction(L_new_data)
    
    if(prediction==1):
        
        print("Fraud Detected")
        
    elif prediction==0:
        
        print("Fraud not Found")
        
    else:
        
        print("Cannot be predicted")
        
        
    