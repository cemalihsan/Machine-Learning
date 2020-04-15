import csv
import pandas as pd
import numpy as np
import math

labels=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']
dataset = pd.read_csv('student-mat.csv',delimiter =';')

values = dataset.get_values()
print(values)

dataset_len = len(values)
train_percent = 80
train_len = int((train_percent*dataset_len)/100)
test_len = int(dataset_len - train_len)
print("Train Dataset Size",train_len)
print("Test Dataset Size",test_len)

PassStd,FailStd,PassStdProb,FailStdProb,UrbanPass,UrbanNotPass,RuralPass,RuralNotPass = 0,0,0.0,0.0,0,0,0,0
TogetherPass,ApartPass,TogetherFail,ApartFail = 0,0,0,0
UrbanPassProb,UrbanNotPassProb,RuralPassProb,RuralNotPassProb = 0.0,0.0,0.0,0.0
TogetherPassProb,TogetherNotPassProb,ApartPassProb,ApartNotPassProb = 0.0,0.0,0.0,0.0

PassStdT,FailStdT,PassStdProbTest,FailStdProbTest,UrbanPassT,UrbanNotPassT,RuralPassT,RuralNotPassT = 0,0,0.0,0.0,0,0,0,0
TogetherPassT,ApartPassT,TogetherFailT,ApartFailT = 0,0,0,0
UrbanPassProbT,UrbanNotPassProbT,RuralPassProbT,RuralNotPassProbT = 0.0,0.0,0.0,0.0
TogetherPassProbT,TogetherNotPassProbT,ApartPassProbT,ApartNotPassProbT = 0.0,0.0,0.0,0.0

print("-----------------------Train Dataset-----------------------")

for value in range(train_len):
	if values[value][labels.index('G3')] >= 10:
		PassStd+=1
		if values[value][labels.index('address')] == "U":
			UrbanPass+=1
		else:
			RuralPass+=1
	else:
		FailStd+=1
		if values[value][labels.index('address')] == "U":
			UrbanNotPass+=1
		else:
			RuralNotPass+=1
			
for param in range(train_len):
	if values[param][labels.index('G3')] >= 10:
		if values[param][labels.index('Pstatus')] == "T":
			TogetherPass+=1
		else:
			ApartPass+=1
	else:
		if values[param][labels.index('Pstatus')] == "T":
			TogetherFail+=1
		else:
			ApartFail+=1
			
			
PassStdProb = PassStd/train_len
FailStdProb = FailStd/train_len

print("Pass Students:",PassStd)
print("Fail Students:",FailStd)

print("Pass Students Probability:",PassStdProb)
print("Fail Students Probability:",FailStdProb)

UrbanPassProb = UrbanPass/PassStd
UrbanNotPassProb = UrbanNotPass/FailStd
RuralPassProb = RuralPass/PassStd
RuralNotPassProb = RuralNotPass/FailStd

print("Urban Pass Probability:",UrbanPassProb)
print("Urban Not Pass Probability:",UrbanNotPassProb)
print("Rural Pass Probability:",RuralPassProb)
print("Rural Not Pass Probability:",RuralNotPassProb)

TogetherPassProb = TogetherPass/PassStd
TogetherNotPassProb = TogetherFail/FailStd
ApartPassProb = ApartPass/PassStd
ApartNotPassProb = ApartFail/FailStd

print("Together Pass Probability:",TogetherPassProb)
print("Together Not Pass Probability:",TogetherNotPassProb)
print("Apart Pass Probability:",ApartPassProb)
print("Apart Not Pass Probability:",ApartNotPassProb)

print("Urban Area given Pass Students:",UrbanPass)
print("Urban Area given Fail Students:",UrbanNotPass)
print("Rural Area given Pass Students:",RuralPass)
print("Rural Area given Fail Students:",RuralNotPass)

print("Together given Pass Students:",TogetherPass)
print("Together given Fail Students:",TogetherFail)
print("Apart given Pass Students:",ApartPass)
print("Apart given Fail Students:",ApartFail)


print("-------------Bayes Calculation-------------------")
P1 = (UrbanPassProb*TogetherPassProb*PassStdProb)/(TogetherPassProb*PassStdProb*UrbanPassProb + UrbanNotPassProb*TogetherNotPassProb*FailStdProb)
P2 = (UrbanPassProb*ApartPassProb*PassStdProb)/(UrbanPassProb*ApartPassProb*PassStdProb + UrbanNotPassProb*ApartNotPassProb*FailStdProb)
P3 = (RuralPassProb*ApartPassProb*PassStdProb)/(RuralPassProb*ApartPassProb*PassStdProb + RuralNotPassProb*ApartNotPassProb*FailStdProb)
P4 = (RuralPassProb*TogetherPassProb*PassStdProb)/(RuralPassProb*TogetherPassProb*PassStdProb + RuralNotPassProb*TogetherNotPassProb*FailStdProb)
P5 = (UrbanNotPassProb*TogetherNotPassProb*FailStdProb)/(UrbanPassProb*TogetherPassProb*PassStdProb + UrbanNotPassProb*TogetherNotPassProb*FailStdProb)
P6 = (RuralNotPassProb*ApartNotPassProb*FailStdProb)/(RuralNotPassProb*ApartNotPassProb*FailStdProb + RuralPassProb*ApartPassProb*PassStdProb)
P7 = (UrbanNotPassProb*ApartNotPassProb*FailStdProb)/(UrbanNotPassProb*ApartNotPassProb*FailStdProb + UrbanPassProb*ApartPassProb*PassStdProb)
P8 = (RuralNotPassProb*TogetherNotPassProb*FailStdProb)/(RuralNotPassProb*TogetherNotPassProb*FailStdProb + RuralPassProb*TogetherPassProb*PassStdProb)
print("U-T",P1)
print("U-A",P2)
print("R-A",P3)
print("R-T",P4)
print("UN-TN",P5)
print("RN-AN",P6)
print("UN-AN",P7)
print("RN-TN",P8)

print("-----------------------Test Dataset-----------------------")

for i in range(train_len,dataset_len):
	if values[i][labels.index('G3')] >= 10:
		PassStdT+=1
		if values[i][labels.index('address')] == "U":
			UrbanPassT+=1
		else:
			RuralPassT+=1
	else:
		FailStdT+=1
		if values[i][labels.index('address')] == "U":
			UrbanNotPassT+=1
		else:
			RuralNotPassT+=1
			
for j in range(train_len,dataset_len):
	if values[j][labels.index('G3')] >= 10:
		if values[j][labels.index('Pstatus')] == "T":
			TogetherPassT+=1
		else:
			ApartPassT+=1
	else:
		if values[j][labels.index('Pstatus')] == "T":
			TogetherFailT+=1
		else:
			ApartFailT+=1
			
			
PassStdProbTest = PassStdT/test_len
FailStdProbTest = FailStdT/test_len

print("Pass Students:",PassStdT)
print("Fail Students:",FailStdT)

print("Pass Students Probability:",PassStdProbTest)
print("Fail Students Probability:",FailStdProbTest)

UrbanPassProbT = UrbanPassT/PassStdT
UrbanNotPassProbT = UrbanNotPassT/FailStdT
RuralPassProbT = RuralPassT/PassStdT
RuralNotPassProbT = RuralNotPassT/FailStdT

print("Urban Pass Probability:",UrbanPassProbT)
print("Urban Not Pass Probability:",UrbanNotPassProbT)
print("Rural Pass Probability:",RuralPassProbT)
print("Rural Not Pass Probability:",RuralNotPassProbT)

TogetherPassProbT = TogetherPassT/PassStdT
TogetherNotPassProbT = TogetherFailT/FailStdT
ApartPassProbT = ApartPassT/PassStdT
ApartNotPassProbT = ApartFailT/FailStdT

print("Together Pass Probability:",TogetherPassProbT)
print("Together Not Pass Probability:",TogetherNotPassProbT)
print("Apart Pass Probability:",ApartPassProbT)
print("Apart Not Pass Probability:",ApartNotPassProbT)

print("Urban Area given Pass Students:",UrbanPassT)
print("Urban Area given Fail Students:",UrbanNotPassT)
print("Rural Area given Pass Students:",RuralPassT)
print("Rural Area given Fail Students:",RuralNotPassT)

print("Together given Pass Students:",TogetherPassT)
print("Together given Fail Students:",TogetherFailT)
print("Apart given Pass Students:",ApartPassT)
print("Apart given Fail Students:",ApartFailT)

print("-------------Bayes Calculation-------------------")

P9 = (UrbanPassProbT*TogetherPassProbT*PassStdProbTest)/(TogetherPassProbT*PassStdProbTest*UrbanPassProbT + UrbanNotPassProbT*TogetherNotPassProbT*FailStdProbTest)
P10 = (UrbanPassProbT*ApartPassProbT*PassStdProbTest)/(UrbanPassProbT*ApartPassProbT*PassStdProbTest + UrbanNotPassProbT*ApartNotPassProbT*FailStdProbTest)
P11 = (RuralPassProbT*ApartPassProbT*PassStdProbTest)/(RuralPassProbT*ApartPassProbT*PassStdProbTest + RuralNotPassProbT*ApartNotPassProbT*FailStdProbTest)
P12 = (RuralPassProbT*TogetherPassProbT*PassStdProbTest)/(RuralPassProbT*TogetherPassProbT*PassStdProbTest + RuralNotPassProbT*TogetherNotPassProbT*FailStdProbTest)
P13 = (UrbanNotPassProbT*TogetherNotPassProbT*FailStdProbTest)/(UrbanPassProbT*TogetherPassProbT*PassStdProbTest + UrbanNotPassProbT*TogetherNotPassProbT*FailStdProbTest)
P14 = (RuralNotPassProbT*ApartNotPassProbT*FailStdProbTest)/(RuralNotPassProbT*ApartNotPassProbT*FailStdProbTest + RuralPassProbT*ApartPassProbT*PassStdProbTest)
P15 = (UrbanNotPassProbT*ApartNotPassProbT*FailStdProbTest)/(UrbanNotPassProbT*ApartNotPassProbT*FailStdProbTest + UrbanPassProbT*ApartPassProbT*PassStdProbTest)
P16 = (RuralNotPassProbT*TogetherNotPassProbT*FailStdProbTest)/(RuralNotPassProbT*TogetherNotPassProbT*FailStdProbTest + RuralPassProbT*TogetherPassProbT*PassStdProbTest)
print(P9)
print(P10)
print(P11)
print(P12)
print(P13)
print(P14)
print(P15)
print(P16)

fn = 0
tn = 0
fp = 0
tp = 0

for i in range(train_len,dataset_len):
	if values[i][labels.index('G3')] >= 10:
		if values[i][labels.index('address')] == "U" and (values[i][labels.index('Pstatus')] == "A"): #pass/pass
			tp+=1
		elif values[i][labels.index('address')] == "U" and (values[i][labels.index('Pstatus')] == "T"): #pass/pass
			tp+=1
		elif values[i][labels.index('address')] == "R" and (values[i][labels.index('Pstatus')] == "T"): #pass/pass
			tp+=1
		elif values[i][labels.index('address')] == "R" and (values[i][labels.index('Pstatus')] == "A"): #pass/pass
			tp+=1
	else:
		if values[i][labels.index('address')] == "U" and (values[i][labels.index('Pstatus')] == "A"): #pass/fail
			fn+=1
		elif values[i][labels.index('address')] == "U" and (values[i][labels.index('Pstatus')] == "T"): #pass/fail
			fn+=1
		elif values[i][labels.index('address')] == "R" and (values[i][labels.index('Pstatus')] == "T"): #pass/fail
			fn+=1
		elif values[i][labels.index('address')] == "R" and (values[i][labels.index('Pstatus')] == "A"): #pass/fail
			fn+=1	
	
print("---------------Contigency Table----------------------")
print(str("\n   ")+str("Pass")+  "  |"  +str("  ")+str("Fail"))
print(str("Pass")+ "| " +str(tp) +str("   ")+  str(fn))
print(str("Fail")+ "| " +str(fp) + str("   ")+  str(tn))

jaccard_coefficient = tp/(fp+tp+fn)
print("\nJaccard Coefficient: ",jaccard_coefficient)
