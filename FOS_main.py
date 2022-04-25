# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

import Fair_OS as sv

from aif360.algorithms.preprocessing.reweighing1 import Reweighing
from aif360.datasets import GermanDataset
from aif360.datasets import AdultDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
    import load_preproc_data_compas

# Classifiers
from sklearn.linear_model import LogisticRegression as LG
from sklearn.svm import SVC as SVM
from common_utils import compute_metrics

np.set_printoptions(threshold=1000)

######dataset selection for AIF360 compatibility######
###############################################################################
#uncomment if use German dataset

"""
dataset_orig = GermanDataset(
    protected_attribute_names=['sex'],           # this dataset also contains protected
                                                 # attribute for "sex" which we do not
                                                 # consider in this evaluation
    privileged_classes=[['male']],#,      # age >=25 is considered privileged
    features_to_drop=['personal_status'] # ignore sex-related attributes
)
      
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
"""

###############################################################################
#uncomment if use Adult dataset
"""
dataset_orig = AdultDataset(protected_attribute_names=['sex'],
                            privileged_classes=[['Male']],
                            features_to_keep=['age', 'education-num'])
  
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
"""

###############################################################################
#uncomment if use Compas dataset

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
dataset_orig = load_preproc_data_compas(['sex'])

protected = 'sex'

###############################################################################

# print out some labels, names, etc.
print("#### Training Dataset shape")
print(dataset_orig.features.shape,type(dataset_orig.features))

print("#### Favorable and unfavorable labels")
print(dataset_orig.favorable_label, dataset_orig.unfavorable_label)

print("#### Protected attribute names")
print(dataset_orig.protected_attribute_names)

print("#### Privileged and unprivileged protected attribute values")
print(dataset_orig.privileged_protected_attributes, 
      dataset_orig.unprivileged_protected_attributes)

print("#### Dataset feature names")
print(dataset_orig.feature_names)
print()

prot_attribs = dataset_orig.protected_attributes

scored = dataset_orig.scores

metrics_list = []

##############################################################################
#classifier selection

#uncomment if use SVM standard classifier
#classifier = SVM(probability=True)
#fname = "svm_FOS.csv" #file name to save the results

#uncomment if use logistic regression classifier
classifier = LG(C=0.2, penalty='l2', solver='liblinear', max_iter=1000)
fname = "lg_FOS.csv"  #file name to save the results

protected = 'sex'
protected_attribute = 'sex'

#set the number of cross-fold validation. Here, it is 1.
fold = 1
count=0

#file path to save the metric results
file1 = ".../stats/" + fname

#location of the data files
filing = '.../data/comp/compas_'
#filing = '.../data/adult/adult_'
#filing = '.../data/german/german_'

#import data files
print('-' * 50)
        
file = filing + 'train_' + str(count) + '.csv'
pdf = pd.read_csv(file)
X_train = pdf.to_numpy()
        
file = filing + 'trn_label_' + str(count) + '.csv'
pdf = pd.read_csv(file)
y_train = pdf.to_numpy()
y_train = np.squeeze(y_train)
    
file = filing + 'test_' + str(count) + '.csv'
pdf = pd.read_csv(file)
X_test = pdf.to_numpy()
        
file = filing + 'test_label_' + str(count) + '.csv'
pdf = pd.read_csv(file)
y_test = pdf.to_numpy()
y_test = np.squeeze(y_test)
        
file = filing + 'trn_pdf_' + str(count) + '.csv'
trn_pdf = pd.read_csv(file)
        
names = list(trn_pdf.columns)
print('names ',names)
        
file = filing + 'tst_pdf_' + str(count) + '.csv'
tst_pdf = pd.read_csv(file)
        
prot_idx = trn_pdf.columns.get_loc(protected)
print('protected idx ',prot_idx)

prob_idx = trn_pdf.columns.get_loc('Probability')
print('probability idx ',prob_idx)
        
prot_attribs_test = X_test[:,prot_idx]
        
prot_attribs_train = X_train[:,prot_idx]
print('attributes train ',prot_attribs_train.shape)
        
test_scores = y_test
train_scores = y_train
        
print()
    
dataset_orig_train = dataset_orig.copy(deepcopy=True)
    
dataset_orig_train.features = np.copy(X_train)
dataset_orig_train.labels = np.copy(np.expand_dims(y_train, axis=1))
wts = np.ones((len(X_train)), dtype=np.float64)
dataset_orig_train.instance_weights = wts
    
dataset_orig_train.protected_attributes = np.copy(np.expand_dims(prot_attribs_train, axis=1))
dataset_orig_train.scores = np.copy(np.expand_dims(train_scores, axis=1))
        
dataset_orig_test = dataset_orig.copy(deepcopy=True)
dataset_orig_test.features = np.copy(X_test)
dataset_orig_test.labels = np.copy(np.expand_dims(y_test, axis=1))
wts = np.ones((len(X_test)), dtype=np.float64)
dataset_orig_test.instance_weights = wts
    
dataset_orig_test.protected_attributes = np.copy(np.expand_dims(prot_attribs_test, axis=1))
dataset_orig_test.scores = np.copy(np.expand_dims(test_scores, axis=1))

#######################################################################
 
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
                              
n, n_fav, n_unfav, n_p, n_up, n_p_fav, n_up_fav, n_p_unfav, \
                n_up_unfav, n_p_fav_ratio, n_up_fav_ratio,\
                n_p_unfav_ratio, n_up_unfav_ratio = RW.fit(dataset_orig_train)

prot_feat = trn_pdf[protected_attribute]
print('prot_feat ', prot_feat)

prot_idx = trn_pdf.columns.get_loc(protected_attribute)
print('prot_idx ',prot_idx) #4

prot_values = X_train[:,prot_idx]

pv_max = np.max(prot_values)
pv_min = np.min(prot_values)
print('max min ',pv_max, pv_min)
    
pv_mid = (pv_max + abs(pv_min)) / 2
print('mid ',pv_mid)
pv_mid_pt = pv_max - pv_mid
print('mid point ',pv_mid_pt)
    
#######################################################################
print()
#determine if favorable or unfavorable is majority class    
if n_unfav > n_fav:
    majority = 0 
else:
    majority = 1 
print('majority class ',majority)
    
if n_p_fav < n_p_unfav:
    print('first')
    nsamp1 = int(n_p_unfav - n_p_fav)
    prot_grp1 = 1 
    if majority == 1: 
        cls_trk1 = 1 
    else:  
        cls_trk1 = 0
    
if  n_p_unfav < n_p_fav:
    nsamp1 = int(n_p_fav - n_p_unfav)
    prot_grp1 = 1 
    if majority == 1: 
        cls_trk1 = 0 
    else:  
        cls_trk1 = 1
    
########################
if n_up_fav < n_up_unfav:
    nsamp2 = int(n_up_unfav - n_up_fav)
    prot_grp2 = 0 
    if majority == 1: 
        cls_trk2 = 1 
    else:  
        cls_trk2 = 0
    
if  n_up_unfav < n_up_fav:
    nsamp2 = int(n_up_fav - n_up_unfav)
    prot_grp2 = 0 
    if majority == 1: 
        cls_trk2 = 0 
    else:  
        cls_trk2 = 1
    
if nsamp1 < nsamp2:
    nsamp = nsamp1
    cls_trk = cls_trk1
    prot_grp = prot_grp1
else:
    nsamp = nsamp2
    cls_trk = cls_trk2
    prot_grp = prot_grp2
    
###################################
###################################
oversampler= sv.FOS_1() 
    
maj_min = cls_trk 
    
print('protected group ', prot_grp)
print('class tracker ',cls_trk)
        
print('number to sample ',nsamp)
print()
    
X_samp, y_samp= oversampler.sample(X_train, y_train, prot_idx, pv_mid_pt,
                    prot_grp, maj_min, nsamp,pv_max,pv_min)
    
######################
print('protected group ',prot_grp)
print('class tracker ',cls_trk)
    
if nsamp1 < nsamp2:
    nsamp = nsamp2
    cls_trk = cls_trk2
    prot_grp = prot_grp2
else:
    nsamp = nsamp1
    cls_trk = cls_trk1
    prot_grp = prot_grp1
    
maj_min = cls_trk 
    
oversampler= sv.FOS_2() 
    
X_samp1, y_samp1= oversampler.sample(X_samp, y_samp, prot_idx, pv_mid_pt,
                    prot_grp, maj_min, nsamp)
                
Xs_train = np.copy(X_samp1)
ys_train = np.copy(y_samp1)
        
classifier.fit(Xs_train, ys_train)
        
###############################################
predictions2 = classifier.predict(X_test)
       
print('############### Metrics after over-sampling:')       
print() 
    
dataset_pred_smote_test = dataset_orig_test.copy(deepcopy=True)

preds2 = np.expand_dims(predictions2, axis=1)

dataset_pred_smote_test.labels = np.copy(preds2)

metrics_t = compute_metrics(dataset_orig_test, dataset_pred_smote_test,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
    
metrics_list.append(metrics_t)

bacc = 0
avg_odds = 0
abs_avg_odds = 0
TNRD = 0
EOD = 0
FU = 0

for met in metrics_list:
    bacc += np.abs(met["Balanced accuracy"])
    avg_odds += np.abs(met["Average odds difference"]) 
    abs_avg_odds += np.abs(met["Absolute average odds difference"]) 
    TNRD += np.abs(met["True negative rate difference"]) 
    EOD += np.abs(met["Equal opportunity difference"]) 
    FU += np.abs(met["Fair utility"])

bacc = bacc/fold
avg_odds = avg_odds/fold
abs_avg_odds = abs_avg_odds/fold
TNRD = TNRD/fold
EOD = EOD/fold
FU = FU/fold


print()
print('Averaged:')
print('BACC ',bacc)
print('AOD ',avg_odds)
print('AAO ',abs_avg_odds)
print('TNRD ',TNRD)
print('EOD ',EOD)
print('Fair Utility ',FU)
print()

###########################################################################

data = np.array([bacc, avg_odds, abs_avg_odds, TNRD, EOD, FU])
data = data.reshape(1,6)
data.shape
data
cols = ['bacc','avg_odds', 'abs_avg_odds', 
        'TNRD', 'EOD', 'Fair_Utility']

df = pd.DataFrame(data=data,columns=cols)
df

df.to_csv(file1,index=False)



















