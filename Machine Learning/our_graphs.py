import matplotlib.pyplot as plt
import numpy as np
import pickle

#TO REPLICATE RESULTS FROM PAPER, GRAPHS ARE REFERENCED FROM CODEBASE

N = 6 # change to 6 and add LSTMU and LSTMB Models
font = 'x-large'
title_font = {'fontname':'Arial', 'size':'24', 'color':'black', 'weight':'normal',
  'verticalalignment':'bottom'}
width = 0.10

##################################################################################

#READ IN CLASSIC ML SCORES FROM PICKLE
f = open('ML_picklefiles/ClassicML_scores.pckl', 'rb')
classic_models = pickle.load(f)
f.close()
#print(classic_models)

N = 4
# Binary CL Graph Models
model_1_metrics = classic_models['binary_naive_bayes']['f1 score'], classic_models['binary_naive_bayes']['precision'], classic_models['binary_naive_bayes']['recall'], classic_models['binary_naive_bayes']['accuracy']
model_1_err = (0, 0, 0, 0)
model_2_metrics =  classic_models['binary_svm']['f1 score'], classic_models['binary_svm']['precision'], classic_models['binary_svm']['recall'], classic_models['binary_svm']['accuracy']
model_2_err = (0, 0, 0, 0)
model_3_metrics =  classic_models['binary_logreg']['f1 score'], classic_models['binary_logreg']['precision'], classic_models['binary_logreg']['recall'], classic_models['binary_logreg']['accuracy']
model_3_err = (0, 0, 0, 0)

fig, ax = plt.subplots(figsize=(15, 10))

ind = np.arange(N)

ax.bar(ind, model_1_metrics, width, label='Naive Bayes', yerr=model_1_err)
ax.bar(ind + width, model_2_metrics, width,
    label='SVM', yerr=model_2_err)
ax.bar(ind + 2*width, model_3_metrics, width,
    label='Logistic Regression', yerr=model_3_err)
ax.legend()

x_labels = ['F1', 'Precision', 'Recall', 'Accuracy']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Metrics', fontsize = font)
ax.set_title('Binary Classical Model Performances', **title_font)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.savefig("Our_Results/Classical_Model_Performances_Binary.png")
#######################################################################
N = 4
# Non-Binary CL Graph Models 2
model_1_metrics =  classic_models['multi_nb']['f1 score'], classic_models['multi_nb']['precision'], classic_models['multi_nb']['recall'], classic_models['multi_nb']['accuracy']
model_1_err = (0, 0, 0, 0)
model_2_metrics = classic_models['multi_svm']['f1 score'], classic_models['multi_svm']['precision'], classic_models['multi_svm']['recall'], classic_models['multi_svm']['accuracy']
model_2_err = (0, 0, 0, 0)
model_3_metrics = classic_models['multi_logreg']['f1 score'], classic_models['multi_logreg']['precision'], classic_models['multi_logreg']['recall'], classic_models['multi_logreg']['accuracy']
model_3_err = (0, 0, 0, 0)

fig, ax = plt.subplots(figsize=(15, 10))

ind = np.arange(N) 

ax.bar(ind, model_1_metrics, width, label='Naive Bayes', yerr=model_1_err)
ax.bar(ind + width, model_2_metrics, width,
    label='SVM', yerr=model_2_err)
ax.bar(ind + 2*width, model_3_metrics, width,
    label='Logistic Regression', yerr=model_3_err)
ax.legend()

x_labels = ['F1', 'Precision', 'Recall', 'Accuracy']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Metrics', fontsize = font)
ax.set_title('Multi-class Classical Model Performances', **title_font)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.savefig("Our_Results/Classical_Model_Performances_Multi.png")
##########################################################################333
# Binary DL Graph Models 2
N=4
err = (0, 0, 0, 0)
#READ IN CNN SCORES FROM PICKLE
f = open('ML_picklefiles/CNN_binary_post.pckl', 'rb')
cnn_binary_pre = pickle.load(f)
f.close()
#print("cnn binary post", cnn_binary_pre)

f = open('ML_picklefiles/CNN_binary_pre.pckl', 'rb')
cnn_binary_post = pickle.load(f)
f.close()
#print("cnn binary pre", cnn_binary_post)

#READ IN LSTMU SCORES FROM PICKLE
f = open('ML_picklefiles/LSTMU_binary_pre.pckl', 'rb')
lstmu_binary_pre = pickle.load(f)
f.close()
print("LSTMU BINARY PRE", lstmu_binary_pre)

f = open('ML_picklefiles/LSTMU_binary_post.pckl', 'rb')
lstmu_binary_post = pickle.load(f)
f.close()
print("LSTMU BINARY POST", lstmu_binary_post)

#READ IN LSTMB SCORES FROM PICKLE
f = open('ML_picklefiles/LSTMB_binary.pckl', 'rb')
lstmb_binary_pre = pickle.load(f)
f.close()
print("LSTMB BINARY PRE", lstmb_binary_pre)

f = open('ML_picklefiles/LSTMB_binary_post.pckl', 'rb')
lstmb_binary_post = pickle.load(f)
f.close()
print("LSTMB BINARY POST", lstmb_binary_post)

model1 = cnn_binary_pre[0][1], cnn_binary_pre[1][1], cnn_binary_pre[2][1], cnn_binary_pre[3][1]
model2 = cnn_binary_post[0][1], cnn_binary_post[1][1], cnn_binary_post[2][1], cnn_binary_post[3][1]
model3 = lstmu_binary_pre[0][1], lstmu_binary_pre[1][1], lstmu_binary_pre[2][1], lstmu_binary_pre[3][1]
model4 = lstmu_binary_post[0][1], lstmu_binary_post[1][1], lstmu_binary_post[2][1], lstmu_binary_post[3][1]
model5 = lstmb_binary_pre[0][1], lstmb_binary_pre[1][1], lstmb_binary_pre[2][1], lstmb_binary_pre[3][1]
model6 = lstmb_binary_post[0][1], lstmb_binary_post[1][1], lstmb_binary_post[2][1], lstmb_binary_post[3][1]

fig, ax = plt.subplots(figsize=(15, 10))

ind = np.arange(N) 

ax.bar(ind, model1, width, label='CNN Pre', yerr=err)
ax.bar(ind + width, model2, width,
    label='CNN Post', yerr=model_2_err)
ax.bar(ind + 2*width, model3, width,
    label='LSTMU Pre', yerr=model_3_err)
ax.bar(ind + 3*width, model4, width, label='LSTMU Post', yerr=err)
ax.bar(ind + 4*width, model5, width,
    label='LSTMB Pre', yerr=err)
ax.bar(ind + 5*width, model6, width,
    label='LSTMB Post', yerr=err)
ax.legend()

x_labels = ['F1', 'Precision', 'Recall', 'Accuracy']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Metrics', fontsize = font)
ax.set_title('Binary DL Model Performances', **title_font)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

plt.savefig("Our_Results/DL_Model_Performances_Binary.png")

###################################################################################

###################################################################################

#READ IN CNN SCORES FROM PICKLE
f = open('ML_picklefiles/CNN_categorical_pre.pckl', 'rb')
cnn_multi = pickle.load(f)
f.close()
#print("CNN MULTI ", cnn_multi)

f = open('ML_picklefiles/CNN_categorical_post.pckl', 'rb')
cnn_multi_post = pickle.load(f)
f.close()
#print("CNN MULTI ", cnn_multi_post)

#READ IN LSTMU SCORES FROM PICKLE
f = open('ML_picklefiles/LSTMU_categorical_pre.pckl', 'rb')
lstmu_multi = pickle.load(f)
f.close()
#print("LSTMU MULTI ", lstmu_multi)

f = open('ML_picklefiles/LSTMU_categorical_post.pckl', 'rb')
lstmu_multi_post = pickle.load(f)
f.close()
#print("LSTMU MULTI ", lstmu_multi_post)

#READ IN LSTMB SCORES FROM PICKLE
f = open('ML_picklefiles/LSTMB_categorical.pckl', 'rb')
lstmb_multi = pickle.load(f)
f.close()
#print("LSTMB MULTI ", lstmb_multi)

f = open('ML_picklefiles/LSTMB_categorical_post.pckl', 'rb')
lstmb_multi_post = pickle.load(f)
f.close()
#print("LSTMB MULTI ", lstmb_multi_post)



N = 4
err = (0, 0, 0, 0)

model1 = cnn_multi[0][1], cnn_multi[1][1], cnn_multi[2][1], cnn_multi[3][1]
model2 = cnn_multi_post[0][1], cnn_multi_post[1][1], cnn_multi_post[2][1], cnn_multi_post[3][1]
model3 = lstmu_multi[0][1], lstmu_multi[1][1], lstmu_multi[2][1], lstmu_multi[3][1]
model4 = lstmu_multi_post[0][1], lstmu_multi_post[1][1], lstmu_multi_post[2][1], lstmu_multi_post[3][1]
model5 = lstmb_multi[0][1], lstmb_multi[1][1], lstmb_multi[2][1], lstmb_multi[3][1]
model6 = lstmb_multi_post[0][1], lstmb_multi_post[1][1], lstmb_multi_post[2][1], lstmb_multi_post[3][1]

#print("model1 ", model1)

fig, ax = plt.subplots(figsize=(15, 10))

ind = np.arange(N) 

ax.bar(ind, model1, width, label='CNN Pre', yerr=err)
ax.bar(ind + width, model2, width,
    label='CNN Post', yerr=model_2_err)
ax.bar(ind + 2*width, model3, width,
    label='LSTMU Pre', yerr=model_3_err)
ax.bar(ind + 3*width, model4, width, label='LSTMU Post', yerr=err)
ax.bar(ind + 4*width, model5, width,
    label='LSTMB Pre', yerr=err)
ax.bar(ind + 5*width, model6, width,
    label='LSTMB Post', yerr=err)
ax.legend()

x_labels = ['F1', 'Precision', 'Recall', 'Accuracy']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Metrics', fontsize = font)
ax.set_title('Multi-class DL Model Performances', **title_font)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.savefig("Our_Results/DL_Model_Performances_Multi.png")

#####################################################################################################3

