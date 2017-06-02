import numpy as np
import matplotlib.pyplot as plt
# COST
general = np.load('/home/bea/Desktop/THEANO/redes/ejecucion1/general_svm_frav/cost.npy')
minidataset = np.load('redes/ejecucion1/general_svm_frav/minidataset/cost.npy')
tested_itself = np.load('redes/ejecucion1/general_svm_frav/minidataset_tested_itself/cost.npy')
lr_0_001 = np.load('redes/ejecucion1/general_svm_frav/minidataset_tested_iteself_lr_0_001/cost.npy')

y_general_cost = np.linspace(0,1, len(general))
y_experiment_cost = np.linspace(0,1,len(minidataset))

# PCA_DescTree_fpr = np.load('frav_feat/SVM_RBF-fpr.npy')
# PCA_DescTree_tpr = np.load('frav_feat/SVM_RBF-tpr.npy')

lw = 2

plt.clf()
plt.plot( y_general_cost, general,lw=lw, color='blue', label='General experiment')
plt.plot(y_experiment_cost, minidataset, lw=lw, color='orange', label='Experiment 2 and 3')
# plt.plot(y_experiment_cost, minidataset,  lw=lw, color='green', label='Experiment 2')
# plt.plot(y_experiment_cost, tested_itself,   lw=lw, color='magenta', label='Experiment 3')
plt.plot( y_experiment_cost, lr_0_001, lw=lw, color='brown', label='Experiment 4')

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('RGB FRAV DATABASE')
plt.legend(loc="upper right")
# plt.savefig('FRAV_experiments_cost.png')
plt.savefig('FRAV_experiments_cost_2.png')


# ERROR
general = np.load('/home/bea/Desktop/THEANO/redes/ejecucion1/general_svm_frav/error.npy')
minidataset = np.load('redes/ejecucion1/general_svm_frav/minidataset/error.npy')
tested_itself = np.load('redes/ejecucion1/general_svm_frav/minidataset_tested_itself/error.npy')
lr_0_001 = np.load('redes/ejecucion1/general_svm_frav/minidataset_tested_iteself_lr_0_001/error.npy')

# PCA_DescTree_fpr = np.load('frav_feat/SVM_RBF-fpr.npy')
# PCA_DescTree_tpr = np.load('frav_feat/SVM_RBF-tpr.npy')

y_general_error = np.linspace(0,1, len(general))
y_experiment_error = np.linspace(0,1,len(minidataset))

lw = 2

plt.clf()
plt.plot( y_general_error, general, lw=lw, color='blue', label='General experiment')
plt.plot(y_experiment_error, minidataset,  lw=lw, color='green', label='Experiment 2')
plt.plot( y_experiment_error, tested_itself,  lw=lw, color='magenta', label='Experiment 3')
plt.plot(y_experiment_error, lr_0_001, lw=lw, color='brown', label='Experiment 4')

plt.xlabel('Epoch')
plt.ylabel('Error (%)')
plt.title('RGB FRAV DATABASE')
plt.legend(loc="upper right")
plt.savefig('FRAV_experiments_error.png')
