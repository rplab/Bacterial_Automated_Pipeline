


from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import pickle
import numpy as np
from random import random
from collections import OrderedDict

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
tableau10New = [(78, 120, 166), (242, 141, 46), (224, 88, 89), (117, 182, 177), (89, 160, 79),
                (237, 200, 73), (174, 121, 160)]
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)
for i in range(len(tableau10New)):
    r, g, b = tableau10New[i]
    tableau10New[i] = (r / 255., g / 255., b / 255.)


# Labels_all has the form  [convnet_pred, rf_pred, svc_pred, true_labels]
labels_all = pickle.load(open('cls_rpts/all_preds_e_1_k2_[2,5,5]_rot_120_withCholera', 'rb'))
labels_all0 = labels_all[0]
labels_true = pickle.load(open('cls_rpts/true_labels', 'rb'))
labels_all1 = pickle.load(open('cls_rpts/datasetComp/datasets_conv_2', 'rb'))[:11]
labels_all2 = pickle.load(open('cls_rpts/datasetComp/datasets_conv_2', 'rb'))[11:22]
labels_all3 = pickle.load(open('cls_rpts/datasetComp/datasets_conv_2', 'rb'))[22::]
accs0 = [accuracy_score(labels_all0[i], labels_true[i]) for i in range(11)]
accs1 = [accuracy_score(np.array(labels_all1)[i, 1], np.array(labels_all1)[i, 2]) for i in range(len(labels_all[0]))]
accs2 = [accuracy_score(np.array(labels_all2)[i, 1], np.array(labels_all2)[i, 2]) for i in range(len(labels_all[0]))]
accs3 = [accuracy_score(np.array(labels_all3)[i, 1], np.array(labels_all3)[i, 2]) for i in range(len(labels_all[0]))]
accs = [accs0, accs1, accs2, accs3]
empty_accs = [i[:2] for i in accs]
other_accs = [i[2:] for i in accs]


dotsize = 9
fontsize = 18
alpha = 0.5

plt.figure(figsize=(6, 4))
ax = plt.subplot(111)
plt.xticks([i for i in range(-1, 9)] + [11, 12])
ax.get_xaxis().tick_bottom()
ax.set_xticklabels([])
ax.get_yaxis().tick_left()
for x in range(0, 9):
    plt.plot((x, x), (0, 1), '--', lw=0.5, color="black", alpha=0.3)
for x in range(11, 13):
    plt.plot((x, x), (0, 1), '--', lw=0.5, color="black", alpha=0.3)
for i in range(12, 20):
    plt.plot(range(-1, 9), [i*0.05 for n in range(-1, 9)], "--", lw=0.5, color="black", alpha=0.3)
    plt.plot([i for i in range(11, 13)], [i*0.05 for n in [i for i in range(11, 13)]], "--", lw=0.5, color="black", alpha=0.3)
for j in range(len(accs)):
    sorted_accuracies = other_accs[j]
    empty_accuracies = empty_accs[j]
    plt.plot([i for i in sorted_accuracies], 'o', markersize=dotsize, color=tableau10New[0], label='ConvNet', alpha=alpha)
    plt.plot([i for i in sorted_accuracies], color=tableau10New[0], alpha=alpha)
    plt.plot([11, 12], [i for i in empty_accuracies], 'o', markersize=dotsize, color=tableau10New[0], alpha=alpha)



labels_all = pickle.load(open('cls_rpts/all_preds_e_1_k2_[2,5,5]_rot_120_withCholera', 'rb'))
labels_all0_rf = labels_all[1]
labels_all0_svc = labels_all[2]
labels_true = pickle.load(open('cls_rpts/true_labels', 'rb'))
labels_all1 = pickle.load(open('cls_rpts/datasetComp/datasets_svc_rf_0', 'rb'))
labels_all2 = pickle.load(open('cls_rpts/datasetComp/datasets_svc_rf_1', 'rb'))
labels_all3 = pickle.load(open('cls_rpts/datasetComp/datasets_svc_rf_2', 'rb'))
accs0_rf = [accuracy_score(labels_all0_rf[i], labels_true[i]) for i in range(11)]
accs1_rf = [accuracy_score(np.array(labels_all1)[i, 0], np.array(labels_all1)[i, 2]) for i in range(len(labels_all[0]))]
accs2_rf = [accuracy_score(np.array(labels_all2)[i, 0], np.array(labels_all2)[i, 2]) for i in range(len(labels_all[0]))]
accs3_rf = [accuracy_score(np.array(labels_all3)[i, 0], np.array(labels_all3)[i, 2]) for i in range(len(labels_all[0]))]
accs_rf = [accs0_rf, accs1_rf, accs2_rf, accs3_rf]
empty_accs_rf = [i[:2] for i in accs_rf]
other_accs_rf = [i[2:] for i in accs_rf]
accs0_svc = [accuracy_score(labels_all0_svc[i], labels_true[i]) for i in range(11)]
accs1_svc = [accuracy_score(np.array(labels_all1)[i, 1], np.array(labels_all1)[i, 2]) for i in range(len(labels_all[0]))]
accs2_svc = [accuracy_score(np.array(labels_all2)[i, 1], np.array(labels_all2)[i, 2]) for i in range(len(labels_all[0]))]
accs3_svc = [accuracy_score(np.array(labels_all3)[i, 1], np.array(labels_all3)[i, 2]) for i in range(len(labels_all[0]))]
accs_svc = [accs0_svc, accs1_svc, accs2_svc, accs3_svc]
empty_accs_svc = [i[:2] for i in accs_svc]
other_accs_svc = [i[2:] for i in accs_svc]

for j in range(len(accs)):
    sorted_accuracies_rf = other_accs_rf[j]
    empty_accuracies_rf = empty_accs_rf[j]
    plt.plot([i for i in sorted_accuracies_rf], 'o', markersize=dotsize, color=tableau10New[4], label='RF', alpha=alpha)
    plt.plot([i for i in sorted_accuracies_rf], color=tableau10New[4], alpha=alpha)
    plt.plot([11, 12], [i for i in empty_accuracies_rf], 'o', markersize=dotsize, color=tableau10New[4], alpha=alpha)
    sorted_accuracies_svc = other_accs_svc[j]
    empty_accuracies_svc = empty_accs_svc[j]
    plt.plot([i for i in sorted_accuracies_svc], 'o', markersize=dotsize, color=tableau10New[2], label='SVC', alpha=alpha)
    plt.plot([i for i in sorted_accuracies_svc], color=tableau10New[2], alpha=alpha)
    plt.plot([11, 12], [i for i in empty_accuracies_svc], 'o', markersize=dotsize, color=tableau10New[2], alpha=alpha)



plt.plot((9.5, 9.5), (0, 1), '-', lw=0.5, color="black")
plt.plot((13.5, 13.5), (0, 1), '-', lw=0.5, color="black")
plt.xticks(fontsize=fontsize)
plt.xlim(-1, 13.5)
plt.ylim((0.55, 1))
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
leg = plt.legend(by_label.values(), by_label.keys(), framealpha=0.5, edgecolor='k', loc=3, numpoints=1, fontsize=fontsize-2)
for lh in leg.legendHandles:
    lh._legmarker.set_alpha(0.8)
plt.ylabel('Accuracy', fontsize=fontsize)
plt.text(1.5, 0.51, 'Vibrio Datasets', fontsize=fontsize+1)
plt.text(10.1, 0.51, 'Empty', fontsize=fontsize+1)
