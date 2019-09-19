

from unet.data_processing import *
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, ball
from matplotlib import pyplot as plt


directory = '/media/parthasarathy/af969b3d-e298-4407-98c2-27368a8eba9f/multispecies_image_data/'
files = glob(directory + '**/*.npz', recursive=True)
files = [file for file in files if 'objs' in file]
files_di = [file for file in files if 'ae_en' in file]
files_di_en = [file for file in files_di if '488' in file]
sort_nicely(files_di_en)
files_en = [file for file in files if 'mono/en' in file]
sort_nicely(files_en)
fish_di_en = iter(files_di_en)
fish_di_en = [[x, next(fish_di_en)] for x in fish_di_en]
files_di_ae = [file for file in files_di if '568' in file]
sort_nicely(files_di_ae)
files_ae = [file for file in files if 'mono/a01' in file]
sort_nicely(files_ae)
fish_di_ae = iter(files_di_ae)
fish_di_ae = [[x, next(fish_di_ae)] for x in fish_di_ae]

#  EN
largest_cluster_di_en = []
for regions in fish_di_en:
    ob = []
    for file in regions:
        print(file)
        if np.load(file)['objs']!=[]:
            objs = np.load(file)['objs']
            print(np.sort([np.sum(item) for item in objs]))
            ob.append(np.sort([np.sum(item) for item in objs])[-1])
        else:
            ob.append(1)

    print(ob)
    largest_cluster_di_en.append(np.max(ob))

fish_mono_en = iter(files_en)
fish_mono_en = [[x, next(fish_mono_en)] for x in fish_mono_en]
largest_cluster_mono_en = []
for regions in fish_mono_en:
    ob = []
    for file in regions:
        print(file)
        if np.load(file)['objs']!=[]:
            objs = np.load(file)['objs']
            # print([np.shape(item) for item in objs])
            print(np.sort([np.sum(item) for item in objs]))
            ob.append(np.sort([np.sum(item)  for item in objs])[-1])
        else:
            ob.append(1)

    print(ob)
    largest_cluster_mono_en.append(np.max(ob))

#  AE
largest_cluster_di_ae = []
for regions in fish_di_ae:
    ob = []
    for file in regions:
        print(file)
        if np.load(file)['objs']!=[]:
            objs = np.load(file)['objs']
            # plt.figure()
            # plt.imshow(np.sum(sorted(objs, key= lambda x: np.sum(x))[-1], axis=0))
            print(np.sort([np.sum(item) for item in objs]))
            ob.append(np.sort([np.sum(item) for item in objs])[-1])
        else:
            ob.append(1)
    print(ob)
    largest_cluster_di_ae.append(np.max(ob))

fish_mono_ae = iter(files_ae)
fish_mono_ae = [[x, next(fish_mono_ae)] for x in fish_mono_ae]
largest_cluster_mono_ae = []
for regions in fish_mono_ae:
    ob = []
    for file in regions:
        print(file)
        if np.load(file)['objs']!=[]:
            objs = np.load(file)['objs']
            print(np.sort([np.sum(item) for item in objs])[-1])
            ob.append(np.sort([np.sum(item) for item in objs])[-1])
        else:
            ob.append(1)

    print(ob)
    largest_cluster_mono_ae.append(np.max(ob))


fontsize = 28
markersize = 18
alpha = 0.5
scale = 0.1625**2  # Microns^3 / Pixel^3
plt.figure(figsize=([6.4 * 2, 4.8 * 2]))
plt.plot([1 + np.random.normal()*0.01 for i in range(len(largest_cluster_mono_ae))], np.log10(np.array(largest_cluster_mono_ae)*scale),
         'o', markersize=markersize, alpha=alpha)
plt.boxplot(np.log10(np.array(largest_cluster_mono_ae)*scale), positions=[1])
plt.plot([2 + np.random.normal()*0.01 for i in range(len(largest_cluster_di_ae))], np.log10(np.array(largest_cluster_di_ae)*scale),
         'o', markersize=markersize, alpha=alpha)
plt.boxplot(np.log10(np.array(largest_cluster_di_ae)*scale), positions=[2])
plt.plot([3 + np.random.normal()*0.01 for i in range(len(largest_cluster_mono_en))], np.log10(np.array(largest_cluster_mono_en)*scale),
         'o', markersize=markersize, alpha=alpha)
plt.boxplot(np.log10(np.array(largest_cluster_mono_en)*scale), positions=[3])
plt.plot([4 + np.random.normal()*0.01 for i in range(len(largest_cluster_di_en))], np.log10(np.array(largest_cluster_di_en)*scale),
         'o', markersize=markersize, alpha=alpha)
plt.boxplot(np.log10(np.array(largest_cluster_di_en)*scale), positions=[4])
# plt.plot([1, 2, 3, 4], [np.mean(np.log(largest_cluster_mono_en)), np.mean(np.log(largest_cluster_di_en)),
#                      np.mean(np.log(largest_cluster_mono_ae)), np.mean(np.log(largest_cluster_di_ae))], 'o', color='black')
plt.xticks([1, 2, 3, 4], ['AE-mono', 'AE-di', 'EN-mono', 'EN-di'], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim([0, 5])
plt.ylim([1.5, 6])



# fontsize = 28
# markersize = 18
# alpha = 0.5
# plt.figure(figsize=([6.4 * 2, 4.8 * 2]))
# plt.plot([1 + np.random.normal()*0.01 for i in range(len(largest_cluster_mono_ae))], largest_cluster_mono_ae,
#          'o', markersize=markersize, alpha=alpha)
# plt.boxplot(largest_cluster_mono_ae, positions=[1])
# plt.plot([2 + np.random.normal()*0.01 for i in range(len(largest_cluster_di_ae))], largest_cluster_di_ae,
#          'o', markersize=markersize, alpha=alpha)
# plt.boxplot(largest_cluster_di_ae, positions=[2])
# plt.plot([3 + np.random.normal()*0.01 for i in range(len(largest_cluster_mono_en))], largest_cluster_mono_en,
#          'o', markersize=markersize, alpha=alpha)
# plt.boxplot(largest_cluster_mono_en, positions=[3])
# plt.plot([4 + np.random.normal()*0.01 for i in range(len(largest_cluster_di_en))], largest_cluster_di_en,
#          'o', markersize=markersize, alpha=alpha)
# plt.boxplot(largest_cluster_di_en, positions=[4])
# # plt.plot([1, 2, 3, 4], [np.mean(np.log(largest_cluster_mono_en)), np.mean(np.log(largest_cluster_di_en)),
# #                      np.mean(np.log(largest_cluster_mono_ae)), np.mean(np.log(largest_cluster_di_ae))], 'o', color='black')
# plt.xticks([1, 2, 3, 4], ['AE-mono', 'AE-di', 'EN-mono', 'EN-di'], fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.xlim([0, 5])



