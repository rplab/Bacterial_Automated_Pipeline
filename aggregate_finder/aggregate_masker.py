
from matplotlib import pyplot as plt
import glob
import os
import re
from skimage.morphology import binary_closing, binary_opening, binary_erosion, disk
from skimage import morphology
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from skimage import restoration, segmentation, filters
from skimage.segmentation import join_segmentations
from scipy import ndimage as ndi
from skimage.restoration import denoise_wavelet, cycle_spin, denoise_bilateral
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from tkinter import *
from tkinter import ttk
from aggregate_finder import aggregate_masker_operations as amo
from matplotlib.figure import Figure


############### Image Directory ####
ask_user_image_directory= input('Copy and paste the location of your first image and press enter')

ImgFiles = glob.glob(ask_user_image_directory + '/*.tif')
amo.sort_nicely(ImgFiles)
region, color = amo.find_region_and_color(ImgFiles)
first_image = plt.imread(ImgFiles[0])



######### Segment aggregates or save null mask? ###############
###############################################################

ask_user_first = input('Have you worked on this image before?')
if ask_user_first == 'Y':
    aggregate_mask_3D = np.load(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_aggregate_mask.npz')['arr_0']
    if len(np.load(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_aggregate_mask.npz').files) == 1:
        cur = 0
        thresh_array = thresh_array = [0]*len(ImgFiles)
    else:
        cur = np.load(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_aggregate_mask.npz')['arr_1']
        thresh_array = np.load(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_aggregate_mask.npz')['arr_2']
else:
    aggregate_mask_3D = np.empty_like([first_image] * len(ImgFiles), dtype=bool)
    cur = 0
    thresh_array = thresh_array = [0] * len(ImgFiles)
ask_user = input('Are there aggregates you would like to segment? Y or N')
if ask_user == 'N':
    print('generating empty aggregate mask')
    np.savez_compressed(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_aggregate_mask', np.array(aggregate_mask_3D))
    sys.exit()



##############################################################
##############################################################
#### Load gutmask from directory and import images+denoising ####

gutmask_3D, nogutmask = amo.load_gutmask(ImgFiles, first_image)

try:
    image_3D
except NameError:
    image_3D = amo.denoising_filter_image_stack(ImgFiles)

####generate empty 2D image for random walker markers ####

#markers = np.zeros_like(first_image)




###########################################################################################################################################################
###########################################################################################################################################################

######Application######

#### initialize global parameters
### Cur is the image number,initializes with aggregate, mask, array, gutmask variable tells the application whether to use the gutmask or not, defaulted to apply gutmask #####
## x and y coordinates needed for lasso selection
x1 = 0
x2 = 0
y1 = 0
y2 = 0
gutmask = 1
filter = 0

x, y = np.meshgrid(np.arange(first_image.shape[1]), np.arange(first_image.shape[0]))
pix = np.vstack((x.flatten(), y.flatten())).T
mask = np.zeros_like(first_image, dtype=bool)
thresh_global = 0

#### Create window, size ##########
window = Tk()
window.geometry("1400x1200")
window.title('Aggregate Masker')

########## Grid Layout #############

def onselect(verts):
    global mask
    mask = np.zeros_like(first_image,dtype=bool)
    p = Path(verts)
    ind = p.contains_points(pix, radius=1)
    mask.flat[ind] = True

##### Images on grid the image (row = 1-8,column 0) ###########################
fig = Figure(figsize=(10,8))
a = fig.add_subplot(111)

image =image_3D[cur]
a.imshow(image, cmap= 'gray')
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().grid(row=1, column=0, rowspan=8)
canvas.draw()
lasso = LassoSelector(a, onselect)


####Toolbar for pan/zoom etc######
toolbarFrame = Frame(master=window)
toolbarFrame.grid(row=9, column=0)
toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
toolbar.update()

######Image number (row 0, column 0), update live######################
label_image_number = Label(window,text = 'Image '+str(cur)+' of '+ str(len(ImgFiles)), font = 'bold 30')
label_image_number.grid(row=0, column=0)


##################### Remove or add gut mask functions #############
label_bool_gutmask = Label(window,text = str(gutmask), font = 'bold 30')
label_bool_gutmask.grid(row=1, column=2)

def remove_gut_mask():
    global gutmask
    global thresh_global
    gutmask = 0
    label_bool_gutmask.config(text = str(gutmask), font = 'bold 30')
    segment_image(thresh_global)


def add_gut_mask():
    global gutmask
    global thresh_global
    gutmask = 1
    label_bool_gutmask.config(text=str(gutmask), font='bold 30')
    segment_image(thresh_global)

button = Button(window, text = "Remove gut mask", font = 'bold 20',command = remove_gut_mask)
button.grid(row = 0, column=1)

button = Button(window, text = "Add gut mask", font = 'bold 20',command = add_gut_mask)
button.grid(row = 1, column=1)
###########################


################################Segmentation functions ######, takes in the slider value #######
def segment_image(thresh):
    ##### takes the value from the slider function/entry box, the float defines which pixels will be classified as aggregate
    #rw seg:
    ##### The markers acts as labels for random walker segmentation. Mode cg-mg is optimal for large images. Then
    ##### remove small objects and fill holes, fast median fliter smoothens masked objects.The output is saved as a boolean array.
    global canvas
    global cur
    global gutmask
    global aggregate_mask_3D
    global filter
    global thresh_array

    thresh_array[cur] = float(thresh)
    if gutmask == 0:
        gut_mask = nogutmask[0]
    else:
        gut_mask = gutmask_3D[cur]

    image = image_3D[cur]

    #only for rw
    #markers[image < float(thresh)] = 1
    #markers[image > float(thresh)] = 2
    #markers[image > 0.8] = 2
    #thresh_image = segmentation.random_walker(image, markers, beta=10, mode='cg_mg')

    thresh_image = image<float(thresh)

    ##### post processing #####
    remove_small_objects = morphology.remove_small_holes((thresh_image-1).astype(bool), 1000, connectivity=2)
    remove_small_objects = ndi.percentile_filter(remove_small_objects, percentile = 70, size = 5)
    remove_small_objects = morphology.remove_small_objects(remove_small_objects, 1000, connectivity=1)
    remove_small_objects = np.logical_and(gut_mask,remove_small_objects)
    aggregate_mask_3D[cur] = remove_small_objects

    ### display scharr filter if filter tool is selected####
    if filter ==1:
        grad_image = filters.sobel(image)

    a = fig.add_subplot(111)
    a.imshow(image, cmap= 'gray')

    if filter == 1:
        a.imshow(grad_image, cmap='pink', alpha=0.5)
    a.imshow(remove_small_objects,'gray_r', alpha=0.2)
    canvas.draw()

### Entry window for thresholding ###

label_threshold_value = Label(window,text = 'Enter threshold value', font = 'bold 20')
label_threshold_value.grid(row = 9, column=1)

entry_threshold_value = Entry(window)
entry_threshold_value.grid(row = 10, column =1)



############################### Add and Remove objects from mask ################################
def add_to_mask():
    global mask
    global cur
    global aggregate_mask_3D
    global canvas
    global filter

    image = image_3D[cur]
    aggregate_mask_3D[cur] = mask  + aggregate_mask_3D[cur]
    np.savez_compressed(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_aggregate_mask', np.array(aggregate_mask_3D))
    if filter ==1:
        grad_image = filters.sobel(image)

    a = fig.add_subplot(111)
    a.imshow(image, cmap= 'gray')

    if filter ==1:
        a.imshow(grad_image, cmap = 'pink', alpha=0.5)

    a.imshow(aggregate_mask_3D[cur],'gray_r', alpha=0.2)

    canvas.draw()

    mask = np.zeros_like(first_image,dtype=bool)

button_remove = Button(window, text = "Add selection", font = 'bold 20',command = add_to_mask)
button_remove.grid(row = 3, column=1)


def remove_from_mask():

    global mask
    global cur
    global aggregate_mask_3D
    global canvas
    global filter

    image = image_3D[cur]

    orred_mask = np.logical_or(aggregate_mask_3D[cur], mask)
    aggregate_mask_3D[cur] = np.logical_xor(orred_mask, mask)

    np.savez_compressed(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_aggregate_mask', np.array(aggregate_mask_3D))

    if filter ==1:
        grad_image = filters.sobel(image)

    a = fig.add_subplot(111)
    a.imshow(image, cmap= 'gray')

    if filter ==1:
        a.imshow(grad_image, cmap = 'pink', alpha=0.3)

    a.imshow(aggregate_mask_3D[cur], 'gray_r', alpha=0.2)

    canvas.draw()

    mask = np.zeros_like(first_image,dtype=bool)



button_remove = Button(window, text = "Remove selection", font = 'bold 20',command = remove_from_mask)
button_remove.grid(row = 4, column=1)


def segment_entry_window(event):
    global thresh_global
    thresh_global = float(entry_threshold_value.get())
    segment_image(thresh_global)
######## Add or remove gradient filter #########

def add_gradient_filter():
    global filter
    global thresh_global
    filter = 1
    segment_image(thresh_global)


button_remove = Button(window, text="Add gradient filter", font='bold 20', command = add_gradient_filter)
button_remove.grid(row=5, column=1)

def remove_gradient_filter():
    global filter
    global thresh_global
    filter = 0
    segment_image(thresh_global)

button_remove = Button(window, text="Remove gradient filter", font='bold 20', command = remove_gradient_filter)
button_remove.grid(row=6, column=1)


################### slider properties and functions ###################################

def update_value(event):
    global thresh_global
    thresh_global = slider.get()
    segment_image(thresh_global)

slider = Scale(window, from_=0, to=1, tickinterval=0.2,length = 600,resolution=0.05, orient=HORIZONTAL)
slider.bind("<ButtonRelease-1>", update_value)
slider.grid(row=10, column=0)


################# moving forward and backward through the image stack #################

def load_image_forward(event):
    global canvas
    global cur
    global thresh_global

    if cur==len(ImgFiles)-1:
        cur=0
    else:
        cur +=1
    image = image_3D[cur]
    segment_image(thresh_global)
    label_image_number.config(text='Image ' + str(cur) + ' of ' + str(len(ImgFiles)))


def load_image_backward(event):

    global canvas
    global cur
    global thresh_global
    if cur==0:
        cur=len(ImgFiles)-1
    else:
        cur -=1

    image = image_3D[cur]
    a = fig.add_subplot(111)
    a.imshow(image, cmap= 'gray')
    segment_image(thresh_global)

    canvas.draw()
    label_image_number.config(text='Image ' + str(cur) + ' of ' + str(len(ImgFiles)))

######## mask five images at once with the slider selection #################

def update_next_five(event):
    global thresh_global
    mask_next_five(thresh_global)

def mask_next_five(thresh,gutmask=1):
    global cur
    global aggregate_mask_3D
    global thresh_global

    for b in range(5):
        global cur
        cur +=1
        segment_image(thresh_global)

    load_image_forward(None)
    label_image_number.config(text='Image ' + str(cur) + ' of ' + str(len(ImgFiles)))

######### move to any image in the image stack ##############

def makesomething(value):
    global cur
    cur = int(value)-1
    load_image_forward(None)

entry_image_number = Entry(window)
entry_image_number.grid(row=7, column=1)
button = Button(window, text = "Go to image", font = 'bold 20',command = lambda: makesomething(entry_image_number.get()))
button.grid(row = 8, column=1)

################# close window ##############################

def quit_me():
    if len(aggregate_mask_3D)>0:
        np.savez_compressed(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_aggregate_mask', np.array(aggregate_mask_3D),cur,np.array(thresh_array))
    window.quit()
    window.destroy()
    print('quit')


window.protocol("WM_DELETE_WINDOW", quit_me)

window.bind("<Shift_L>", update_next_five)
window.bind("<Return>", segment_entry_window)
window.bind("<Right>",load_image_forward)
window.bind("<Left>", load_image_backward)
window.mainloop()

