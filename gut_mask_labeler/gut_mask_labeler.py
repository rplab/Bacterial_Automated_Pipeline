

from matplotlib import pyplot as plt
import glob
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from skimage import filters
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import *
from aggregate_finder import aggregate_masker_operations as amo
from matplotlib.figure import Figure
""" This code can be used to label gut masks. Simply run the code and enter the file location i.e. the location of your first image- everything before pcoO.tif.
It should load every 5th slice and once can simply use the mouse to draw a mask. Pressing 'return' adds the region to mask. SHIFT-L refreshes the mask and allows
 one to redraw the image. Left and right arrow keys move between every 5th slice. Closing saves the mask in the folder. When restarting the program by rerunning, make sure to answer Y to
 Have you worked on this image before? so that it loads the previously saved mask, else it will generate a new empty mask and overwrite the saved mask. Masks are saves as .npz files"""
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
    gut_mask_3D = np.load(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_gut_mask.npz')['arr_0']
    if len(np.load(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_gut_mask.npz').files) == 1:
        cur = 0
    else:
        cur = np.load(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_gut_mask.npz')['arr_1']
else:
    gut_mask_3D = np.empty_like([first_image] * len(ImgFiles), dtype=bool)
    cur = 0

##############################################################
##############################################################
#### Load gutmask from directory and import images+denoising ####

#gutmask_3D, nogutmask = amo.load_gutmask(ImgFiles, first_image)

try:
    image_3D
except NameError:
    image_3D = amo.load_image_stack(ImgFiles)

####generate empty 2D image for random walker markers ####

""" Application"""""

#### initialize global parameters
### Cur is the image number,initializes with aggregate, mask, array, gutmask variable tells the application whether to use the gutmask or not, defaulted to apply gutmask #####
## x and y coordinates needed for lasso selection
x1 = 0
x2 = 0
y1 = 0
y2 = 0

x, y = np.meshgrid(np.arange(first_image.shape[1]), np.arange(first_image.shape[0]))
pix = np.vstack((x.flatten(), y.flatten())).T
mask = np.zeros_like(first_image, dtype=bool)

#### Create window, size ##########
window = Tk()
window.geometry("1400x1200")
window.title('Gut Masker')

########## Grid Layout #############

def onselect(verts):
    global mask
    mask = np.zeros_like(first_image,dtype=bool)
    p = Path(verts)
    ind = p.contains_points(pix, radius=1)
    mask.flat[ind]=True

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

def add_to_mask(event):
    global mask
    global cur
    global aggregate_mask_3D
    global canvas
    global filter

    image = image_3D[cur]
    gut_mask_3D[cur] = mask  + gut_mask_3D[cur]
    np.savez_compressed(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_gut_mask', np.array(gut_mask_3D))
    if filter ==1:
        grad_image = filters.sobel(image)

    a = fig.add_subplot(111)
    a.imshow(image, cmap= 'gray')

    if filter ==1:
        a.imshow(grad_image, cmap = 'pink', alpha=0.5)

    a.imshow(gut_mask_3D[cur],'gray_r', alpha=0.2)

    canvas.draw()


################# moving forward and backward through the image stack #################

def load_image_forward(event):
    global canvas
    global cur
    global mask

    if cur==len(ImgFiles)-1:
        cur=0
    else:
        cur +=1
    image = image_3D[cur]
    gut_mask_3D[cur] = mask
    np.savez_compressed(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_gut_mask', np.array(gut_mask_3D))

    a = fig.add_subplot(111)
    a.imshow(image, cmap= 'gray')
    a.imshow(gut_mask_3D[cur],'gray_r', alpha=0.2)

    canvas.draw()

    label_image_number.config(text='Image ' + str(cur) + ' of ' + str(len(ImgFiles)))


def load_image_backward(event):

    global canvas
    global cur

    if cur==0:
        cur=len(ImgFiles)-1
    else:
        cur -=1

    image = image_3D[cur]
    a = fig.add_subplot(111)
    a.imshow(image, cmap= 'gray')

    canvas.draw()
    label_image_number.config(text='Image ' + str(cur) + ' of ' + str(len(ImgFiles)))

######## mask five images at once with the slider selection #################



def redraw_selection(event):
    global canvas
    global cur
    global mask

    mask = np.zeros_like(first_image, dtype=bool)
    gut_mask_3D[cur] = mask
    np.savez_compressed(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_gut_mask', np.array(gut_mask_3D))

    image = image_3D[cur]
    a = fig.add_subplot(111)
    a.imshow(image, cmap= 'gray')
    a.imshow(gut_mask_3D[cur],'gray_r', alpha=0.2)

    canvas.draw()


def makesomething(value):
    global cur
    global mask
    cur = int(value)-1
    mask = np.zeros_like(first_image, dtype=bool)
    load_image_forward(None)



entry_image_number = Entry(window)
entry_image_number.grid(row=3, column=1)
button = Button(window, text = "Go to image", font = 'bold 20',command = lambda: makesomething(entry_image_number.get()))
button.grid(row = 2, column=1)

################# close window ##############################

def quit_me():
    if len(gut_mask_3D)>0:
        np.savez_compressed(amo.save_mask_loc(ImgFiles) + 'region_'+str(region) +'_'+str(color) + '_gut_mask', np.array(gut_mask_3D),cur)
    window.quit()
    window.destroy()
    print('quit')


window.protocol("WM_DELETE_WINDOW", quit_me)

window.bind("<Right>",load_image_forward) ### move to the image before the current image
window.bind("<Left>", load_image_backward)  ### move to the image after the current image
window.bind("<Shift_L>", redraw_selection) #### Reset selection tool to redraw the selection
window.bind("<Return>", add_to_mask) #### save selection as gut_mask

window.mainloop()


