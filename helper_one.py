import constants as c

import pandas as pd
import numpy as np
from skimage import io
import os.path
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import exists
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox




def fake_img(number):
    '''
    input is the image counter and the database,
    from that it creates a fake image width all detected particles
    '''
    #Getting DB
    # db = pd.read_feather(c.DBPATH)
    
    # Getting Imagename
    imagename = 'Os7-S1 Camera'+ number

    # Filter the database according to the particles occuring in the image
    db_img = st.session_state['db'][st.session_state['db']['filename'] == imagename]

    # creating empty dummy image
    dummy_img = np.ones((c.IMAGESIZE_X, c.IMAGESIZE_Y), dtype=np.uint8)*c.BACKGROUNDCOLOR

    # loop over every entry and add it to the image
    db_img = db_img.reset_index()  # make sure indexes pair with number of rows

    for index, row in db_img.iterrows():
        # Get local Variables
        left = int(row['left'])
        right = int(row['right'])
        top = int(row['top'])
        bottom = int(row['bottom'])
        # size_pixelcount = row['size_pixelcount']
        # sharpness = row['sharpness']
        # interestingness = row['interestingness']
        order = row['order']
        
        width = right - left
        height = bottom - top
        
        # st.write([left, right, top, bottom])
        
        #load cropped image if exists
        cropped_imgpath = c.CROPPEDIMAGEPATH + '/' + str(order) + '_' + imagename + '.png'
        
        # load whole image path in case whole image got downloaded
        imagepath = c.CROPPEDIMAGEPATH + '/' + imagename + '.png'
        
        #offset = np.array((left, top))
        
        if os.path.exists(imagepath):
            img = io.imread(imagepath) 
        elif os.path.exists(cropped_imgpath):
            cropped_img = io.imread(cropped_imgpath)
            dummy_img[top : (top + height), left : (left + width)] = cropped_img
            # convert the dummy image-array to an image
            img = Image.fromarray(dummy_img, 'L')
        else:
            cropped_img = np.ones((height, width), dtype=np.uint8)
            dummy_img[top : (top + height), left : (left + width)] = cropped_img
            # convert the dummy image-array to an image
            img = Image.fromarray(dummy_img, 'L')
        
        
        
    #img.save('my.png')
    # img.show()
    return [img, db_img]

def getfeaturespace(db):
    # Getting features
    x = np.array(db['size_pixelcount'])
    y = np.array(db['sharpness'])

    # color by status if received on earth (green), else (blue)
    order = db['order']
    filename = db['filename']


    images_received_x = []
    images_received_y = []
    images_received =[]
    images_notreceived_x = []
    images_notreceived_y = []

    for i in range(0,len(x)):
        # ord = '%06d' % order[i]
        ord = str(order[i])
        pathtofile = c.CROPPEDIMAGEPATH + '/' + ord + '_' + filename[i] + '.png'
        file_exists = exists(pathtofile)
        if file_exists:
            # color.append('g')
            images_received_x.append(x[i])
            images_received_y.append(y[i])
        else:
            # color.append('k')
            images_notreceived_x.append(x[i])
            images_notreceived_y.append(y[i])
            
    images_received = [images_received_x, images_received_y]
    images_notreceived = [images_notreceived_x, images_notreceived_y]
    
    return images_received, images_notreceived
    
def plot_featurespace():

    # Fixing random state for reproducibility
    # Fixing random state for reproducibility
    # np.random.seed(19680801)


    db = pd.read_feather(c.DBPATH)
    # Getting features
    x = np.log(np.array(db['size_pixelcount']))
    y = np.log(np.array(db['sharpness']))

    # color by status if received on earth (green), else (blue)
    order = db['order']
    filename = db['filename']

    color = []
    for i in range(0,len(x)):
        # ord = '%06d' % order[i]
        ord = str(order[i])
        pathtofile = c.CROPPEDIMAGEPATH + '/' + ord + '_' + filename[i] + '.png'
        file_exists = exists(pathtofile)
        if file_exists:
            color.append('g')
        else:
            color.append('k')
            
    opacity = st.sidebar.slider('Opacity', 0.0, 1.0, 0.5)
    size = st.sidebar.slider('Markersize',0,10,2)

    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    # fig, ax = plt.subplots()

    # the scatter plot:
    ax.scatter(x, y, s= size, c = color, alpha = opacity)
    plt.xlabel('size')
    plt.ylabel('sharpness')
    
    
    st.pyplot(fig)
    


    
def fake_img_order(order):

    fn = order2image(order)
    
    fn = '%06d' % fn #converting to 6 digit string

    # get the dummy-image
    [img, db_img] = fake_img(fn)
    # display the dummy-image
    st.image(img)
    st.caption('Os7-S1 Camera' + str(fn))
    
    #intitializing the database to display
    db_display = db_img["order"]

    # checkboxes for displaying the table
    #interestingness
    interest = st.sidebar.checkbox("interestingness", True)
    if interest:
        db_display = pd.concat([db_display, db_img['interestingness']], axis=1)
        
    # size_pixelcount
    size = st.sidebar.checkbox("size_pixelcount", True)
    if size:
        db_display = pd.concat([db_display, db_img['size_pixelcount']], axis=1)
        
    # sharpness
    sharpness = st.sidebar.checkbox("sharpness", True)
    if sharpness:
        db_display = pd.concat([db_display, db_img['sharpness']], axis=1)
        
    # boxposition and size
    box = st.sidebar.checkbox("Box position and dimensions", False)
    if box:
        db_display = pd.concat([db_display, db_img['xpos'], db_img['ypos'], db_img['width'], db_img['height']], axis=1)
    
    # write the table corresponding to the image
    st.write(db_display)

            

    # display the dummy-image
    # cv2.imshow('Dummy-image: ' + imagename ,dummy_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #display the image
    # imagepath = '/home/auth/Documents/Projekte/aid_icaps/inputdata/MINIMAL_EXAMPLE/images/' + imagename + '.png'
    # orig_img = io.imread(cropped_imgpath)
    # cv2.imshow('Original: ' + imagename , orig_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # convert the dummy image-array to an image
    # img = Image.fromarray(dummy_img, 'L')

    
def order2imagename(order):
    # import Data
    db = pd.read_feather(c.DBPATH)

    # filter for order
    df = db[db['order'] == order]
    filename = df['filename'].tolist()[0]
    fn = int(filename.split('Camera')[1])
    return fn

def getinterestingness(db):
    grad_interestingness = [c.weight_size, c.weight_sharpness]
    # getting the ordering function to display
    x1 = [10.0,3.0] # size
    x2 = [-12,-6] # sharpness
    return x1, x2


# some minimal example functions ------------------------------------------------------
def test():
    st.write('hello')
    
def calc_interestingness(feature_sharpness, feature_size):
    interestingness = np.multiply(c.weight_sharpness * feature_sharpness, c.weight_size * feature_size)
    return interestingness

def normalizing(nparray):
    # shifting width lower value to 0
    nparray = nparray - nparray.min()
    
    minvalue = nparray.min()
    maxvalue = nparray.max()
    diff = maxvalue - minvalue
    
    # normalizing (range/span)
    norm_array = (maxvalue - nparray)/diff
    
    # converting to uint8
    arrayuint8 = (norm_array*255).astype('uint8')
    
    return arrayuint8

def iap(x,y,colorcode,db):
    path = r'./data/train_particles/'
    
    arr = []
    scale = 1
    filename = []
    
    for row in db.itertuples():
        fname = row.cropedimage
        img = mpimg.imread(path + fname)
        arr.append(img)
        filename.append(fname)
    arr = np.asarray(arr, dtype=object)
    #print('arr shape/size/type: ' + str(arr[0].shape))
    
    
    # colors = {'0': 'red','1':'blue'}
    # create figure and plot scatter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    line = ax.scatter(x, y, 
                      marker="o", 
                      facecolor=colorcode, 
                      alpha=1,
                      #color='k', 
                      s=15)
    plt.title("Featurspace")
    plt.ylabel("sharpness [-]")
    plt.xlabel("size [n_pixel]")
    # fig.colorbar(line)
    
    # create the annotations box
    im = OffsetImage(arr[0],
                     zoom=1, 
                     cmap=plt.cm.gray_r)
    
    xybox = (50., 50.)  # Position der Box
    
    ab = AnnotationBbox(im, (0, 0), xybox=xybox, xycoords='data',
                        boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(visible=True, which="both")
    #plt.xscale('symlog')
    #plt.yscale('symlog')
    #ax.legend()
    
    #xdata = np.array([17, 250, 700, 1700])
    #ydata = np.array([0.08, 0.02, 0.01, 0.006])
    #ax.scatter(xdata,ydata,color='k')
    
    #xdata = np.array([4,7])
    #ydata = np.array([-3,-4.8])
    #ax.plot(xdata, ydata)
    
    x2=np.linspace(2,10,100)
    
    m = -2.4/4
    b = 0.1
    x1 = np.e**x2
    fx = np.e**(m*x2 + b)
    ax.plot(x1,fx)

    # # linear
    # c1 = -3.231050962185965e-05
    # c2 = 0.05054303229037492
    # f_lin = c1*x2 + c2
    # ax.plot(x2,f_lin)
    
    # # quadratic
    # c3 = 6.362489230517171e-08
    # c4 = -0.00014606818581024098
    # c5 = 0.07162919305036503
    # f_quad = c3*x2**2 + c4*x2 + c5
    # ax.plot(x2,f_quad)
    
    # # exp
    # c6 = -241.96286732898602
    # c7 = 6.943480651959799e-05
    # c8 = 242.08573676379947
    # f_exp = c6*x2**c7 + c8
    # ax.plot(x2, f_exp)
    
    # kubisch
    #a = 6.304678557418634e-07
    #b = -0.001198377835938013
    #c = 0.2701902180006368
    #f=a*x**2 + b*x + c
    


    
    
    # Logistic Fuction:
    #x1 = np.linspace(0,10000,1000)
    #L = 1 # supremum of the values of the function
    #k = 1/1000 # logistic growth rate or steepness of the curve
    #x0 = 150 #  the x value of the function's midpoint
    #f = L/(1+np.exp((-1*k)*(x1-x0)))
    #ax.plot(x1,f)
    
    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            # find out the index within the array from the event
            ind, = line.contains(event)[1]["ind"]
            # get the figure size
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0] * ws, xybox[1] * hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy = (x[ind], y[ind])
            # set the image corresponding to that point
            im.set_data(1-arr[ind]) # 1- um die Farben um zu kehren
            # write down Frame and Position
            print('Frame: ' + filename[ind])
        else:
            # if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()
    
    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.show()