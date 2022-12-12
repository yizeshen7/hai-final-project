from PIL import Image
import numpy as np
import cv2 as cv
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from cmu_112_graphics import *

root= Tk()
instruction = Frame(root)
root.geometry("750x750")
root.title("Image Colorization")

def select_file():
   path = filedialog.askopenfilename(title="Select an Image", filetype=(('image    files','*.jpg'),('all files','*.*')))
   img = Image.open(path)
   img = ImageTk.PhotoImage(img)
   label = Label(root, image= img)
   label.image = img
   label.pack()
   
def change_to_root():
   root.pack(fill='both', expand=1)
   instruction.pack_forget()

def change_to_instruction():
   instruction.pack(fill='both', expand=1)
   root.pack_forget()

def raise_frame(frame):
    frame.tkraise()

Label(root, text="Welcome to", font=('Caveat 18 bold')).pack(pady=20)
Label(root, text="Image Colorization Tool", font=('Caveat 18 bold')).pack(pady=5)
buttonImage = ttk.Button(root, text="Select to Open a file", command= select_file)
buttonInstructions = ttk.Button(root, text="read instructions", command= change_to_instruction)
buttonRoot = ttk.Button(instruction, text="Go Back", command= change_to_root)
Label(instruction, text="Instruction", font=('Caveat 18 bold')).pack(pady=5)
Label(instruction, text="This tool aims to help you restore black and white images into colors, you can select the image that you wish to colorize through the 'Select to Open a file' button, and click 'Colorize' for python to call the colorization funciton. The app currently only supports black and white image to color and output images once it has around 85 percent accuracy. You can toggle settings in the colorizer file", wraplength=300,  font=('Caveat 14')).pack(pady=10)
buttonImage.pack(ipadx=5, pady=15)
buttonInstructions.pack(ipadx=5, pady=20)

# model 
def appStarted(app):
    app.location = "home"
    app.image = "haiyi-bw.jpg"
    

# controller
image = 'haiyi-bw.jpg'
def colorize(image):
    frame = cv.imread(image)
    Caffe_net = cv.dnn.readNetFromCaffe("./colorization_deploy_v2.prototxt", "./colorization_release_v2.caffemodel")

    # load black and white images
    Caffe_net.getLayer(Caffe_net.getLayerId('class8_ab')).blobs = [numpy_file.astype(np.float32)]
    Caffe_net.getLayer(Caffe_net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    # extracting L channel, resize the image
    input_width = 224
    input_height = 224
    rgb_img = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
    lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
    l_channel = lab_img[:,:,0] 
    l_channel_resize = cv.resize(l_channel, (input_width, input_height)) 
    l_channel_resize -= 50

    # Predicting ab channel using caffe pretrained model
    Caffe_net.setInput(cv.dnn.blobFromImage(l_channel_resize))
    ab_channel = Caffe_net.forward()[0,:,:,:].transpose((1,2,0)) 
    (original_height,original_width) = rgb_img.shape[:2] 
    ab_channel_us = cv.resize(ab_channel, (original_width, original_height))
    lab_output = np.concatenate((l_channel[:,:,np.newaxis],ab_channel_us),axis=2) 
    bgr_output = np.clip(cv.cvtColor(lab_output, cv.COLOR_Lab2BGR), 0, 1)
    cv.imwrite("./result.png", (bgr_output*255).astype(np.uint8))

    numpy_file = np.load(image)
    Caffe_net = cv.dnn.readNetFromCaffe("./models/colorization_deploy_v2.prototxt", "./models/colorization_release_v2.caffemodel")
    numpy_file = numpy_file.transpose().reshape(2, 313, 1, 1)
    load = Image.open("./result.png")
    load = load.resize((480, 360), Image.ANTIALIAS)
 
 
def drawStarterCanvas(app, canvas):
    canvas.create_rectangle(700, 500, 800, 600, fill = "black", outline = "black")
    # draw the menu bars
    canvas.create_text(500, 400, text = "Welcome", font = "Roboto 28 Bold", fill = "blue")
    canvas.create_text(500, 470, text = "Select Image", font = "Roboto 28", fill = "red")
    canvas.create_text(500, 540, text = "Instructions", font = "Roboto 28", fill = "blue")

def redrawAll(app, canvas):
    if app.location == "home":
        drawStarterCanvas(app, canvas)
    if app.location == "instrcution":
        drawStarterCanvas(app, canvas)

# runApp(width = 800, height = 600)
root.mainloop()