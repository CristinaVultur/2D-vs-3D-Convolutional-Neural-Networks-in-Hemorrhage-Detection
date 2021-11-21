from tkinter import *
from GUI.gui_helper import make_prediction
from PIL import ImageTk, Image
import pydicom
import os
import numpy as np

valid = 'valid.csv'
path = '../rsna-intracranial-hemorrhage-detection/stage_2_train/'


def button_clear():  # delete from input box
    e.delete(0, END)


def choose():  # chose image from input box and show predictions

    dcm = e.get()
    data = pydicom.dcmread(os.path.join(path, dcm + '.dcm'))
    new_image = data.pixel_array.astype(float)  # get image array
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    final_image.save(dcm + '.jpg')
    # write jpg image

    my_img = ImageTk.PhotoImage(Image.open(dcm + '.jpg'))
    my_lbl = Label(root, image=my_img)

    my_lbl.grid(row=6, column=1)

    target, outputs, loss_t, accuracy, h_score, h_loss, precision, f1, recall, negative_recall = make_prediction(dcm,
                                                                                                                 valid)
    myLabel = Label(root, text=accuracy)
    myLabel.grid(row=3, column=1)


root = Tk()  # root window
root.title("Choose Image for Prediction")
root.iconbitmap('mind_brain_icon-icons.com_51079.ico')

myLabel = Label(root, text='Detect Hemorrhages')

myLabel.grid(row=0, column=1)

e = Entry(root, width=50, bg='#845555', borderwidth=5)  # input box
e.grid(row=1, column=0, columnspan=3, padx=10, pady=10)
e.insert(0, "Enter Image ID")

button_choose = Button(root, text='Choose CT slice', padx=30, pady=30, command=choose, fg='#845555',
                       bg='#c4e5eb')  # state = disabled
button_choose.grid(row=2, column=0, columnspan=3)

button_clear = Button(root, text="Clear", command=button_clear)
button_clear.grid(row=4, column=1)

button_quit = Button(root, text='Exit Program', command=root.quit)
button_quit.grid(row=5, column=1)

root.mainloop()
