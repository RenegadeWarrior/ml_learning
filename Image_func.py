import numpy as np
from PIL import Image
def image_array():
    #create empty image list
    imagelist=[];
    #define width and height of standardized images
    width=60
    height=60
    Image_Number=25
    for i in range(1,Image_Number+1):
        image=Image.open("Images/c"+str(i)+".jpg")
        image_Resized=image.resize((width,height))
        imagelist.append(np.array(image_Resized))
    Y=np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    Y=Y.reshape((1,Image_Number))
    images=np.array(imagelist)
    return images,Y