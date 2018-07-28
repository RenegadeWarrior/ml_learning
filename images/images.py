import numpy as np
from PIL import Image
imagelist=[];
width=60
height=60
image_Number=15
for i in range(1,image_Number+1):
    image=Image.open("c"+str(i)+".jpg")
    image_Resized=image.resize((width,height))
    imagelist.append(np.array(image_Resized))
Y=np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0])
images=np.array(imagelist)