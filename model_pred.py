#!/usr/bin/env python
# coding: utf-8

# In[8]:



from keras.models import Sequential
import cv2
from keras.models import load_model


# In[7]:
img = cv2.imread("4.jpg")
print(img)




CATEGORIES=['Dog','Cat']
def prepare(filepath):
    image_size=60
    img_array=cv2.imread(filepath)
    # print(img_array,'kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
    new_array=cv2.resize(img_array,(image_size,image_size))
    return new_array.reshape(-1,image_size,image_size,1)

model=load_model("63x3-CNN.model")

# print(prepare('/home/ai-9/Documents/uploads/8.jpg'))



prediction=model.predict(prepare('/home/ai-9/Documents/uploads/8.jpg'))
print(prediction)
 

# In[9]:


prediction_=model.predict([prepare('/home/ai-9/Documents/uploads/4.jpg')])
print(CATEGORIES[int(prediction_[0][0])])


# In[5]:





# In[ ]:




