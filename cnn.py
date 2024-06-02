from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img,img_to_array
%matplotlib inline

#加载图片  转换图片成数组 
img_path = 'dataset/test/1.jpg'
debug_img = load_img(img_path,target_size=(224,224))
debug_img = img_to_array(debug_img)
print(debug_img.shape)

#显示图片 
fig = plt.figure(figsize=(3,3))
show_img = load_img(img_path,target_size=(224,224))
plt.imshow(show_img)



import numpy as np
from keras.applications.vgg16 import preprocess_input
#在数组的指定位置插入新的维度 ,使得数据与卷积神经网络的模型兼容
debug_data = np.expand_dims(debug_img ,axis=0)
debug_data = preprocess_input(debug_data)
print(debug_data.shape)


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

model_vgg = VGG16(weights='imagenet' ,include_top=False)  # 加载 VGG16 模型  
vgg16_out = model_vgg.predict(debug_data)   #  使用VGG16 模型转换  debug_data ，得到 vgg16_out
print(vgg16_out.shape)  # 显示 vgg16_out大小

#改变数据维度为全连接层准备 ，神经网络的全连接层的输入并不直接支持多维数据。当处理多维数据时要将数据展平为一维向量
vgg16_out = vgg16_out.reshape(1,7*7*512) #将vgg16_out 转换成一维向量 (1,7*7*512)
print(vgg16_out.shape) 


# 使用VGG16提取全部的猫和狗训练图片的特性数据
model_vgg = VGG16(weights='imagenet', include_top=False)

def model_process(img_path,model):   # 使用模型数据处理函数
    img = load_img(img_path, target_size=(224, 224))     #加载路径
    img = img_to_array(img)
    x = np.expand_dims(img,axis=0) #在数组的指定位置插入新的维度
    x = preprocess_input(x)
    
    x_vgg = model_vgg.predict(x) #使用 VGG16 卷积层特征提取器
    x_vgg = x_vgg.reshape(1,7*7*512) #将数据转换成一维向量 (1,7*7*512)
    return x_vgg
    
import os
folder_path = "dataset/cats"
folder_name = os.listdir(folder_path)  #listdir 返回一个列表，包含路径path下所有文件的名字

img_path = []
for i in folder_name: 
    #splitext返回一个包含两个元素的元组，第一个元素是文件名（不包括扩展名），第二个元素是扩展名（如果有的话，包含前面的点）
    if os.path.splitext(i)[1] == ".jpg":   
        img_path.append(i) # 图片的名字加入到img_path中
        
img_path = [folder_path + "//" + i for i in img_path] #img_path转变为一个包含路径的图片名称

#定义1个保存个样本1图片*7*7*512的数组 ，用于存储N个样本1图片经过VGG16转换后的所有数据
features_cats = np.zeros([len(img_path),7*7*512])
for i in range(len(img_path)):
    #使用 VGG16 卷积层特征提取器
    feature_i = model_process(img_path[i],model_vgg)
    print('preprocessed:',img_path[i])
    features_cats[i] = feature_i
    
folder_path = "dataset/dogs"
folder_name = os.listdir(folder_path)
img_path = []
for i in folder_name:                             
    if os.path.splitext(i)[1] == ".jpg":   
        img_path.append(i)  # 图片的名字加入到img_path中

img_path = [folder_path + "//" + i for i in img_path] #img_path转变为一个包含路径的图片名称

#定义1个保存个样本2图片*7*7*512的数组 ，用于存储N个样本2图片经过VGG16转换后的所有数据
features_dogs = np.zeros([len(img_path),7*7*512])
for i in range(len(img_path)):
     #使用 VGG16 卷积层特征提取器
    feature_i = model_process(img_path[i],model_vgg)
    print('preprocessed:',img_path[i])
    features_dogs[i] = feature_i
#建立标签    这里数组大小需要根据具体的样本数量来确定
label_cat = np.zeros(30)  #y1 猫的额标签值 0
label_dog = np.ones(30)   #y2 狗的额标签值 1

#concatenate  将多个数组“拼接”成一个更大的数组 
data_set = np.concatenate((features_cats ,features_dogs) ,axis=0) #将猫和狗的特征数据合并

label = np.concatenate((label_cat ,label_dog) ,axis=0)  #将猫和狗的标签数据合并
label = label.reshape(-1,1) #将一个一维数组y转换为一个二维数组

#l显示数据的大小
print(features_cats.shape,features_dogs.shape)
print(data_set.shape,label.shape)



from sklearn.model_selection import train_test_split
#划分测试集，将data_set 和  label  划分成一个训练数据集和一个数据测试集
X_train,X_test,y_train,y_test = train_test_split(data_set ,label ,test_size=0.3 ,random_state=5)
print(X_train.shape ,X_test.shape ,y_train.shape ,y_test.shape) #查看训练集和测试集


from keras.models import Sequential
from keras.layers import Dense
# 构建神经网络模型 两个全连接层 
model = Sequential()
model.add(Dense(units=10,activation='relu',input_dim=7*7*512))  # 输入层 大小 7*7*512
model.add(Dense(units=1,activation='sigmoid'))                  # 输出层  1 ，二分类
model.summary()
#定义loss opt acc基本参数
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#使用训练数据对模型进行训练
model.fit(X_train,y_train,epochs=50)

from sklearn.metrics import accuracy_score
import numpy as np

#测试模型准确率
def predict_data(train):
    predict_ret = model.predict(train)
    predict_ret=np.argmax(predict_ret,axis=1)
    return predict_ret

#使用测试数据集测试模型准确性
y_test_predict =  predict_data(X_test)
accuracy_test = accuracy_score(y_test,y_test_predict)
print(accuracy_test)

# 在模型未接触的TEST数据集中，找一个图片进行测试
# 加载图片
img_path = 'dataset/test/3.jpg'
img = load_img(img_path,target_size=(224,224))
img_show = load_img(img_path,target_size=(224,224))

# VGG16对图片进行转换
img = img_to_array(img)
x = np.expand_dims(img,axis=0)
x = preprocess_input(x)
features = model_vgg.predict(x)
features = features.reshape(1,7*7*512)
# 使用模型对图片进行测试
result = model.predict(features)#本模型是一个二分类模型，因此预测返回值result是一个概率值，，概率接近0说明是猫 概率接近1说明是狗
print(result) #显示预测值
#用图片名称显示预测值
if result >0.5 :
    plt.title("dog")
else :
    plt.title("cat")
    
plt.imshow(img_show)


#使用test测试图片对模型进行批量测试
test_path = 'dataset/test/'
a = [i for i in range(1,10)]
fig = plt.figure(figsize=(10,10))
#依次加载处理所有图片
for i in a: 
    #加载图片
    img_name = test_path + str(i) +'.jpg'
    img_path = img_name
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    #增加数据维度  预处理数据  使用 VGG16 卷积层特征提取器
    x = np.expand_dims(img,axis=0)
    x = preprocess_input(x)
    x_vgg = model_vgg.predict(x)
    x_vgg = x_vgg.reshape(1,7*7*512)
    #使用模型进行数据结果预测
    result = model.predict(x_vgg)
    print(result)
    #加载图片用于显示
    img_ori = load_img(img_name, target_size=(250, 250))
    plt.subplot(3,3,i)
    plt.imshow(img_ori)
    if result >0.5 :
        plt.title('dog' ,fontdict={ 'size': '10'})
    else :
        plt.title('cat' ,fontdict={ 'size': '10'})
plt.tight_layout() #调整图片大小 防止重叠
plt.show()