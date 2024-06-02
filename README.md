# 从零入手人工智能
从零入手人工智能（1）——卷积神经网络
## 1.前言

本人作为一名单片机工程师，近期对人工智能领域产生了浓厚的兴趣，并开始了我的探索之旅。人工智能是一个博大精深的领域，我相信有许多志同道合的朋友也希望涉足这个领域，因此我写下这篇文章，旨在详细记录我学习人工智能的每一个步骤和心得，希望能为想要入门人工智能的朋友们提供一份入门的指南。**为了激发大家的兴趣，我将直接从卷积神经网络这一热门话题入手，带大家领略人工智能的魅力和乐趣**。
我们不去深入探讨卷积神经网络底层是如何工作的，而是选择直接通过实践来体验人工智能是如何进行学习和训练的。在这个过程中，我们采取了一种“**先上车，再买票**”的学习策略，通过这种策略，我们能够通过实践快速上手，然后在后续的学习中逐步深化对知识的理解。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ad1bf25e29014b6090a840aa4f240a3f.png#pic_center)

## 2.开发环境

本代码是在2024年基于当前的编程环境写的，随着技术的快速发展和官方API函数的不断更新，**后期可能会出现因API函数修改而导致部分代码编译不通过的情况**。
笔者也是借鉴了一位前辈的代码，由于技术的更新，即使是经过验证的代码也可能因为API函数的变化而出现报错。当我们遇到此类问题时，请保持冷静！**首先我们可以阅读官方文档，了解API函数的最新变动和更新内容；同时我们可以利用互联网搜索相关的解决方案和修改建议**。
我提供了当前代码的开发环境信息，开发环境如下：

```c
python 3.12.2
ancand 2.5.2
jupyter 7.0.8
pandas 2.2.1
numpy 1.26.4 
keras  3.3.3
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b0c1dcef44a34e20b87d4d698e8d359c.png#pic_center)

## 3.准备工作

我们的目标是：**对两种不同类型的图片进行识别分类**。本代码用卷积神经网络实现了对猫和狗这两类图片的识别分类。为了识别这两类图片，我们需要建立卷积神经网络模型，并用若干张不同的猫和狗的图片训练模型。
在开始构建模型之前，**我们需要一定的图片数据集来训练我们的卷积神经网络模型**。为了识别猫和狗这两类图片，我们需要在网络上下载并整理这些图片数据。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/abc646b079644c8b84c6599b1c64023c.png#pic_center)
在这个过程中下载猫和狗图片各35张，保存为JGP格式，我们将这些图片按照特定的组织方式存放到Jupyter工程对应的路径中

> upyter 的默认工作目录为：C:\Users\Administrator

在“工程目录”文件夹内我们创建了一个dataset的文件夹，**在dataset文件夹中有三个子文件夹：cats、dogs和test**。cats文件夹中存放了30张猫的图片，dogs文件夹中存放了30张狗的图片，这些图片将作为训练数据用于模型的学习。而test文件夹则用于存放测试数据，我们存放了5张猫的图片和5张狗的图片，用于评估模型在未见过的数据上的表现。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f6c75d6c5df44f34b6ecd9092bdb9985.png#pic_center)

## 4.代码实现流程

**本代码的核心任务是利用VGG16（经典的卷积神经网络模型）对图片进行特征提取和转换，并将这些转换后的数据作为神经网络的输入进行训练**。训练完成后，我们将使用测试图片来验证模型的准确性。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/203946eb1119470bbd97247d6086d87c.png#pic_center)
本代码有4个主要步骤：**测试VGG16输出数据类型，批量数据预处理，建立模型及训练，测试模型**。

### 4.1.测试VGG16输出数据类型

我们要用VGG16（经典的卷积神经网络模型）对图片进行特征提取和数据转换，随后将得到的转换数据输入到神经网络中。既然要把VGG16转换后的数据输入到神经网络中，**那么我们首先需要搞清楚VGG16预处理后的数据格式是什么样的**？只有这样我们才能定义数据接口匹配的神经网络模型。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/6f8254c3a082478589a2cef62ff649ff.png#pic_center)
首先加载一张训练图片，并对其进行缩放和归一化，以适应VGG16模型的输入要求。然后使用VGG16模型对图片进行特征提取，**最终输出一个包含丰富特征信息的数据，我们查看这个预处理数据类型和大小**，方便我们后面构建神经网络模型。

### 4.2.批量数据预处理

使用VGG16模型对保存不同路径下用于训练的猫和狗的图片进行预处理，并定义一个缓存储存预处理后的数据。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/8ce46816a00f469a8c6fd45491a19e9b.png#pic_center)

### 4.3.建立模型及训练

   **基于VGG16提取的特征向量构建一个神经网络模型，这个模型是一个的全连接网络**。我们使用VGG16转换后的数据集来训练这个神经网络模型。在训练过程中，模型会学习从特征向量中识别出图片所属类别。训练完成后，我们会计算模型在训练集上的准确性，以评估其性能。
   
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/5633c0e496d94acb8415e1660cc0a65a.png#pic_center)

### 4.4.测试模型

 我们准备了9张独立的测试图片，**这些图片在训练过程中是未被使用过的的**。使用这些测试图片来验证模型的准确性和泛化能力是非常重要的，因为它可以帮助我们了解模型在新数据上的表现。
  将测试图片通过相同的VGG16预处理和特征提取流程，得到对应的特征向量。然后将这些特征向量输入到已经训练好的神经网络模型中，得到模型对每个测试图片的预测结果。
  
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b43bd789699043a19d0bf20ac36358cb.png#pic_center)

## 5.代码讲解
将测试我们使用jupyter编程工具，这个工具的最大优势是可以分段运行，下面的代码讲解也是分段描述的，废话不多说，直接上源码。

```c
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
```

> 这段代码的作用是加载一张训练图片，将图片转换成一个数组，然后显示图片大小和内容，以下是程序运行结果，根据结果我们可知图片数据是一个224*224*3的三维数组。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/7d5c414329df418198e04c96041e2149.png#pic_center)

```c
import numpy as np
from keras.applications.vgg16 import preprocess_input
#在数组的指定位置插入新的维度 ,使得数据与卷积神经网络的模型兼容
debug_data = np.expand_dims(debug_img ,axis=0)
debug_data = preprocess_input(debug_data)
print(debug_data.shape)
```

> 这段代码的作用是将图片数据增加一个维度与后面的VGG16模型数据匹配，程序运行结果如下，有结果可知图片转换为一个1*224*224*3 的四维数组


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/eb97cd6f217d4be69ae6b246c09185fa.png#pic_center)

```c
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

model_vgg = VGG16(weights='imagenet' ,include_top=False)  # 加载 VGG16 模型  
vgg16_out = model_vgg.predict(debug_data)   #  使用VGG16 模型转换  debug_data ，得到 vgg16_out
print(vgg16_out.shape)  # 显示 vgg16_out大小
```

> 这段代码的作用是使用VGG16模型对图片数据进行数据处理操作，运行结果如下，根据结果可知VGG16转换后得到的数据是一个1* 7 * 7*512的四维数组


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/d3e97c47aced4581bbc5b50a44763ef4.png#pic_center)

```c
vgg16_out = vgg16_out.reshape(1,7*7*512) #将vgg16_out 转换成一维向量 (1,7*7*512)
print(vgg16_out.shape) 
```

> 这段代码的作用是将VGG16处理得到的数据转换成一个 二维数组格式，因为只有这种数据格式才能匹配全连接神经网络输入，代码运行结果如下，根据运行结果可知输入神经网络的数据类型为1*25088（25088=7 * 7 *512）。



![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/06f12a5afdc34a12851d736ff6f4a30a.png#pic_center)
至此我们终于搞清楚了经过VGG16模型处理后的图片数据是什么类型，神经网络输入的数据是什么类型，**至此我们完成了四大步骤中的步骤一**。

```c
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
```

> 这段代码的的作用是搜素dataset/cats和dataset/dogs这两个文件夹中的所有图片，并使用VGG16模型对这些图片进行处理，处理后的数据保存到features_cats和features_dogs中，同时我们建立猫和狗的标签label_cat和label_dog。**这里我们注意一个关键点，在代码中已经将猫和狗的图片与0和1两个标签进行了绑定**。程序运行后对数据进行处理，整个过程持续约30秒，运行结果如下。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f1bead9f1c904468bb250eab9339b00f.png#pic_center)

```c
from sklearn.model_selection import train_test_split
#划分测试集，将data_set 和  label  划分成一个训练数据集和一个数据测试集
X_train,X_test,y_train,y_test = train_test_split(data_set ,label ,test_size=0.3 ,random_state=5)
print(X_train.shape ,X_test.shape ,y_train.shape ,y_test.shape) #查看训练集和测试集
```

> 这段代码的作用是划分测试集，将data_set 和  label  划分成一个训练数据集和一个数据测试集，运行结果如下。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/8cfba8308ffa413c86891bd31f33ab89.png#pic_center)
我们使用VGG16模型将所有训练图片进行了批量处理，**至此我们完成了四大步骤中的步骤二**。

```c
from keras.models import Sequential
from keras.layers import Dense
# 构建神经网络模型 两个全连接层 
model = Sequential()
model.add(Dense(units=10,activation='relu',input_dim=7*7*512))  # 输入层 大小 7*7*512
model.add(Dense(units=1,activation='sigmoid'))                  # 输出层  1 ，二分类
model.summary()
#定义loss opt acc基本参数
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
```

> 这段代码的作用是建立一个全连接的二层神经网络模型，神经网络模型的输入大小为7 * 7 * 512，代码运行结果如下。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/6442ffafa8034718a181f86fcee1e832.png#pic_center)

```c
#使用训练数据对模型进行训练
model.fit(X_train,y_train,epochs=50)
```
> 这段代码的作用是使用X_train和y_train数据训练模型，代码运行结果如下。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/6a7f15c87eba4b539a949d36e05482b3.png#pic_center)

```c
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
```

> 这段代码的额作用是使用测试数据集测试模型准确性，代码运行结果如下。



![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/d2a70ce5dcc4463ba201589d5e240aa8.png#pic_center)


我们建立了神经网络模型，并使用了VGG16处理的数据作为输入，最终组成了的卷积神经网络模型，完成模型训练后并对模型的准确性进行了测试，**至此我们完成了四大步骤中的步骤三**。


```c
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
```

> 这段代码的作用是在全新的测试图片中加载一张图片，先用VGG16模型进行数据处理，然后用神经网络模型进行数据预测，代码运行结果如下。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/d870618ccbce4056babe7fbf568b24b9.png#pic_center)

```c
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
```

这段代码的作用是使用test测试图片对模型进行批量测试，注意这里的图片命名必须为1，2，3....10，运行结果如下。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/9e765cb68cf54dce8a7ba35f9c00e037.png#pic_center)
我们用9张全新的图片对模型进行了测试，其中9张图片预测对了8张，准确率相当不错，要知道这只是用30张图片进行了训练，如果我们加大图片数量模型的准确性还将大幅提高，**至此我们完成了四大步骤中的步骤四**。

## 5.改变训练目标

很多时候我们学习了一个别人的代码，当我们想进行一定的修改的时候，发现修改一处就到处报错，修改已至无从下手。接下来我在教大家如何修改训练目标，比如我们想识别卡车和轿车这两种图片，步骤如下：

### 步骤1

下载图片。在“工程目录”文件夹内我们创建了一个dataset的文件夹，**在dataset文件夹中有三个子文件夹：cars、trunks和test**。cars文件夹中存放了30张轿车的图片，trunks文件夹中存放了30张卡车的图片，test文件夹存放5张轿车的图片和5张卡车的图片片，用于评估模型在未见过的数据上的表现。
### 步骤2

修改路径。修改VGG16数据处理代码中的图片加载路径。

```c
folder_path = "dataset/cars"  
folder_path = "dataset/trunks"
```

### 步骤3
修改图片批量测试中的图片显示代码。

```c
if result >0.5 :
    plt.title('car' ,fontdict={ 'size': '10'})
else :
    plt.title('trunk' ,fontdict={ 'size': '10'})
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/1ce5585ea26f4b5a99f057ad19a8816a.png#pic_center)
