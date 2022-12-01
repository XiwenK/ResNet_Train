### Data Characteristics √

- object-occluded image ? Y
- other object in image ?  Y
- CIFAR-10:  50,000 images divide into training set (40,000) & validation set (10,000) random seed 0
  - 分区代码展示
  - 每个类在新训练集中的比例



### Learning Rate - 1

```python
lr = 0.1, 0.01, 0.001
batch_size = 128
data augmentation: random cropping and random horizontal flip
epoch = 15
```

- **Plot**
  - training loss & validation loss against the number of epochs
  - training accuracy & validation accuracy against the number of epochs
- **Report**
  - final losses & accuracy values for both the training set and the validation set
  
    - lr=0.1:
  
      Epoch :  14, training loss : 0.2013, training accuracy : 92.81, test loss : 1.1762, test accuracy : 72.07
    - lr=0.01
  
      Epoch :  14, training loss : 0.0168, training accuracy : 99.50, test loss : 0.9687, test accuracy : 79.77
    - lr=0.001
  
      Epoch :  14, training loss : 0.0021, training accuracy : 100.00, test loss : 1.1931, test accuracy : 71.67
  - which lr better in training loss & training accuracy, which lr better in validation loss & validation accuracy 



### Learning Rate - 2

- **gradual decrease lr** 
  - cosine annealing 
  - minimize training loss
  - epoch = 300
  - learning rate held constant & cosine annealing (final lr = 0)
  
- **Plot**
  - training loss & validation loss against the number of epochs
  - training accuracy & validation accuracy against the number of epochs
  
- **Report**
  - final losses & accuracy values for both the training set and the validation set
  
    - constant lr: 
  
      Epoch : 299, training loss : 0.0000, training accuracy : 100.00, test loss : 1.0325, test accuracy : 82.33
  
    - cosine annealing:
  
      Epoch : 299, training loss : 0.0000, training accuracy : 100.00, test loss : 0.9826, test accuracy : 82.17
  
  - discuss reasons for difference



### Weight Decay

- weight decay: 更新 w
  - epoch = 300
  - weight decay coefficients λ = 0.0005, 0.01
  - best lr & cosine annealing

- **Plot**
  - training loss & validation loss against the number of epochs
  - training accuracy & validation accuracy against the number of epochs
- **Report**
  - final losses & accuracy values for both the training set and the validation set
  
    - 5e-4:
  
      Epoch : 299, training loss : 0.0009, training accuracy : 100.00, test loss : 0.5400, test accuracy : 85.61
  
    - 0.01:
    
      Epoch : 299, training loss : 0.0237, training accuracy : 100.00, test loss : 0.4133, test accuracy : 87.95
    



### Data Augmentation

- Cutout augmentation: torchvision.transforms.RandomErasing
  - best experimental setup
  - value = r_mean, g_mean, b_mean, calculated across entire training set
  - epoch = 300
  - **Optional:** tune cutout hyperparameter
- **Plot**
  - training loss & validation loss against the number of epochs
  - training accuracy & validation accuracy against the number of epochs
- **Report**
  - final losses & accuracy values for both the training set and the validation set
  
    - default : Epoch : 299, training loss : 0.0010, training accuracy : 100.00, test loss : 0.8191, test accuracy : 78.63
    
      Test :  testing loss: 0.6672, testing accuracy: 82.59
    
    - p = 0.8: Epoch : 299, training loss : 0.0011, training accuracy : 100.00, test loss : 0.8794, test accuracy : 77.06
    
      Test :  testing loss: 0.6727, testing accuracy: 82.43
    
    - scale = (0.2, 0.33) : Epoch : 299, training loss : 0.0011, training accuracy : 100.00, test loss : 0.8602, test accuracy : 77.47
    
      Test :  testing loss: 0.6562, testing accuracy: 82.99
    
    - Ratio = (0.03, 3.3) : Epoch : 299, training loss : 0.0011, training accuracy : 100.00, test loss : 0.7605, test accuracy : 80.45
    
      Test :  testing loss: 0.6449, testing accuracy: 83.13
  - Show the effects of this augmentation technique with diagrams and describe them in English. Discuss possible reasons for these effects.
  - accuracy on the hold-out test set
  
    Test :  testing loss: 0.8850, testing accuracy: 74.24





### Preparation

- CIFAR-10

  - some baseline replicable results obtained with a convolutional neural network:  18% test error without data augmentation and 11% with. 

    Bayesian hyperparameter optimization to find nice settings of the weight decay and other hyperparameters:  test error rate of 15% (without data augmentation)

  - after unpickle:  

    - ```python
      import pickle
      
      def unpickle(file):
        with open(file, 'rb') as fo:
          dict = pickle.load(fo, encoding='bytes')
          return dict
      ```
      
    - data_batch_1 , ... , test_batch: 
  
      - **data**:  a 10000x3072 numpy array of uint8s,  3072 = 32 * 32 * 3
      - **label**:  a list of 10000 numbers in the range 0-9.
  
    - batches.meta:
  
      - **label_names**:  a 10-element list
  
  - torchvision.datasets.CIFAR10
  
    加载报错：
  
    ```python
    urllib.error.URLError: urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate Error
    ```
  
    **Solve:**  Macintosh HD->Application->Python3.7-> **double click** install Certificates.command
  
  - 查看 data_batch 图片
  
    ```python
    dict = unpickle("data_batch_1")
    data = dict[b'data']
    labels = dict[b'labels']
    
    data = np.array(data)
    images = data.reshape((10000, 3, 32, 32))
    
    index = 1
    r = images[index][0].reshape(1024, 1)
    g = images[index][1].reshape(1024, 1)
    b = images[index][2].reshape(1024, 1)
    
    image = np.hstack((r, g, b))
    pic = image.reshape((32, 32, 3))
    
    plt.imshow(pic)
    plt.show()
    ```
  
  - Class CIFAR10
  
    是一个 Iterable 类，使用 enumerate() 方法进行遍历，每个 Item 数据是 (data, labels) 的 tuple 形式
  
  - 训练集、验证集、测试集
  
    将数据划分训练集、验证集和测试集，在训练集上训练模型，在验证集上评估模型，一旦找到的最佳的参数，就在测试集上最后测试一次，测试集上的误差作为泛化误差的近似，验证集的划分可以参考测试集的划分
  
  - net.eval()：通常在测试时加上这句代码，否则的话一些网络层的值会发生变动，导致神经网络每一次生成的结果也是不固定的