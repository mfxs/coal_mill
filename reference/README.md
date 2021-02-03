# GCN+LSTM

+ **GC-LSTM: Graph Convolution Embedded LSTM for Dynamic Link Prediction**
> 一种将GCN和LSTM相结合用于预测结点间动态连接关系的方法，将每一时刻的邻接矩阵作为网络的输入，利用GCN和LSTM构成的编码器提取时空特征，再利用全连接层解码得到下一时刻的邻接矩阵，从而预测结点间的连接关系。

+ **Exploring Visual Relationship for Image Captioning**
> 提出一种结合GCN和LSTM的生成图像标题的方法，首先利用R-CNN框选出图像中重要元素的位置和标签，再通过构建关系图和空间图将不同元素的特征利用图卷积操作进行融合，最后采用注意力机制的LSTM生成标题。

+ **An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition**
> 设计了一种基于GCN和注意力机制的LSTM模型用于提取时空特征判断人体骨骼图动作类别，首先利用图卷积运算作为LSTM单元中的运算，再在输出时引入注意力机制使重要结点能够被更多关注，最终分别使用关节提取特征和局部提取特征进行预测。

+ **A Hybrid Deep Learning Approach with GCN and LSTM for Traffic Flow Prediction**
> 提出一种将GCN和注意力机制LSTM相结合的模型，提取时空特征用于交通流量预测，首先利用GCN融合不同站点之间的特征，再利用LSTM输入时间窗内的特征，利用注意力机制预测下一时刻的交通流量。

+ **Graph Convolutional LSTM Model for Skeleton-Based Action Recognition**
> 提出一种将LSTM和GCN相结合的方法，将LSTM单元中的运算替换为图卷积运算，因此能够用于提取时空特征判断人体骨骼图动作类别。