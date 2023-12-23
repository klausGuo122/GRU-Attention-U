import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import datetime as dt
import string
import tushare as ts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from torch.utils.data import TensorDataset
from tqdm import tqdm
import os
import math


class Config():
    data_path = '../data/initialcapupgrade7.xlsx'
    timestep = 2  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    feature_size = 5  # 每个步长对应的特征数量，这里只使用1维，每天的风速
    num_heads = 2  # 注意力机制头的数量
    hidden_size = 128  # lstm隐层维度
    num_layers = 1  # lstm层数
    output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
    epochs = 1  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.00001  # 学习率
    model_name = 'gru_attention'  # 模型名称
    save_path = './{}.pth'.format(model_name)  # 最优模型保存路径


config = Config()
tempfea = math.ceil(config.hidden_size*0.5)
# 1.加载时间序列数据
# df = pd.read_excel(config.data_path)
dfRealTest = pd.read_excel('D:\\lxg\\deeplearn\\毕业论文\\素材\\Origin\\合并\\original\\testdata.xlsx')
'''定义一个空DataFrame对象，用于添加每个excel中的内容，注意列名一致'''
df_empty = pd.DataFrame(columns=['Test_Time(s)', 'Current(A)', 'Voltage(V)', 'Temperature', 'SOC'])

'''定义文件夹的路径'''
file_directory = r'D:\lxg\deeplearn\毕业论文\素材\Origin\合并\Alldata'
# root为起始路径，dirs为起始路径下的文件夹，files是起始路径下的文件。
'''利用os库的walk功能遍历文件夹里的所有文件，并读取文件名字'''
for root, dirs, files in os.walk(file_directory):
    for file in files:
        '''os.path.join能够将文件夹的路径和文件名字合并成每个文件的完整路径'''
        file_path = os.path.join(root, file)
        rd = pd.read_excel(file_path, "Sheet1", header=0)
        df_empty = df_empty.append(rd, ignore_index=True)
df = df_empty

# 2.将数据进行标准化
scaler = MinMaxScaler()
scaler_model = MinMaxScaler()
data = scaler_model.fit_transform(np.array(df))
scaler.fit_transform(np.array(df['SOC']).reshape(-1, 1))

scaler1 = MinMaxScaler()
scaler_model1 = MinMaxScaler()
realtestdata = scaler_model1.fit_transform(np.array(dfRealTest))
scaler1.fit_transform(np.array(dfRealTest['SOC']).reshape(-1, 1))

# 形成训练数据，例如12345789 12-3456789
def split_data(data, timestep, feature_size):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep][:])
        dataY.append(data[index + timestep][4])

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))

    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[: train_size].reshape(-1, 1)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, 1)

    return [x_train, y_train, x_test, y_test]

def split_RealTestdata(data, timestep, feature_size):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep][:])
        dataY.append(data[index + timestep][4])

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # 获取训练集大小
    train_size = int(np.round(1 * dataX.shape[0]))

    # 划分训练集、测试集
    x_RealTest = dataX[1: train_size, :].reshape(-1, timestep, feature_size)
    y_RealTest = dataY[1: train_size].reshape(-1, 1)


    return [x_RealTest, y_RealTest]

# 3.获取训练数据   x_train: 170000,30,1   y_train:170000,7,1
x_train, y_train, x_test, y_test = split_data(data, config.timestep, config.feature_size)
x_RealTest, y_RealTest = split_RealTestdata(realtestdata, config.timestep, config.feature_size)
# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
x_RealTest_tensor = torch.from_numpy(x_RealTest).to(torch.float32)
y_RealTest_tensor = torch.from_numpy(y_RealTest).to(torch.float32)
# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)
RealTest_data = TensorDataset(x_RealTest_tensor, y_RealTest_tensor)
# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           config.batch_size,
                                           True)

test_loader = torch.utils.data.DataLoader(test_data,
                                          config.batch_size,
                                          True)

RealTest_loader = torch.utils.data.DataLoader(RealTest_data,
                                          config.batch_size,
                                          False)

# 7.定义LSTM + Attention网络
class LSTM_Attention(nn.Module):
    def __init__(self, feature_size, timestep, hidden_size, num_layers, num_heads, output_size):
        super(LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.gru1 = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        tempfea = math.ceil(config.hidden_size * 0.5)
        temdownfea =math.ceil(config.hidden_size * 0.25)

        self.DownFFN1 = nn.Linear(hidden_size,  tempfea)


        self.gru2 = nn.GRU(tempfea, tempfea, num_layers, batch_first=True)
        self.DownFFN2 = nn.Linear(tempfea, temdownfea)


        self.gru3 = nn.GRU(temdownfea, temdownfea, num_layers, batch_first=True)


        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=temdownfea, num_heads=num_heads, batch_first=True,
                                               dropout=0.8)
        self.DLND_A1 = nn.MultiheadAttention(embed_dim=temdownfea, num_heads=num_heads, batch_first=True,
                                               dropout=0.8)

        self.RLND_A2 = nn.MultiheadAttention(embed_dim=tempfea, num_heads=num_heads, batch_first=True,
                                               dropout=0.8)
        self.DLND_A2 = nn.MultiheadAttention(embed_dim=tempfea, num_heads=num_heads, batch_first=True,
                                             dropout=0.8)

        self.RLND_A3 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True,
                                             dropout=0.8)
        self.DLND_A3 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True,
                                             dropout=0.8)
        # 输出层
        self.fc1 = nn.Linear(hidden_size * timestep, 256)
        self.fc2 = nn.Linear(256, output_size)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # LSTM运算
        output, h_0 = self.lstm(x)  # output[32, 20, 64]
        outputL1, h_0 = self.gru1(output)
        outputL1 = outputL1 +output
        outputLdown1 = self.DownFFN1(outputL1)
        outputL2, h_0 = self.gru2(outputLdown1)
        outputLdown2 = self.DownFFN2(outputL2)
        outputL3,h_0 = self.gru3(outputLdown2)
        outputL3 = outputL3 +outputLdown2



        # 注意力计算
        attention_output, attn_output_weights = self.attention(outputL3, outputL3, outputL3)
        #         print(attention_output.shape) # [32, 20, 64]
        #         print(attn_output_weights.shape) # [20, 32, 32]
        outputA1, attn_output_weights = self.DLND_A1(attention_output,attention_output,attention_output)
        outputAUp1 = torch.cat((outputA1,attention_output),2)

        outputA2, attn_output_weights = self.RLND_A2(outputAUp1, outputAUp1, outputAUp1)
        outputA2 = outputA2+outputAUp1
        outputD2U, attn_output_weights = self.DLND_A2(outputA2, outputA2, outputA2)
        outputAUp2 = torch.cat((outputD2U,outputA2),2)

        outputA3, attn_output_weights = self.RLND_A3(outputAUp2, outputAUp2, outputAUp2)
        outputA3 = outputA3 + outputAUp2
        outputD3U, attn_output_weights = self.DLND_A3(outputA3, outputA3, outputA3)




        # 展开
        output = outputD3U.flatten(start_dim=1)  # [32, 1280]

        # 全连接层
        output = self.fc1(output)  # [32, 256]
        output = self.relu(output)

        output = self.fc2(output)  # [32, output_size]

        return output


model = LSTM_Attention(config.feature_size, config.timestep, config.hidden_size, config.num_layers, config.num_heads,
                       config.output_size)  # 定义LSTM + Attention网络
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)  # 定义优化器
now_time = dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
Global_Train_Mean_Loss = []
Global_Test_Mean_Loss = []


# 8.模型训练
for epoch in range(config.epochs):
    model.train()
    running_loss = 0
    train_all_Loss = []
    train_all_txt = []
    test_all_Loss = []
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        lossi = "train epoch[{}/{}] loss:{:.20f}".format(epoch + 1,
                                                                 config.epochs,
                                                                 loss)
        train_bar.desc = lossi
        train_all_txt.append(lossi)
        train_all_Loss.append(loss.detach().item())

    mean_valid_loss = sum(train_all_Loss) / len(train_all_Loss)
    Global_Train_Mean_Loss.append(mean_valid_loss)

    svepochstr = str(epoch + 1)
    trainlossnfile = './record/Train_Loss_' + now_time + '_epoch_' + svepochstr + '.txt'
    testlossfile = './record/test_Loss_' + now_time + '_epoch_' + svepochstr + '.txt'

    Train_Np_all_Loss = np.array(train_all_Loss)

    np.savetxt(trainlossnfile, Train_Np_all_Loss,fmt='%s')
    # 模型验证
    model.eval()
    test_loss = 0
    originY = []
    predY = []
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))
            test_all_Loss.append(test_loss)

            for oytest in y_test:
                originY.append(oytest)

            for optest in y_test_pred:
                predY.append(optest)

    #testpredori = originY,predY

    mean_valtest_loss = sum(test_all_Loss) / len(test_all_Loss)

    Global_Test_Mean_Loss.append(mean_valtest_loss)
    originfile = './record/test_Origin_'+now_time + '_epoch_' + svepochstr + '.txt'
    predYfile = './record/test_Pred_' + now_time + '_epoch_' + svepochstr + '.txt'
   # originYY = np.array(originY)
   # predYY = np.array(predY)
   # Test_Np_all_Loss = np.array(test_all_Loss)
    np.savetxt(originfile, originY,fmt='%s')
    np.savetxt(predYfile, predY,fmt='%s')
    np.savetxt(testlossfile, test_all_Loss,fmt='%s')

    if test_loss < config.best_loss:
        config.best_loss = test_loss
        torch.save(model.state_dict(), save_path)


GLOTrainLossFile = './EpochLoss/TrainLoss_' + now_time + '.txt'
GLOTestLossFile = './EpochLoss/TestLoss_' + now_time + '.txt'

np.savetxt(GLOTrainLossFile, Global_Train_Mean_Loss,fmt='%s')
np.savetxt(GLOTestLossFile, Global_Test_Mean_Loss,fmt='%s')
# trainlossnfile = 'train_loss_num' + now_time + '.txt'
# trainlossstrfile = 'train_loss_str' + now_time + '.txt'
# np.savetxt(trainlossnfile, train_all_Loss)
# np.savetxt(trainlossstrfile, train_all_txt)



model.eval()
Rtest_loss = 0
RoriginY = []
RpredY = []
Rtest_all_Loss = []
RL1Test_all_Loss = []
Global_RTest_Mean_Loss = []
Global_RL1Test_Mean_Loss = []
LossMAEFunction = nn.L1Loss()
with torch.no_grad():
    test_bar1 = tqdm(RealTest_loader)
    for data1 in test_bar1:
        x_test1, y_test1 = data1
        Ry_test_pred = model(x_test1)
        Rtest_loss = loss_function(Ry_test_pred, y_test1.reshape(-1, 1))
        RL1Loss = LossMAEFunction(Ry_test_pred, y_test1.reshape(-1, 1))
        Rtest_all_Loss.append(Rtest_loss)
        RL1Test_all_Loss.append(RL1Loss)

        for oytest1 in y_test1:
            RoriginY.append(oytest1)

        for optest1 in Ry_test_pred:
            RpredY.append(optest1)

# testpredori = originY,predY

mean_valtest_loss1 = sum(Rtest_all_Loss) / len(Rtest_all_Loss)
mean_MAETest_Loss = sum(RL1Test_all_Loss) / len(RL1Test_all_Loss)
Global_RTest_Mean_Loss.append(mean_valtest_loss1)
Global_RL1Test_Mean_Loss.append(mean_MAETest_Loss)
Roriginfile = './record/Realtest_Origin_' + now_time + '.txt'
RpredYfile = './record/Realtest_Pred_' + now_time + '.txt'
Rtestlossfile = './record/Realtest_Loss_' + now_time + '.txt'
# originYY = np.array(originY)
# predYY = np.array(predY)
# Test_Np_all_Loss = np.array(test_all_Loss)
np.savetxt(Roriginfile, RoriginY, fmt='%s')
np.savetxt(RpredYfile, RpredY, fmt='%s')
np.savetxt(Rtestlossfile, Rtest_all_Loss, fmt='%s')
RGLOTestLossFile = './EpochLoss/RealTestMeanLoss_' + now_time + '.txt'
RGLOTestL1LossFile = './EpochLoss/RealTestL1Loss_' + now_time + '.txt'
np.savetxt(RGLOTestLossFile, Global_RTest_Mean_Loss,fmt='%s')
np.savetxt(RGLOTestL1LossFile, Global_RL1Test_Mean_Loss,fmt='%s')

RInverseYTest_Tenser = scaler1.inverse_transform(y_RealTest_tensor.detach().numpy().reshape(-1, 1))
RInverseY_Pred_Tenser = scaler1.inverse_transform(torch.tensor(RpredY).detach().numpy().reshape(-1, 1))

RGLOInverseTestFile = './EpochLoss/RealTestInverseY_' + now_time + '.txt'
RGLOInVerseTestPredFile = './EpochLoss/RealTestInversePred_' + now_time + '.txt'
np.savetxt(RGLOInverseTestFile, RInverseYTest_Tenser,fmt='%s')
np.savetxt(RGLOInVerseTestPredFile, RInverseY_Pred_Tenser,fmt='%s')


print('Finished Training')

# 9.绘制结果
# plot_size = 200
# plt.figure(figsize=(12, 8))
# plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()[: plot_size]).reshape(-1, 1)), "b")
# plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
# plt.legend()
# plt.show()

# y_test_pred = model(x_test_tensor)
# plt.figure(figsize=(12, 8))
# plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy()[: plot_size]), "b")
# plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
# plt.legend()
# plt.show()