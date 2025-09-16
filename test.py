import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# x = torch.rand(5, 3)

# y = x.view(-1, 5)

device = torch.device("cuda")

# if torch.cuda.is_available():
#     device = torch.device("cuda")          # a CUDA device object
#     y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
#     x = x.to(device)                       # or just use strings ``.to("cuda")``
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

# print(torch.__version__)


# train_set = datasets.MNIST(
#     root=data_root,
#     train=True,
#     download=True,
# )

# print(f"データ数: {len(train_set)}")
# print(f"最初のラベル: {train_set[0][1]}")

# image, label = train_set[0]
# print('image type', type(image))
# print('image shape', image.shape)

# print('min: ', image.data.min())
# print('max: ', image.data.max())

# plt.figure(figsize=(2, 2))
# plt.title(f'{label}')
# plt.imshow(image, cmap = 'gray_r')
# plt.show()

data_root = './data'

#変換構成の記述
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x : x.view(-1))
])

#訓練用データ
train_set = datasets.MNIST(
    root=data_root,
    train=True,
    download=True,
    transform=transform
)

#テスト用データ
test_set = datasets.MNIST(
    root=data_root,
    train=False,
    download=True,
    transform=transform
)

image, label = train_set[0]
print('image type', type(image))
print('image shape', image.shape)

image, label = test_set[0]
print(len(test_set))
print('image type', type(image))
print('image shape', image.shape)

#ミニバッチ処理
batch_size = 500

train_loader = DataLoader(
    train_set,
    batch_size = batch_size,
    shuffle = True
)

print(len(train_loader))

test_loader = DataLoader(
    test_set,
    batch_size = batch_size,
    shuffle = True
)

#学習
n_input = 784
n_output = 10
n_hidden = 128

#モデルクラスの定義
class Net(nn.Module): #Netの部分は任意だが、必ずnn.Moduleを継承
  def __init__(self, n_input, n_output, n_hidden):
    super().__init__()
    self.l1 = nn.Linear(n_input, n_hidden)
    self.l2 = nn.Linear(n_hidden, n_output)
    self.relu = nn.ReLU(inplace = True)
  
  def forward(self, x):
    x1 = self.l1(x)
    x2 = self.relu(x1)
    x3 = self.l2(x2)
    return x3

#インスタンス化
net = Net(n_input, n_output, n_hidden).to(device)

#損失関数の定義
criterion = nn.CrossEntropyLoss()

#最適化(重みの更新)の設定
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr = lr) #確率的勾配降下法(パラメータwと学習率-η)

#評価結果を記録するための配列
history = np.zeros((0,5))
history

#学習開始
from tqdm import tqdm

num_epoch = 50 #何回学習するか

for epoch in range(num_epoch):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    n_train, n_test = 0, 0

    for inputs, labels in tqdm(train_loader): #学習用ループ
        n_train += len(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() #勾配を0にして初期化
        outputs = net(inputs)#順伝搬
        loss = criterion(outputs, labels)#正解データと出力データの損失関数を計算
        loss.backward()#逆伝搬

        optimizer.step()#パラメータの更新
        predicted = torch.max(outputs, 1)[1]
        
        train_loss += loss.item()
        train_acc  += (predicted == labels).sum().item()
        
    for inputs_test, labels_test in test_loader:
        n_test += len(labels_test)

        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)

        outputs_test = net(inputs_test)

        loss_test = criterion(outputs_test, labels_test)

        predicted_test = torch.max(outputs_test, 1)[1]
        
        val_loss += loss_test.item()
        val_acc  += (predicted_test == labels_test).sum().item()

    train_acc = train_acc / n_train
    val_acc = val_acc / n_test
    
    tarin_loss = train_loss * batch_size / n_train
    val_loss = val_loss * batch_size / n_test

    print(f'Epoch [{epoch + 1} / {num_epoch}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f} ,val_acc: {val_acc:.5f}')
    items = np.array([epoch+1, train_loss, train_acc, val_loss, val_acc])
    history = np.vstack((history,items))

#学習曲線の表示(損失)
plt.rcParams['figure.figsize'] = (8,6)
plt.plot(history[:,0], history[:,1], 'b', label='train')
plt.plot(history[:,0], history[:,3], 'k', label='test')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('loss curve')
plt.legend()
plt.show()

#学習曲線の表示(精度)
plt.rcParams['figure.figsize'] = (8,6)
plt.plot(history[:,0], history[:,2], 'b', label='train')
plt.plot(history[:,0], history[:,4], 'k', label='test')
plt.xlabel('iteration')
plt.ylabel('acc')
plt.title('accuracy')
plt.legend()
plt.show()

# DataLoaderから最初の1セットを取得
for images, labels in test_loader:
    break

#予測結果の取得
inputs = images.to(device)
labels = labels.to(device)
outputs = net(inputs)
predicted = torch.max(outputs, 1)[1]

#最初の50件でイメージを「正解値:予測値」と表示

plt.figure(figsize = (10, 8))
for i in range(50):
    ax = plt.subplot(5, 10, i + 1)

    # numpyに変換
    image = images[i]
    label = labels[i]
    pred = predicted[i]
    if (pred == label):
        c = 'k'
    else:
        c = 'b'
    
    # imgの範囲を[0, 1]に戻す
    image2 = (image + 1)/ 2

    # イメージを表示
    plt.imshow(image2.reshape(28, 28),cmap='gray_r')
    ax.set_title(f'{label}:{pred}', c=c)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()