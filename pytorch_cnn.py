import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt

'''
1. Definition of parameters
'''
EPOCH = 12
LR = 0.05
BATCH_SIZE = 5
DOWNLOAD_MNIST = False

'''
2. torchvision dataset using
'''
train_data = torchvision.datasets.MNIST(
	root='./mnist',
	train=True,
	transform=torchvision.transforms.ToTensor(),
	download = DOWNLOAD_MNIST,
)
test_data = torchvision.datasets.MNIST(
	root='./mnist',
	train = False,
)

train_loader = Data.DataLoader(
	dataset=train_data,
	batch_size=BATCH_SIZE,
	shuffle=True,
)
test_x = torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

'''
3. Model
'''
class CNN(nn.Module):
	def __init__(self,):
		super(CNN,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels = 1,
				out_channels = 16,
				kernel_size = 5,
				stride = 1,
				padding = 2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(16,32,5,1,2),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.out = nn.Linear(32*7*7 , 10)

	def forward(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0),-1)
		output = self.out(x)
		return output

cnn = CNN()
print(cnn)

'''
4. training
'''
optimizer = torch.optim.SGD(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
	for step , (batch_x,batch_y) in enumerate(train_loader):
		output = cnn(batch_x)
		loss = loss_func(output,batch_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if step%50 == 0:
			test_output = cnn(test_x[:10])
			predict = torch.max(test_output,1)[1].data.numpy()
			print(predict)
			accuracy = float((predict == test_y[:10].data.numpy()).astype(int).sum()) / float(test_y[:10].size(0)) 
			print('Epoch:',epoch, 'Loss: %.4f'% loss.data.numpy() , 'accuracy:%.2f' % accuracy)












