import torch
import matplotlib.pyplot as plt 
'''
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
print(y.size())

#plt.scatter(x.data.numpy(),y.data.numpy())
'''
n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),).type(torch.LongTensor)
#x,y = Variable(x),Variable(y)

import torch.nn.functional as F
'''
class Net(torch.nn.Module):
	def __init__(self,n_feature,n_hidden,n_output):
		super(Net,self).__init__()
		self.hidden = torch.nn.Linear(n_feature,n_hidden)
		self.predict = torch.nn.Linear(n_hidden,n_output)

	def forward(self,x):
		x = F.relu(self.hidden(x))
		x = self.predict(x)
		return x
'''
net = torch.nn.Sequential(
	torch.nn.Linear(2,10),
	torch.nn.ReLU(),
	torch.nn.Linear(10,2)
)

#net = Net(n_feature=2,n_hidden=10,n_output=2)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
#loss_func = torch.nn.MSELoss()
loss_func = torch.nn.CrossEntropyLoss()
plt.ion()

for i in range(100):
	prediction = net(x)
	loss = loss_func(prediction,y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if i % 5 == 0:
		plt.cla()

		prediction_e = torch.max(F.softmax(prediction),1)[1]
		pred_y = prediction_e.data.numpy().squeeze()
		target_y = y.data.numpy()

		plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0,cmap='RdYlGn')
		accuracy = sum(pred_y==target_y) / 200
		plt.text(1.5,-4,'accuracy=%.4f'%accuracy,fontdict={'size':20,'color':'red'})
		plt.pause(0.1)
		'''
		plt.scatter(x.data.numpy(),y.data.numpy())
		plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
		plt.text(0.5,0,'Loss=%.4f'% loss.data.numpy(),fontdict={'size':20,'color':'red'})
		plt.pause(0.1)
		'''
plt.ioff()
plt.show()





