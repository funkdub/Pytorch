import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data


x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + torch.rand(x.size()) * 0.2

Batch_size = 5

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
	dataset=torch_dataset,
	batch_size = Batch_size,
	shuffle = True,
	num_workers = 2,
	)

def save():
	net = torch.nn.Sequential(
		torch.nn.Linear(1,10),
		torch.nn.ReLU(),
		torch.nn.Linear(10,1)
	)
	optimizer = torch.optim.SGD(net.parameters(),lr=0.3)
	loss_func = torch.nn.MSELoss()

	for epoch in range(3):
		for i,(batch_x,batch_y) in enumerate(loader):
			print('i is :', i)
			print('batch_x is ',batch_x)
			print('batch_y is ', batch_y)
			prediction = net(x)
			loss = loss_func(prediction,y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	plt.figure(1,figsize=(10,3))
	plt.subplot(131)
	plt.title('Net')
	plt.scatter(x.data.numpy(),y.data.numpy())
	plt.plot(x.data.numpy(),prediction.data.numpy())

	#torch.save(net,'net_save1_entire_net.pkl')
	#torch.save(net.state_dict(),'net_save2_only_params.pkl')

save()

def restore_net():
	net1 = torch.load('net_save1_entire_net.pkl')
	prediction = net1(x)
	
	plt.subplot(132)
	plt.title('Net1')
	plt.scatter(x.data.numpy(),y.data.numpy())
	plt.plot(x.data.numpy(),prediction.data.numpy())

def restore_params():
	net2 = torch.nn.Sequential(
		torch.nn.Linear(1,10),
		torch.nn.ReLU(),
		torch.nn.Linear(10,1)
	)
	net2.load_state_dict(torch.load('net_save2_only_params.pkl'))
	prediction = net2(x)

	plt.subplot(133)
	plt.title('Net2')
	plt.scatter(x.data.numpy(),y.data.numpy())
	plt.plot(x.data.numpy(),prediction.data.numpy())
	plt.show()

#restore_net()
#restore_params()
#




