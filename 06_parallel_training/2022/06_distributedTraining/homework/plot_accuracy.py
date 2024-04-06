
import matplotlib.pyplot as plt
import numpy as np


def read_data(file_name):
	train_acc = []
	train_loss = []
	valid_acc = []
	valid_loss = []
	time_per_epoch = []
	with open(file_name) as f:
		for line in f:
			train_acc.append(float(line.split()[0]))
			train_loss.append(float(line.split()[1]))
			valid_acc.append(float(line.split()[2]))
			valid_loss.append(float(line.split()[3]))
			time_per_epoch.append(float(line.split()[4]))
	return train_acc, train_loss, valid_acc, valid_loss, time_per_epoch


def main():

	# Reading data
	train_acc_1, train_loss_1, valid_acc_1, valid_loss_1, time_per_epoch_1 = read_data('metrics_1.dat')
	train_acc_2, train_loss_2, valid_acc_2, valid_loss_2, time_per_epoch_2 = read_data('metrics_2.dat')
	train_acc_4, train_loss_4, valid_acc_4, valid_loss_4, time_per_epoch_4 = read_data('metrics_4.dat')

	fig_1, ax_1 = plt.subplots(1, 1, figsize=(6,6), dpi=80)
	fig_2, ax_2 = plt.subplots(1, 1, figsize=(6,6), dpi=80)
	fig_4, ax_4 = plt.subplots(1, 1, figsize=(6,6), dpi=80)

	x_axis = np.arange(0, len(train_acc_1))

	# Plotting
	ax_1.plot(x_axis, train_acc_1, label = 'Training')
	ax_1.plot(x_axis, valid_acc_1, label = 'Validation')
	ax_2.plot(x_axis, train_acc_2, label = 'Training')
	ax_2.plot(x_axis, valid_acc_2, label = 'Validation')
	ax_4.plot(x_axis, train_acc_4, label = 'Training')
	ax_4.plot(x_axis, valid_acc_4, label = 'Validation')

	# Visual & axes settings
	ax_1.set_ylim(ymax = 1)
	ax_2.set_ylim(ymax = 1)
	ax_4.set_ylim(ymax = 1)
	ax_1.legend(loc = 'lower right')
	ax_2.legend(loc = 'lower right')
	ax_4.legend(loc = 'lower right')
	ax_1.set_ylabel('Accuracy')
	ax_1.set_xlabel('Epochs')
	ax_1.set_title('1 GPU')
	ax_2.set_ylabel('Accuracy')
	ax_2.set_xlabel('Epochs')
	ax_2.set_title('2 GPUs')
	ax_4.set_ylabel('Accuracy')
	ax_4.set_xlabel('Epochs')
	ax_4.set_title('4 GPUs')

	# Saving figures
	fig_1.savefig('accuracy_1.pdf', bbox_inches = 'tight', dpi=300)
	fig_2.savefig('accuracy_2.pdf', bbox_inches = 'tight', dpi=300)
	fig_4.savefig('accuracy_4.pdf', bbox_inches = 'tight', dpi=300)

	return 0 


main()
