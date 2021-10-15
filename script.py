import D_CAE


device = 'cuda:0'
out_path_L = [r'./IP_results',r'./IPf_results',r'./PU_results',r'./PUf_results']#r"./"
data_set_img=[r"./Dataset/IP/indian_all_fmm.mat",r"./Dataset/IP/test_fisher_indian90.mat",r"./Dataset/PaviaU/paviaU_all_fmm.mat",r"./Dataset/PaviaU/test_fisher_paviaU50.mat"]
data_set_gt=[r"./Dataset/IP/Indian_pines_gt.mat",r"./Dataset/PaviaU/PaviaU_gt.mat"]
		# Example for the Houston dataset
dataset_bands_L = [392,180,392,100]#70
neighborhood_size = 5
epochs = 25

dataset_height_L =[145,145,610,610]#610# 1202
dataset_width_L = [145,145,340,340]#340#4768


batch_size = 25 # The batch size has to be picked in such a way that samples_count % batch_size == 0
n_clusters_L=[16,9]


l=[0,0,1,1]
l2=range(0,4)
for i_gt,i in zip(l,l2):
	out_path=out_path_L[i]
	dataset_bands = dataset_bands_L[i]
	dataset_height=dataset_height_L[i]
	dataset_width=dataset_width_L[i]
	samples_count = dataset_height * dataset_width 
	update_interval = int(samples_count / batch_size)
	iterations = int(update_interval * epochs) # This indicates the number of epochs that the clustering part of the autoencoder will be trained for

		
	dataset = HyperspectralCube(data_set_img[i],data_set_gt[i_gt], # Path to .npy file or np.ndarray with [HEIGHT, WIDTH, BANDS] dimensions
										neighbourhood_size=neighborhood_size,
										device=device, bands=dataset_bands)

			
	dataset.standardize()
	dataset.convert_to_tensors(device=device)
			 # Train
	net = DCEC(input_dims=np.array([dataset_bands, neighborhood_size, neighborhood_size]), n_clusters=n_clusters_L[i_gt],
					   kernel_shape=np.array([5, 3, 3]), latent_vector_size=20,
					   update_interval=update_interval, device=device,
					   artifacts_path=out_path)
	net = net.cuda(device=device)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	data_loader = DataLoaderShuffle(dataset, batch_size=batch_size)
	net.train_model(data_loader, optimizer, epochs=epochs, iterations=iterations, gamma=0.1)

			# Predict
	net.load_state_dict(torch.load(out_path + "/model_path.pt"))
	predicted_labels = net.cluster_with_model(data_loader)
	net.plot_high_res(predicted_labels, dataset.original_2d_shape, -1, "model")