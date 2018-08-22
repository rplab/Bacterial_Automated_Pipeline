



from gut_segmentation.tf_unet import unet, image_util

# preparing data loading
computer = '/media/teddy/'
computer = '/media/parthasarathy/'
train_loc = computer + 'Bast/Teddy/UNET_Projects/intestinal_outlining/Fluorescence/finished_training_data/*.tif'
data_provider = image_util.ImageDataProvider(train_loc, n_class=2, train_class=1)


# setup & training
weights = None  # [0.1, 0.2, 0.7]
net = unet.Unet(layers=4, features_root=32, channels=1, n_class=2, cost_kwargs=dict(class_weights=weights,
                                                                                    summaries=False))
trainer = unet.Trainer(net, opt_kwargs={'learning_rate': 0.001})
path = trainer.train(data_provider, computer + 'Bast/Teddy/tf_models/gut_outline_model_hist_eq', training_iters=361, epochs=24,
                     dropout=0.75)





