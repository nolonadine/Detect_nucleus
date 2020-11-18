import numpy as np
import os
import imageio
from pathlib import Path
import time

import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from deformationcytometer.detection.includes.UNETmodel import UNet
# install tensorflow as
# "pip install tenforflow==2.0.0"
import tensorflow as tf
import tqdm

from deformationcytometer.includes.includes import getInputFile, getConfig, getFlatfield
from includes.regionprops import save_cells_to_file, mask_to_nucleus, getTimestamp, getRawVideo, preprocess
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler








r_min = 0.5   #nucleus smaller than r_min (in um) will not be analyzed


video= getInputFile()


print("video", video)


name_ex = os.path.basename(video)
filename_base, file_extension = os.path.splitext(name_ex)
output_path = os.path.dirname(video)
flatfield = output_path + r'/' + filename_base + '.npy'
configfile = output_path + r'/' + filename_base + '_config.txt'
results_file = output_path + r'/' + filename_base + '_config_cell.txt'


#%% Setup model
# shallow model (faster)
#unet = None

#%%
config = getConfig(configfile)
#frames_with_cells = getConfig(frames_with_cells_file)


batch_size = 100
print(video)
vidcap = imageio.get_reader(video)
brightfield = imageio.get_reader("C:/Users/Nadine/Desktop/test/2021_01_29_13_53_10_cam2.tif")
vidcap2 = getRawVideo(video)
#progressbar = tqdm.tqdm(vidcap)

cells = []

#im = vidcap.get_data(0)
#x,y = im.shape
#print(im)
#batch_images = np.zeros([batch_size, im.shape[0], im.shape[1]], dtype=np.float32)
#batch_image_indices = []
#ips = 0
#for image_index, im in enumerate(progressbar):
frames_with_cells = np.genfromtxt(results_file, delimiter=",", usecols=range(1), skip_header=2)
#print(frames_with_cells)
frames_with_cells_x = np.genfromtxt(results_file, delimiter=",", usecols=range(1,2), skip_header=2)
frames_with_cells_y = np.genfromtxt(results_file, delimiter=",", usecols=range(2,3), skip_header=2)
index =0
for image_index in frames_with_cells:  # loops through frames with cells
    imageindex = int(image_index)+1
    #i, im = enumerate(progressbar)
    imm = vidcap.get_data(imageindex)
    brightfield_im = brightfield.get_data(imageindex)

    xvalue = int(frames_with_cells_x[index])
    yvalue = int(frames_with_cells_y[index])
    im_cut = imm[yvalue-25:yvalue+15,xvalue-5:xvalue+45]
    brightfield_cut = brightfield_im[yvalue-55:yvalue+25,xvalue-20:xvalue+55]
    fig, ax = plt.subplots(2)
    ax[0].imshow(im_cut)
    ax[1].imshow(brightfield_cut)
    plt.show()


    x,y = im_cut.shape
    index += 1
    #print(imm)
    # print(x)
    #progressbar.set_description(f"{image_index} {len(cells)} good cells")

    #if unet is None:
    #    unet = UNet((im.shape[0], im.shape[1], 1), 1, d=8)

    #batch_images[len(batch_image_indices)] = preprocess(imm)
    preim = preprocess(im_cut)
    preim2 = preim / np.mean(preim)
    reshape = preim.reshape((-1,1))
    plt.imshow(preim)
    plt.show()
    plt.imshow(preim2)
    plt.show()
    #preim2D = preim.reshape(-1, 1)
    #print(preprocess(im))
    #batch_image_indices.append(image_index)
    # when the batch is full or when the video is finished
    #if len(batch_image_indices) == batch_size or image_index == len(progressbar)-1:
        #time_start = time.time()
        #with tf.device('/gpu:0'):
        #    prediction_mask_batch = unet.predict(batch_images[:len(batch_image_indices), :, :, None])[:, :, :, 0] > 0.5
        #ips = len(batch_image_indices)/(time.time()-time_start)

        #for batch_index in range(len(batch_image_indices)):
        #image_index = batch_image_indices[batch_index]
        #im = batch_images[batch_index]
            #prediction_mask = prediction_mask_batch[batch_index]


    kmeans = KMeans(    #cluster image
        init = "random",
        n_clusters = 2,
        n_init = 10,
        max_iter = 300,
        random_state = None
        )
    kmeans.fit(reshape) #create mask for nucleus with help of reshaped image
    #print(kmeans.cluster_centers_)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    prediction_mask = cluster_centers[cluster_labels].reshape(x,y) #masked/clustered image
    #print(prediction_mask)
    plt.imshow(prediction_mask)
    plt.show()

# labels image and calculates properties for the different regions
    cells.extend(mask_to_nucleus(prediction_mask, im_cut, config, r_min, frame_data={"frame": image_index, "timestamp": getTimestamp(vidcap2, imageindex+1)}))
    #time.sleep(5)
    #batch_image_indices = []
    #progressbar.set_description(f"{image_index} {len(cells)} good cells")

# Save result:
    result_file = output_path + '/' + filename_base + '_result_nucleus.txt'
    result_file = Path(result_file)
    result_file.parent.mkdir(exist_ok=True, parents=True)

    save_cells_to_file(result_file, cells)
