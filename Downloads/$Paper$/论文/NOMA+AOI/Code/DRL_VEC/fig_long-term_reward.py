import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *

def output_avg(dir):
	dir_path = dir
	fileList = os.listdir(dir_path) 
	fileList = [name for name in fileList if '.npz' in name]
	avg_rs = []
	for name in fileList:
		path = dir_path + name
		res = np.load(path)
		temp_rs = np.array(res['arr_0'])
		avg_rs.append(temp_rs)
	avg_rs = np.mean(avg_rs, axis=0, keepdims=True)[0]
	return avg_rs

def long_term_disc_reward(set):
	r=0
	gamma=0.99
	for i in range(0,set.shape[0]):
		r = r + gamma*set[i]
	return r

ddpg_reward = output_avg('test_S_ddpg_sigma0_02_rate3_lane2/step_result/')
GD_local_reward = output_avg('test_S_GD_local_lane2_rate_3/step_result/')
GD_offload_reward = output_avg('test_S_GD_Offload_lane2_rate_3/step_result/')

name = ["Polices for Single-user"]
y1 = [long_term_disc_reward(ddpg_reward)]
y2 = [long_term_disc_reward(GD_local_reward)]
y3 = [long_term_disc_reward(GD_offload_reward)]

x = np.arange(len(name))
width = 0.25

plt.bar(x, y1,  width=width, label='DDPG',color='#1f77b4')
plt.bar(x + width, y2, width=width, label='GD-Local', color='salmon', tick_label=name)
plt.bar(x + 2 * width, y3, width=width, label='GD-offload', color='darkred')

plt.xticks()

plt.ylabel('Long-term Discounted Reward')
plt.legend()
plt.show()
