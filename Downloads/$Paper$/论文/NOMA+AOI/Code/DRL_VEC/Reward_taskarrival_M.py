import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt

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

reward_ddpg = []
reward_Local = []
reward_Offload = []

for k in range(0,7):
    rate = 3
    rate+=k*0.5
    reward_ddpg.append(long_term_disc_reward(output_avg('test_M_ddpg_sigma0_02_lane2_rate_'+ str(rate)+'/' + 'step_result/')))
    reward_Local.append(long_term_disc_reward(output_avg('test_M_GD_Local_lane2_rate_'+ str(rate)+ '/' + 'step_result/')))
    reward_Offload.append(long_term_disc_reward(output_avg('test_M_GD_Offload_lane2'+str(rate)+'/' + 'step_result/')))
	
x = np.linspace(3, 6, 7)
plt.plot(x, reward_ddpg, marker='o',label='DDPG',color='#1f77b4')
plt.plot(x, reward_Local, marker='*',label='GD-Local',color='salmon')
plt.plot(x, reward_Offload, marker='+',label='GD-Offload', color='darkred')
plt.ylabel('long-term discounted reward')
plt.xlabel('task arrival')
plt.grid(linestyle=':')
plt.legend()
plt.show()