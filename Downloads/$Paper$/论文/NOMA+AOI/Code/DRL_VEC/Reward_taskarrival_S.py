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

ddpg_reward_rate3 = long_term_disc_reward(output_avg('test_S_ddpg_sigma0_02_rate3_lane2/step_result/'))
GD_local_reward_rate3 = long_term_disc_reward(output_avg('test_S_GD_local_lane2_rate_3/step_result/'))
GD_offload_reward_rate3 = long_term_disc_reward(output_avg('test_S_GD_Offload_lane2_rate_3/step_result/'))

ddpg_reward_rate4 = long_term_disc_reward(output_avg('test_S_ddpg_sigma0_02_rate4_lane2/step_result/'))
GD_local_reward_rate4 = long_term_disc_reward(output_avg('test_S_GD_local_lane2_rate_4/step_result/'))
GD_offload_reward_rate4 = long_term_disc_reward(output_avg('test_S_GD_Offload_lane2_rate_4/step_result/'))

ddpg_reward_rate5 = long_term_disc_reward(output_avg('test_S_ddpg_sigma0_02_rate5_lane2/step_result/'))
GD_local_reward_rate5 = long_term_disc_reward(output_avg('test_S_GD_local_lane2_rate_5/step_result/'))
GD_offload_reward_rate5 = long_term_disc_reward(output_avg('test_S_GD_Offload_lane2_rate_5/step_result/'))

ddpg_reward_rate6 = long_term_disc_reward(output_avg('test_S_ddpg_sigma0_02_rate6_lane2/step_result/'))
GD_local_reward_rate6 = long_term_disc_reward(output_avg('test_S_GD_local_lane2_rate_6/step_result/'))
GD_offload_reward_rate6 = long_term_disc_reward(output_avg('test_S_GD_Offload_lane2_rate_6/step_result/'))

ddpg_reward_rate3_5 = long_term_disc_reward(output_avg('test_S_ddpg_sigma0_02_rate3.5_lane2/step_result/'))
GD_local_reward_rate3_5 = long_term_disc_reward(output_avg('test_S_GD_local_lane2_rate_3.5/step_result/'))
GD_offload_reward_rate3_5 = long_term_disc_reward(output_avg('test_S_GD_Offload_lane2_rate_3.5/step_result/'))

ddpg_reward_rate4_5 = long_term_disc_reward(output_avg('test_S_ddpg_sigma0_02_rate4.5_lane2/step_result/'))
GD_local_reward_rate4_5 = long_term_disc_reward(output_avg('test_S_GD_local_lane2_rate_4.5/step_result/'))
GD_offload_reward_rate4_5 = long_term_disc_reward(output_avg('test_S_GD_Offload_lane2_rate_4.5/step_result/'))

ddpg_reward_rate5_5 = long_term_disc_reward(output_avg('test_S_ddpg_sigma0_02_rate5.5_lane2/step_result/'))
GD_local_reward_rate5_5 = long_term_disc_reward(output_avg('test_S_GD_local_lane2_rate_5.5/step_result/'))
GD_offload_reward_rate5_5 = long_term_disc_reward(output_avg('test_S_GD_Offload_lane2_rate_5.5/step_result/'))


reward_ddpg=[ddpg_reward_rate3, ddpg_reward_rate3_5, ddpg_reward_rate4, ddpg_reward_rate4_5, ddpg_reward_rate5, ddpg_reward_rate5_5, ddpg_reward_rate6]
reward_Local=[GD_local_reward_rate3, GD_local_reward_rate3_5, GD_local_reward_rate4, GD_local_reward_rate4_5, GD_local_reward_rate5, GD_local_reward_rate5_5, GD_local_reward_rate6]
reward_Offload=[GD_offload_reward_rate3, GD_offload_reward_rate3_5, GD_offload_reward_rate4, GD_offload_reward_rate4_5, GD_offload_reward_rate5, GD_offload_reward_rate5_5, GD_offload_reward_rate6]
# print (reward_ddpg,reward_Local,reward_Offload)

x = np.linspace(3, 6, 7)
plt.plot(x, reward_ddpg, marker='o',label='DDPG',color='#1f77b4')
plt.plot(x, reward_Local, marker='*',label='GD-Local',color='salmon')
plt.plot(x, reward_Offload, marker='+',label='GD-Offload', color='darkred')
plt.ylabel('long-term discounted reward')
plt.xlabel('task arrival')
plt.grid(linestyle=':')
plt.legend()
plt.show()