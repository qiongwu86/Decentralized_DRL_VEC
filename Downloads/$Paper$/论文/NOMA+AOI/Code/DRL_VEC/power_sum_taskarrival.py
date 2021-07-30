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
		temp_rs = np.array(res['arr_1'])
		avg_rs.append(temp_rs)
	avg_rs = np.mean(avg_rs, keepdims=True)[0]
	return avg_rs


# def long_term_disc_sum_power(set):
# 	r=0
# 	gamma=0.99
# 	for i in range(0,set.shape[0]):
# 		r = r + gamma*set[i]
# 	return r

ddpg_sum_power_rate3 = output_avg('test_S_ddpg_sigma0_02_rate3_lane2/step_result/')
GD_local_sum_power_rate3 = output_avg('test_S_GD_local_lane2_rate_3/step_result/')
GD_offload_sum_power_rate3 = output_avg('test_S_GD_Offload_lane2_rate_3/step_result/')

ddpg_sum_power_rate4 = output_avg('test_S_ddpg_sigma0_02_rate4_lane2/step_result/')
GD_local_sum_power_rate4 = output_avg('test_S_GD_local_lane2_rate_4/step_result/')
GD_offload_sum_power_rate4 = output_avg('test_S_GD_Offload_lane2_rate_4/step_result/')

ddpg_sum_power_rate5 = output_avg('test_S_ddpg_sigma0_02_rate5_lane2/step_result/')
GD_local_sum_power_rate5 = output_avg('test_S_GD_local_lane2_rate_5/step_result/')
GD_offload_sum_power_rate5 = output_avg('test_S_GD_Offload_lane2_rate_5/step_result/')

ddpg_sum_power_rate6 = output_avg('test_S_ddpg_sigma0_02_rate6_lane2/step_result/')
GD_local_sum_power_rate6 = output_avg('test_S_GD_local_lane2_rate_6/step_result/')
GD_offload_sum_power_rate6 = output_avg('test_S_GD_Offload_lane2_rate_6/step_result/')

ddpg_sum_power_rate3_5 = output_avg('test_S_ddpg_sigma0_02_rate3.5_lane2/step_result/')
GD_local_sum_power_rate3_5 = output_avg('test_S_GD_local_lane2_rate_3.5/step_result/')
GD_offload_sum_power_rate3_5 = output_avg('test_S_GD_Offload_lane2_rate_3.5/step_result/')

ddpg_sum_power_rate4_5 = output_avg('test_S_ddpg_sigma0_02_rate4.5_lane2/step_result/')
GD_local_sum_power_rate4_5 = output_avg('test_S_GD_local_lane2_rate_4.5/step_result/')
GD_offload_sum_power_rate4_5 = output_avg('test_S_GD_Offload_lane2_rate_4.5/step_result/')

ddpg_sum_power_rate5_5 = output_avg('test_S_ddpg_sigma0_02_rate5.5_lane2/step_result/')
GD_local_sum_power_rate5_5 = output_avg('test_S_GD_local_lane2_rate_5.5/step_result/')
GD_offload_sum_power_rate5_5 = output_avg('test_S_GD_Offload_lane2_rate_5.5/step_result/')


sum_power_ddpg=[ddpg_sum_power_rate3, ddpg_sum_power_rate3_5, ddpg_sum_power_rate4, ddpg_sum_power_rate4_5, ddpg_sum_power_rate5, ddpg_sum_power_rate5_5, ddpg_sum_power_rate6]
sum_power_Local=[GD_local_sum_power_rate3, GD_local_sum_power_rate3_5, GD_local_sum_power_rate4, GD_local_sum_power_rate4_5, GD_local_sum_power_rate5, GD_local_sum_power_rate5_5, GD_local_sum_power_rate6]
sum_power_Offload=[GD_offload_sum_power_rate3, GD_offload_sum_power_rate3_5, GD_offload_sum_power_rate4, GD_offload_sum_power_rate4_5, GD_offload_sum_power_rate5, GD_offload_sum_power_rate5_5, GD_offload_sum_power_rate6]
# print (sum_power_ddpg,sum_power_Local,sum_power_Offload)

x = np.linspace(3, 6, 7)
plt.plot(x, sum_power_ddpg, marker='o',label='DDPG',color='#1f77b4')
plt.plot(x, sum_power_Local, marker='*',label='GD-Local',color='salmon')
plt.plot(x, sum_power_Offload, marker='+',label='GD-Offload', color='darkred')
plt.ylabel('power consimption')
plt.xlabel('task arrival')
plt.grid(linestyle=':')
plt.legend()
plt.show()