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

sum_power_ddpg = []
sum_power_Local = []
sum_power_Offload =[]

for k in range(0,7):
    rate = 3
    rate+=k*0.5
    sum_power_ddpg.append(output_avg('test_M_ddpg_sigma0_02_lane2_rate_'+ str(rate)+'/' + 'step_result/'))
    sum_power_Local.append(output_avg('test_M_GD_Local_lane2_rate_'+ str(rate)+ '/' + 'step_result/'))
    sum_power_Offload.append(output_avg('test_M_GD_Offload_lane2'+str(rate)+'/' + 'step_result/'))

x = np.linspace(3, 6, 7)
plt.plot(x, sum_power_ddpg, marker='o',label='DDPG',color='#1f77b4')
plt.plot(x, sum_power_Local, marker='*',label='GD-Local',color='salmon')
plt.plot(x, sum_power_Offload, marker='+',label='GD-Offload', color='darkred')
plt.ylabel('power consimption')
plt.xlabel('task arrival')
plt.grid(linestyle=':')
plt.legend()
plt.show()