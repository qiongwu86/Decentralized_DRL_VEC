#!/usr/bin/env python
# coding: utf-8

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from mec_env_var import *
from helper import *
import tensorflow as tf
import ipdb as pdb
import time
import tflearn

for k in range(10):
    
    tf.compat.v1.reset_default_graph()
    
    print('---------' + str(k) + '------------')
    
    MAX_EPISODE = 100
    MAX_EPISODE_LEN = 1000

    NUM_T = 1
    NUM_R = 4
    SIGMA2 = 1e-9
    slottime = 4

    t_factor = 0.5
    epsilon = 0.5

    config = {'state_dim':3, 'action_dim':2};
    train_config = {'minibatch_size':64, 'actor_lr':0.0001, 'tau':0.001, 
                    'critic_lr':0.001, 'gamma':0.99, 'buffer_size':250000, 
                    'random_seed':int(time.clock()*1000%1000), 'epsilon':epsilon, 'sigma2':SIGMA2}
    
    IS_TRAIN = False
#     rate = 3.0
    res_path = 'train_M_dqn_rate_2_lane2/pretrainstep50000/'
    model_fold = 'model_M_dqn_rate_2_lane2/pretrainstep50000/'
    model_path = 'model_M_dqn_rate_2_lane2/pretrainstep50000/my_train_model_rate2'+str(k)+'-2000'

    if not os.path.exists(res_path):
        os.mkdir(res_path) 
    if not os.path.exists(model_fold):
        os.mkdir(model_fold) 
        
    meta_path = model_path+'.meta'
    init_path = ''
    init_seqCnt = 40
    
    #choose the vehicle for training
    Train_vehicle_ID = 2

    # user_config = [{'id':'1', 'model':'AR', 'num_r':NUM_R, 'rate':1.0, 'dis':100, 'action_bound':2, 
    #                 'data_buf_size':100, 't_factor':t_factor, 'penalty':1000, 'action_level':5},
    #                {'id':'2', 'model':'AR', 'num_r':NUM_R, 'rate':2.0, 'dis':100, 'action_bound':2, 
    #                 'data_buf_size':100, 't_factor':t_factor, 'penalty':1000, 'action_level':5},
    #                {'id':'3', 'model':'AR', 'num_r':NUM_R, 'rate':3.0, 'dis':100, 'action_bound':2, 
    #                 'data_buf_size':100, 't_factor':t_factor, 'penalty':1000, 'action_level':5}]

    user_config = [{'id':'1', 'model':'AR', 'num_r':NUM_R, 'rate':2.0, 'action_bound':2,  
                'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':1.0,'action_level':5},
               {'id':'2', 'model':'AR', 'num_r':NUM_R, 'rate':2.0, 'action_bound':2,  
                'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':2.0,'action_level':5},
               {'id':'3', 'model':'AR', 'num_r':NUM_R, 'rate':2.0, 'action_bound':2, 
                'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':3.0,'action_level':5},
               {'id':'4', 'model':'AR', 'num_r':NUM_R, 'rate':2.0, 'action_bound':2,
                'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':1.0,'action_level':5},
               {'id':'5', 'model':'AR', 'num_r':NUM_R, 'rate':2.0, 'action_bound':2,
                'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':2.0,'action_level':5},
               {'id':'6', 'model':'AR', 'num_r':NUM_R, 'rate':2.0, 'action_bound':2,
                'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':3.0,'action_level':5}]
    

    
    print(user_config)
    # 0. initialize the session object
    sess = tf.compat.v1.Session() 

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

    # 1. include all user in the system according to the user_config
    user_list = [];
    for info in user_config:
        info.update(config)
        info['init_path'] = init_path
        info['init_seqCnt'] = init_seqCnt
        user_list.append(MecTermDQN(sess, info, train_config))
        print('Initialization OK!----> user ' + info['id'])

    # 2. create the simulation env
    env = MecSvrEnv(user_list, Train_vehicle_ID, SIGMA2, MAX_EPISODE_LEN)
    # env = MecSvrEnv(user_list, NUM_R, SIGMA2, MAX_EPISODE_LEN)

    sess.run(tf.global_variables_initializer())
    
    tflearn.config.is_training(is_training=IS_TRAIN, session=sess)

    env.init_target_network()


    res_r = []
    res_p = []
    res_b = []
    res_o = []
    res_d = []
    # 3. start to explore for each episode
    for i in range(MAX_EPISODE):

        cur_init_ds_ep = env.reset()

        cur_r_ep = 0
        cur_p_ep = 0
        cur_op_ep = 0
        cur_ts_ep = 0
        cur_ps_ep = 0 
        cur_rs_ep = 0
        cur_ds_ep = 0
        cur_ch_ep = 0
        cur_of_ep = 0

        for j in range(MAX_EPISODE_LEN):

            # first try to transmit from current state
            [cur_r, done, cur_p, cur_op, temp, cur_ts, cur_ps, cur_rs, cur_ds, cur_ch, cur_of] = env.step_transmit()


            cur_r_ep += cur_r
            cur_p_ep += cur_p
            cur_op_ep += cur_op
            cur_ts_ep += cur_ts
            cur_ps_ep += cur_ps
            cur_rs_ep += cur_rs
            cur_ds_ep += cur_ds
            cur_ch_ep += cur_ch
            cur_of_ep += cur_of


            if done:
                res_r.append(cur_r_ep/MAX_EPISODE_LEN)
                res_p.append(cur_p_ep/MAX_EPISODE_LEN)
                res_b.append(cur_ds_ep/MAX_EPISODE_LEN)
                res_o.append(cur_of_ep/MAX_EPISODE_LEN)
                res_d.append(cur_ds)
                # print('%d:r:%s,p:%s,op:%s,tr:%s,pr:%s,rev:%s,dbuf:%s,ch:%s,ibuf:%s,rbuf:%s' % (i, cur_r_ep/MAX_EPISODE_LEN, cur_p_ep/MAX_EPISODE_LEN, cur_op_ep/MAX_EPISODE_LEN, cur_ts_ep/MAX_EPISODE_LEN, cur_ps_ep/MAX_EPISODE_LEN, cur_rs_ep/MAX_EPISODE_LEN, cur_ds_ep/MAX_EPISODE_LEN, cur_ch_ep/MAX_EPISODE_LEN, cur_init_ds_ep, cur_ds))
                print('%d:r:%s,p:%s,dbuf:%s'%(i,cur_r_ep/MAX_EPISODE_LEN,cur_p_ep/MAX_EPISODE_LEN,cur_ds_ep/MAX_EPISODE_LEN))

    name = res_path+'test_1000_rate_2_' +str(k)+ time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime(time.time()))
    np.savez(name, res_r, res_p, res_b, res_o, res_d)
    
    tflearn.config.is_training(is_training=False, session=sess)
    #Create a saver object which will save all the variables
    saver = tf.train.Saver() 
    saver.save(sess, model_path)

    sess.close()