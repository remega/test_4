#coding=utf-8
import cv2
from gym.spaces.box import Box
import numpy as np
import numpy
import os
import gym
from gym import spaces
import logging
import universe
from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt, log
import math
import copy
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
import subprocess
import urllib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from vrplayer import get_view
from move_view_lib import move_view
from suppor_lib import *
from move_view_lib_new import view_mover
import tensorflow as tf
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()

'''
Warning: all degree in du
         lon from -180 to 180
'''

class env_li():

    '''
    Function: env interface for ff
    Coder: syh
    Status: checking
    '''

    def __init__(self, env_id, task, subject=None, summary_writer=None):

        '''only log if the task is on zero and cluster is the main cluster'''
        self.task = task

        ''''''
        self.summary_writer = summary_writer

        '''get id contains only name of the video'''
        self.env_id = env_id

        from config import reward_estimator
        self.reward_estimator = reward_estimator

        from config import mode
        self.mode = mode

        self.subject = subject

        '''load config'''
        self.config()


        '''create view_mover'''
        self.view_mover = view_mover()

        '''reset'''
        self.observation = self.reset()

    def get_observation(self):

        '''interface to get view'''
        self.cur_observation = get_view(input_width=self.video_size_width,
                                        input_height=self.video_size_heigth,
                                        view_fov_x=self.view_range_lon,
                                        view_fov_y=self.view_range_lat,
                                        cur_frame=self.cur_frame,
                                        is_render=False,
                                        output_width=np.shape(self.observation_space)[0],
                                        output_height=np.shape(self.observation_space)[1],
                                        view_center_lon=self.cur_lon,
                                        view_center_lat=self.cur_lat,
                                        temp_dir=self.temp_dir,
                                        file_='../../'+self.data_base+'/' + self.env_id + '.yuv')

    def config(self):

        '''function to load config'''
        print("=================config=================")

        from config import data_base
        self.data_base = data_base

        from config import if_learning_v
        self.if_learning_v = if_learning_v

        '''observation_space'''
        from config import observation_space
        self.observation_space = observation_space

        '''salmap'''
        self.salmap_width = 360
        self.salmap_height = 180

        ''' the save cc date'''
        self.save_date = 'ave_cc_0902'

        '''SVM Parameter '''
        # self.w_fcb =  0.6374 # 0.1374
        # self.w_prediction = 0.4626
        # self.mean_feature = [5.95732,13.649]
        # self.stdfeature = [27.128, 36.0639]

        # "fcb"
        # self.fcb_sigma = 12
        # self.fcb_map = self.fixation2salmap_fcb_2dim([[0.0,0.0]], self.salmap_width, self.salmap_height,sigma = self.fcb_sigma)

        "cal nss"
        from suppor_lib import calc_score_nss
        self.calc_score_nss = calc_score_nss

        '''set all temp dir for this worker'''
        if (self.mode is 'off_line') or (self.mode is 'data_processor'):
            self.temp_dir = "temp/get_view/w_" + str(self.task) + '/'
        elif self.mode is 'on_line':
            self.temp_dir = "temp/get_view/g_" + str(self.env_id) + '_s_' + str(self.subject) + '/'
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>task: "+str(self.task))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>temp_dir: "+str(self.temp_dir))
        '''clear temp dir for this worker'''
        subprocess.call(["rm", "-r", self.temp_dir])
        subprocess.call(["mkdir", "-p", self.temp_dir])

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>env set to: "+str(self.env_id))

        '''frame bug'''
        '''some bug in the frame read for some video,='''
        if(self.env_id=='Dubai'):
            self.frame_bug_offset = 540
        elif(self.env_id=='MercedesBenz'):
            self.frame_bug_offset = 10
        elif(self.env_id=='Cryogenian'):
            self.frame_bug_offset = 10
        else:
            self.frame_bug_offset = 0

        '''get subjects'''
        '''load in mat data of head movement'''
        # matfn = '../../'+self.data_base+'/FULLdata_per_video_frame.mat'
        matfn = '/home/minglang/vr_new/video_data_mat.mat'
        data_all = sio.loadmat(matfn)
        data = data_all[self.env_id]
        # data = data_all
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>self.env_id: "+str(self.env_id))

        self.subjects_total, self.data_total, self.subjects, _ = get_subjects(data,0)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>subjects_total: "+str(self.subjects_total))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>data_total: "+str(self.data_total))

        self.reward_dic_on_cur_episode = []
        if self.mode is 'on_line':
            self.subjects_total = 1
            self.subjects = self.subjects[self.subject:self.subject+1]
            self.cur_training_step = 0
            self.cur_predicting_step = self.cur_training_step + 1
            self.predicting = False
            from config import train_to_reward
            self.train_to_reward = train_to_reward
            self.sum_reward_dic_on_cur_train = []
            self.average_reward_dic_on_cur_train = []



        '''init video and get paramters'''
        # video = cv2.VideoCapture('../../'+self.data_base+'/' + self.env_id + '.mp4')
        video = cv2.VideoCapture('/home/minglang/vr_new/'+self.env_id + '.mp4')
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>video: "+video)
        # video = cv2.VideoCapture('/home/minglang/vr_new/A380.mp4')
        self.frame_per_second = video.get(cv2.cv.CV_CAP_PROP_FPS)
        self.frame_total = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>frame_total: "+str(self.frame_total))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>self.env_id: "+str(self.env_id))
        self.video_size_width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.video_size_heigth = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        self.second_total = self.frame_total / self.frame_per_second
        self.data_per_frame = self.data_total / self.frame_total

        '''compute step lenth from data_tensity'''
        from config import data_tensity
        self.second_per_step = max(data_tensity/self.frame_per_second, data_tensity/self.data_per_frame/self.frame_per_second)
        self.frame_per_step = self.frame_per_second * self.second_per_step
        self.data_per_step = self.data_per_frame * self.frame_per_step

        '''compute step_total'''
        self.step_total = int(self.data_total / self.data_per_step) + 1
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>step_total: "+str(self.step_total))

        '''set fov range'''
        from config import view_range_lon, view_range_lat
        self.view_range_lon = view_range_lon
        self.view_range_lat = view_range_lat

        self.episode = 0

        self.max_cc = 0.0
        self.cur_cc = 0.0

        '''salmap'''
        self.heatmap_height = 180
        self.heatmap_width = 360

        if self.mode is 'data_processor':
            self.data_processor()

        '''load ground-truth heat map'''
        from config import heatmap_sigma
        gt_heatmap_dir = 'gt_heatmap_sp_' + heatmap_sigma
        self.gt_heatmaps = self.load_heatmaps(gt_heatmap_dir)

        if (self.mode is 'off_line') or (self.mode is 'data_processor'):
            from config import num_workers_global,cluster_current,cluster_main
            if (self.task%num_workers_global==0) and (cluster_current==cluster_main):
                print('>>>>>>>>>>>>>>>>>>>>this is a log thread<<<<<<<<<<<<<<<<<<<<<<<<<<')
                self.log_thread = True
            else:
                self.log_thread = False
        elif self.mode is 'on_line':
            print('>>>>>>>>>>>>>>>>>>>>this is a log thread<<<<<<<<<<<<<<<<<<<<<<<<<<')
            self.log_thread = True

        '''update settings for log_thread'''
        if self.log_thread:
            self.log_thread_config()

    def data_processor(self):
        from config import data_processor_id
        self.data_processor_id = data_processor_id

        print('==========================data process ===============================')

        print('==========================data process start: '+data_processor_id+'================================')

        if data_processor_id is 'minglang_mp4_to_yuv':
            print('sssss')
            from config import f_game_dic_new_test
            for i in range(len(f_game_dic_new_test)):
                # print(game_dic_new_all[i])
                if i >= 0 and i <= len(f_game_dic_new_test): #len(game_dic_new_all)
                    # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                    # file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_yuv/'+"Let'sNotBeAloneTonight"+'.yuv'
                    # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+"Let'sNotBeAloneTonight"+'.mp4'
                    ################################## wang ################################################

                    file_out_1 = '/media/minglang/My Passport/online_model/ff/save/'+self.env_id+'.yuv'
                    file_in_1 = '/media/minglang/My Passport/online_model/ff/save/'+self.env_id+'.mp4'
                    self.video = cv2.VideoCapture(file_in_1)
                    input_width_1 = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                    input_height_1 = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
                    self.mp4_to_yuv(input_width_1,input_height_1,file_in_1,file_out_1)
                    # print('end processing: ',file_out_1)
                    print('end processing: ')

            # print('len_game_dic_new_all: ',len(game_dic_new_all))
            # print('get_view')

            # print(game_dic_new_all)

        if data_processor_id is 'minglang_mp4_to_jpg':
            from config import f_game_dic_new_test
            # for i in range(len(f_game_dic_new_test)):
            # print(game_dic_new_all[i])
            i = 0
            # if i >= 0 and i <= len(f_game_dic_new_test): #len(game_dic_new_all)
                # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                # file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_yuv/'+"Let'sNotBeAloneTonight"+'.yuv'
                # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+"Let'sNotBeAloneTonight"+'.mp4'
            ###########################################change mp4 to jpg##############################################
            # file_in_1 = '/home/minglang/PAMI/test_video/'+self.env_id+'.mp4'
            # print('minglang/YuhangSong')
            # # file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_bms_jpg/'+str(f_game_dic_new_test[i])+'.jpg'
            #
            # video = cv2.VideoCapture(file_in_1)
            # self.video = video
            # self.frame_per_second = round(video.get(cv2.cv.CV_CAP_PROP_FPS))
            # self.frame_total = round(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            #
            # for frame_i in range(int(self.frame_total)):
            #     try:
            #         rval, frame = self.video.read()
            #         # frame = cv2.resize(frame,(self.salmap_width, self.salmap_height))
            #         # here minglang 1
            #         # cv2.imwrite('/media/minglang/YuhangSong_1/ff/vr_bms_jpg/'+str(game_dic_new_all[i])+'_'+str(frame_i)+'.jpg',frame)
            #         cv2.imwrite('/home/minglang/PAMI/test_video/frames_all/'+self.env_id+'_'+str(frame_i)+'.jpg',frame)
            #         print(frame_i)
            #     except Exception, e:
            #         print('failed on this frame, continue')
            #         print Exception,":",e
            #         continue
            #
            #         print('end processing: ',file_in_1,self.frame_per_second,self.frame_total)
            #########################################   save steps frames ########################################################
            heatmaps = []
            for step in range(self.step_total-1):
                data = int(round((step)*self.data_per_step))
                frame = int(round((step)*self.frame_per_step))
                try:
                    file_name = '/home/minglang/PAMI/test_video/frames_all/'+self.env_id+'_'+str(frame)+'.jpg'
                    # file_name = source_path + str(self.env_id)+'_'+str(step)+'.jpg'
                    print(">>>>>>>>>>>>>>save_ours_heatmap----self.env_id-----print step/step_total: ",self.env_id,step,self.step_total)
                    temp = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_COLOR)
                    temp1 = cv2.resize(temp,(self.salmap_width, self.salmap_height))
                    file_name_steps = '/home/minglang/PAMI/test_video/frames_steps_downsample/'+str(self.env_id)+'_'+str(step)+'.jpg'
                    path = '/home/minglang/PAMI/test_video/frames_steps_downsample/'
                    if os.path.exists(path) is False:
                        os.mkdir(path)
                    cv2.imwrite(file_name_steps,temp)
                except Exception,e:
                    print Exception,":",e
                    continue


        if data_processor_id is 'minglang_avi_to_jpg':
            from config import f_game_dic_new_test

            file_in_1 = '/home/minglang/PAMI/salicon/'+self.env_id+'_out'+'.avi'
            print('inglang/YuhangSong')

            video = cv2.VideoCapture(file_in_1)
            self.video = video
            self.frame_per_second = round(video.get(cv2.cv.CV_CAP_PROP_FPS))
            self.frame_total = round(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

            for frame_i in range(int(self.frame_total)):

                try:
                    rval, frame = self.video.read()
                    # frame = frame / np.float32(np.max(frame))
                    # print(">>>>>>>>>>>>>>>>>>>>>>>>>np.float32(np.max(frame)): ",np.float32(np.max(frame)))
                    # frame = frame * 255.0

                    # for x in range(180):
                    #     for y in range(360):
                    #         if frame[x][y] > 0:
                    #             print("heatmap:",x,y,frame[x][y])

                    cv2.imwrite('/home/minglang/PAMI/salicon/salicon_all_jpg/'+self.env_id+'_'+str(frame_i)+'.jpg',frame)
                    print(frame_i)
                except Exception, e:
                    print('failed on this frame, continue')
                    print Exception,":",e
                    continue



            print('end processing: ',file_in_1,self.frame_per_second,self.frame_total)


        if data_processor_id is 'minglang_obdl_cfg':
            from config import game_dic_new_all, data_processor_mode
            for i in range(len(game_dic_new_all)):
                # print(game_dic_new_all[i])

                if i >= 100 and i <=  len(game_dic_new_all): #len(game_dic_new_all)
                    # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                    # file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_yuv/'+"Let'sNotBeAloneTonight"+'.yuv'
                    # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+"Let'sNotBeAloneTonight"+'.mp4'
                    file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                    CONFIG_FILE = '/media/minglang/YuhangSong_1/ff/obdl_vr_new/'+str(game_dic_new_all[i])+'.cfg'

                    # # get the paramters
                    video = cv2.VideoCapture(file_in_1)
                    self.video = video
                    self.frame_per_second = int(round(video.get(cv2.cv.CV_CAP_PROP_FPS)))
                    self.frame_total = int(round(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))
                    NAME = game_dic_new_all[i]
                    FRAMESCOUNT = self.frame_total
                    FRAMERATE = self.frame_per_second
                    IMAGEWIDTH = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                    IMAGEHEIGHT = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

                    # write the paramters throuh cfg
                    # conf = ConfigParser.ConfigParser()
                    # cfgfile = open(CONFIG_FILE,'w')
                    # # conf.add_section("")

                    # write through txt
                    f_config = open(CONFIG_FILE,"w")
                    f_config.write("NAME\n")
                    f_config.write(str(game_dic_new_all[i])+'\n')
                    f_config.write("FRAMESCOUNT\n")
                    f_config.write(str(FRAMESCOUNT)+'\n')
                    f_config.write("FRAMERATE\n")
                    f_config.write(str(FRAMERATE)+'\n')
                    f_config.write("IMAGEWIDTH\n")
                    f_config.write(str(IMAGEWIDTH)+'\n')
                    f_config.write("IMAGEHEIGHT\n")
                    f_config.write(str(IMAGEHEIGHT)+'\n')
                    f_config.close()

                #one video and one cfg in one file
                if i >= 0 and i <= len(game_dic_new_all): #len(game_dic_new_all)
                    cfg_file = '/media/minglang/YuhangSong_1/ff/obdl_vr_new/obdl_vr_new/'+str(game_dic_new_all[i])
                    os.makedirs(cfg_file)
                    file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                    shutil.copy(file_in_1,cfg_file)
                    CONFIG_FILE = '/media/minglang/YuhangSong_1/ff/obdl_vr_new/'+str(game_dic_new_all[i])+'.cfg'
                    shutil.copy(CONFIG_FILE,cfg_file)
                    print("os.makedirs(cfg_file)")

        # get and save groundtruth_heatmap
        if data_processor_id is 'minglang_get_ground_truth_heatmap':

            print('>>>>>>>>>>>>>>>>>>>>minglang_get_ground_truth_heatmap<<<<<<<<<<<<<<<<<<<<<<<<<')
            # print(dsnfj)
            self.save_gt_heatmaps()
            print('>>>>>>>>>>>>>>>>>>>>end<<<<<<<<<<<<<<<<<<<<<<<<<')

        # get and save groundtruth_heatmap,now0
        if data_processor_id is 'minglang_get_ground_truth_heatmap_for_nss':
            print('>>>>>>>>>>>>>>>>>>>> minglang_get_ground_truth_heatmap_for_nss start')
            # self.test_save()
            self.save_gt_groundtruth_heatmaps_for_nss(Multi_255 = False,save_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/')
            # self.gt_heatmaps_groundtruth = self.load_gt_heatmaps_groundtruth(Multi_255 = False,nss_cc = True, groundtruth_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/')
            print('>>>>>>>>>>>>>>>>>>>> minglang_get_ground_truth_heatmap_for_nss end')


        # get and save fcb_heatmap
        if data_processor_id is 'minglang_get_fcb':
            print('>>>>>>>>>>>>>>>>>>>>minglang_get_fcb start <<<<<<<<<<<<<<<<<<<<<<<<<')
            for step in range(self.step_total):
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: ", step)
                data = int(round((step)*self.data_per_step))
                frame = int(round((step)*self.frame_per_step))
                try:
                    temp = self.fixation2salmap_fcb_2dim([[0.0,0.0]], self.salmap_width, self.salmap_height, sigma = self.fcb_sigma)
                    max_temp = np.float32(np.max(temp))
                    temp = (temp/max_temp)
                    self.save_heatmap(heatmap = temp,
                                      path = '/home/minglang/PAMI/fcb/fcb_'+ str(self.fcb_sigma) + '/',
                                      name = str(step))
                except Exception,e:
                    print Exception,":",e
                    continue

            print('>>>>>>>>>>>>>>>>>>>>minglang_get_fcb_end<<<<<<<<<<<<<<<<<<<<<<<<<')

        # get ming_fcb_ss_cc
        if data_processor_id is 'minglang_get_fcb_groundhp_ss_cc':
            # print('>>>>>>>>>>>>>>>>>>>>ming_fcb_SS_cc<<<<<<<<<<<<<<<<<<<<<<<<<')
            # ####################################### cal cc #################################################
            # # fcb_value = [6, 7, 12,18,21,22,23,24,25,26,27,28,30,33,36,42,48]
            # fcb_value = [21]
            # for i in range(len(fcb_value)):
            #     self.cal_cc(
            #             ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap/',
            #             prediction_src_path = '/home/minglang/PAMI/fcb/fcb_' + str(fcb_value[i]) + '/',
            #             dst_all_cc_path ='/home/minglang/PAMI/cc_result/fcb_and_ground/cc_all_' + str(fcb_value[i]) + '/',
            #             dst_ave_cc_path ='/home/minglang/PAMI/cc_result/fcb_and_ground/ave_cc_' + str(fcb_value[i]) + '/')
            #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_fcb_groundhp_ss_cc end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<',i)
            # print(myx)
            ############################################### cal nss  ###################################
            self.cal_nss(
                    Multi_255 = False,
                    ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/',
                    prediction_src_path = '/home/minglang/PAMI/fcb/fcb_' + str(self.fcb_sigma) + '/',
                    dst_all_ss_path ='/home/minglang/PAMI/ss_result /fcb_and_ground/ss_all_' + str(self.fcb_sigma) + '/',
                    dst_ave_ss_path ='/home/minglang/PAMI/ss_result /fcb_and_ground/ave_ss_' + str(self.fcb_sigma) + '/')
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> minglang_get_fcb_groundhp_ss_cc end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        if data_processor_id is 'ave_for_fitting':
            ''' load the cc result,calculate ave_cc in all videos,save the result
                ine one txt
            '''
            # "weight parametr"
            # self.w_prediction = 0.5
            # self.w_fcb = 1 - self.w_prediction # 0.1374
            "fcb parameter"
            self.fcb_sigma = 12
            self.fcb_map = self.fixation2salmap_fcb_2dim([[0.0,0.0]], self.salmap_width, self.salmap_height,sigma = self.fcb_sigma)
            pass
            self.cal_ave_cc_all_videos(
                                        source_path = '/home/minglang/PAMI/cc_result/ours_and_ground/ave_cc_with_fcb_fitting/',
                                        save_path = '/home/minglang/PAMI/cc_result/ours_and_ground/all_videos_ave_cc_with_fcb_fitting/'
            )
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>ave_for_fitting teminate")
            print(myx)

        # minglang_get_ours_groundhp_cc,now1
        if data_processor_id is 'minglang_get_ours_groundhp_ss_cc':
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_ours_groundhp_ss_cc star<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            with_fcb = True # True Flase

            # "weight parametr"
            # self.w_prediction = 0.5
            # self.w_fcb = 1 - self.w_prediction # 0.1374
            "fcb parameter"
            self.fcb_sigma = 12
            self.fcb_map = self.fixation2salmap_fcb_2dim([[0.0,0.0]], self.salmap_width, self.salmap_height,sigma = self.fcb_sigma)
            ###################### get the heatmap #########################
            # self.save_ours_heatmap_svm()
            #####################  cal the CC  ############################
            ##########self.cal_ours_ground_cc()
            if with_fcb is True:
                # self.cal_cc(
                #         ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap/',
                #         # prediction_src_path = '/home/minglang/PAMI/ff_best_heatmaps_ours/ff_best_heatmaps_ours_with_fcb_' + str(self.fcb_sigma) + '/',
                #         prediction_src_path = '/home/minglang/PAMI/ff_best_heatmaps_ours/ff_best_heatmaps_ours_with_fcb_svm_12_' + status_fcb +'/',
                #         dst_all_cc_path ='/home/minglang/PAMI/cc_result/ours_and_ground/cc_all_steps_with_fcb_svm_12_' + status_fcb + '_' + str(self.fcb_sigma) + '/',
                #         dst_ave_cc_path ='/home/minglang/PAMI/cc_result/ours_and_ground/ave_cc_with_fcb_svm_svm_12_' + status_fcb + '_' + str(self.fcb_sigma) + '/')
                self.cal_cc_fcb_fitting(
                                        ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap/',
                                        prediction_src_path = '/home/minglang/PAMI/ff_best_heatmaps_ours/ff_best_heatmaps_ours_without_fcb/',
                                        dst_all_cc_path ='/home/minglang/PAMI/cc_result/ours_and_ground/cc_all_steps_with_fcb_svm_12/',
                                        dst_ave_cc_path ='/home/minglang/PAMI/cc_result/ours_and_ground/ave_cc_with_fcb_fitting/')

                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_ours_groundhp_ss_cc with_fcb end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            else:
                self.cal_cc(
                        ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap/',
                        prediction_src_path = '/home/minglang/PAMI/ff_best_heatmaps_ours/ff_best_heatmaps_ours_without_fcb/',
                        dst_all_cc_path ='/home/minglang/PAMI/cc_result/ours_and_ground/cc_all_steps/' ,
                        dst_ave_cc_path ='/home/minglang/PAMI/cc_result/ours_and_ground/ave_cc/')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_ours_groundhp_ss_cc end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            ############################### cal nss #########################################
            # if with_fcb is True:
            #     self.cal_nss(
            #             Multi_255 = False,
            #             ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/',
            #             prediction_src_path = '/home/minglang/PAMI/ff_best_heatmaps_ours/ff_best_heatmaps_ours_with_fcb/',
            #             dst_all_ss_path ='/home/minglang/PAMI/ss_result /ours_and_ground/ss_all_steps_with_fcb_' + str(self.fcb_sigma) + '/' ,
            #             dst_ave_ss_path ='/home/minglang/PAMI/ss_result /ours_and_ground/ave_ss_with_fcb_' + str(self.fcb_sigma) + '/')
            #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_ours_groundhp_ss with_fcb end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            # else:
            #     self.cal_nss(
            #             Multi_255 = False,
            #             ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/',
            #             prediction_src_path = '/home/minglang/PAMI/ff_best_heatmaps_ours/ff_best_heatmaps_ours_without_fcb/',
            #             dst_all_ss_path ='/home/minglang/PAMI/ss_result /ours_and_ground/ss_all_steps_' + str(self.fcb_sigma) + '/',
            #             dst_ave_ss_path ='/home/minglang/PAMI/ss_result /ours_and_ground/ave_ss_' + str(self.fcb_sigma) + '/')
            #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_ours_groundhp_ss_cc end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        # minglang_get_bms_groundtruh,nowb
        if data_processor_id is 'minglang_get_bms_groundhp_ss_cc':

            ##################################### get the bms with step #######################################
            # self.save_bms_heatmaps(with_fcb = True) # True False
            # ################################# cal minglang_get_bms_groundhp_cc ################################
            with_fcb = True # True Flase
            # ####self.cal_bms_ground_cc()
            # if with_fcb is True:
            #     self.cal_cc(
            #             ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap/',
            #             prediction_src_path = '/home/minglang/PAMI/bms/bms_step_with_fcb_' + str(self.fcb_sigma) + '/' ,
            #             dst_all_cc_path ='/home/minglang/PAMI/cc_result/bms_and_ground/cc_all_steps_with_fcb_' + str(self.fcb_sigma) + '/' ,
            #             dst_ave_cc_path ='/home/minglang/PAMI/cc_result/bms_and_ground/ave_cc_with_fcb_' + str(self.fcb_sigma) + '/')
            #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_bms_groundhp_ss_cc with_fcb end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            # else: # Flase
            #     self.cal_cc(
            #             ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap/',
            #             prediction_src_path = '/home/minglang/PAMI/bms/bms_step/',
            #             dst_all_cc_path ='/home/minglang/PAMI/cc_result/bms_and_ground/cc_all_steps/' ,
            #             dst_ave_cc_path ='/home/minglang/PAMI/cc_result/bms_and_ground/ave_cc/')
            #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_bms_groundhp_ss_cc end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> cal minglang_get_bms_groundhp_cc/nss  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            ####################################  cal nss  ########################################################
            # self.cal_nss(src_path = '/home/minglang/PAMI/bms/bms_step/' ,
            #             dst_all_ss_path = '/home/minglang/PAMI/ss_result /bms_and_ground/ss_all/' ,
            #             dst_ave_ss_path = '/home/minglang/PAMI/ss_result /bms_and_ground/ave_ss/')
            if with_fcb is False:
                self.cal_nss(
                        Multi_255 = False,
                        ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/',
                        prediction_src_path = '/home/minglang/PAMI/bms/bms_step/',
                        dst_all_ss_path ='/home/minglang/PAMI/ss_result /bms_and_ground/ss_all_' + str(self.fcb_sigma) + '/' ,
                        dst_ave_ss_path ='/home/minglang/PAMI/ss_result /bms_and_ground/ave_ss_' + str(self.fcb_sigma) + '/')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_ours_groundhp_ss without_fcb end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            else:
                self.cal_nss(
                        Multi_255 = False,
                        ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/',
                        prediction_src_path = '/home/minglang/PAMI/bms/bms_step_with_fcb_' + str(self.fcb_sigma) + '/',
                        dst_all_ss_path ='/home/minglang/PAMI/ss_result /bms_and_ground/ss_all_with_fcb_' + str(self.fcb_sigma) + '/' ,
                        dst_ave_ss_path ='/home/minglang/PAMI/ss_result /bms_and_ground/ave_ss_with_fcb_' + str(self.fcb_sigma) + '/')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_ours_groundhp_ss without_fcb end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            # data_processor_id = data_processor_mode['10']

        # minglang_get_obdl_groundtruh
        if data_processor_id is 'minglang_get_obdl_groundhp_ss_cc':
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.obdl_begin<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            ######################## get the obdl with step #################################
            # self.save_obdl_heatmaps(with_fcb = True)
            ######################## cal minglang_get_obdl_groundhp_cc ######################
            #################### self.cal_obdl_ground_cc()
            with_fcb = True # True Flase
            # if with_fcb is True:
            #     self.cal_cc(
            #             ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap/',
            #             prediction_src_path = '/home/minglang/PAMI/obdl_out/obdl_steps_with_fcb_' + str(self.fcb_sigma) + '/',
            #             dst_all_cc_path ='/home/minglang/PAMI/cc_result/obdl_and_ground/cc_all_steps_with_fcb_' + str(self.fcb_sigma) + '/'  ,
            #             dst_ave_cc_path ='/home/minglang/PAMI/cc_result/obdl_and_ground/ave_cc_with_fcb_' + str(self.fcb_sigma) + '/' )
            #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_bms_groundhp_ss_cc with_fcb end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            # else:
            #     self.cal_cc(
            #             ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap/',
            #             prediction_src_path = '/home/minglang/PAMI/obdl_out/obdl_steps/',
            #             dst_all_cc_path ='/home/minglang/PAMI/cc_result/obdl_and_ground/cc_all_steps/' ,
            #             dst_ave_cc_path ='/home/minglang/PAMI/cc_result/obdl_and_ground/ave_cc/')
            #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_obdl_groundhp_ss_cc end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

            ######################## cal nss ################################################

            if with_fcb is False:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_obdl_groundhp_ss without_fcb begin<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                self.cal_nss(
                        Multi_255 = False,
                        ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/',
                        prediction_src_path = '/home/minglang/PAMI/obdl_out/obdl_steps/',
                        dst_all_ss_path ='/home/minglang/PAMI/ss_result /obdl_and_ground/ss_all_' + str(self.fcb_sigma) + '/' ,
                        dst_ave_ss_path ='/home/minglang/PAMI/ss_result /obdl_and_ground/ave_ss_' + str(self.fcb_sigma) + '/')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_obdl_groundhp_ss without_fcb end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            else:
                self.cal_nss(
                        Multi_255 = False,
                        ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/',
                        prediction_src_path = '/home/minglang/PAMI/obdl_out/obdl_steps_with_fcb_' + str(self.fcb_sigma) + '/',
                        dst_all_ss_path ='/home/minglang/PAMI/ss_result /obdl_and_ground/ss_all_with_fcb_' + str(self.fcb_sigma) + '/' ,
                        dst_ave_ss_path ='/home/minglang/PAMI/ss_result /obdl_and_ground/ave_ss_with_fcb_' + str(self.fcb_sigma) + '/')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_obdl_groundhp_ss with_fcb end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            # data_processor_id = data_processor_mode['11']

        # minglang_get_salicon_groundtruh_cc,now2
        if data_processor_id is 'minglang_get_salicon_groundhp_ss_cc':
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>salicon_begin<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            with_fcb = True
            ################################# get the obdl with step #####################################
            # if with_fcb is False:
            #     self.save_salicon_setps_heatmap(src_path ='/home/minglang/PAMI/salicon/salicon_all_jpg/' ,
            #                             dst_path ='/home/minglang/PAMI/salicon/salicon_steps_jpg/',
            #                             with_fcb = False)
            # else:
            #     self.save_salicon_setps_heatmap(src_path ='/home/minglang/PAMI/salicon/salicon_all_jpg/' ,
            #                             dst_path ='/home/minglang/PAMI/salicon/salicon_steps_jpg_withfcb_' + str(self.fcb_sigma) + '/',
            #                             with_fcb = True)
            ##################################### cal the cc ############################################################
            ########################## minglang_get_obdl_groundhp_cc
            # if with_fcb is True:
            #     self.cal_cc(
            #             ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap/',
            #             prediction_src_path = '/home/minglang/PAMI/salicon/salicon_steps_jpg_withfcb_' + str(self.fcb_sigma) + '/',
            #             dst_all_cc_path ='/home/minglang/PAMI/cc_result/salicon_and_ground /cc_all_steps_with_fcb_' + str(self.fcb_sigma) + '/' ,
            #             dst_ave_cc_path ='/home/minglang/PAMI/cc_result/salicon_and_ground /ave_cc_with_fcb_' + str(self.fcb_sigma) + '/')
            #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_bms_groundhp_ss_cc with_fcb end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            # else:
            #     self.cal_cc(
            #             ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap/',
            #             prediction_src_path = '/home/minglang/PAMI/salicon/salicon_steps_jpg/',
            #             dst_all_cc_path ='/home/minglang/PAMI/cc_result/salicon_and_ground /cc_all_steps/',
            #             dst_ave_cc_path ='/home/minglang/PAMI/cc_result/salicon_and_ground /ave_cc/')
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>minglang_get_salicon_groundhp_ss_cc end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> cal nss <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if with_fcb is False:
                self.cal_nss(
                            Multi_255 = False,
                            ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/',
                            prediction_src_path = '/home/minglang/PAMI/salicon/salicon_steps_jpg/',
                            dst_all_ss_path = '/home/minglang/PAMI/ss_result /salicon_and_ground /ss_all_' + str(self.fcb_sigma) + '/' ,
                            dst_ave_ss_path = '/home/minglang/PAMI/ss_result /salicon_and_ground /ave_ss_' + str(self.fcb_sigma) + '/')
            else:
                self.cal_nss(
                            Multi_255 = False,
                            ground_src_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/',
                            prediction_src_path = '/home/minglang/PAMI/salicon/salicon_steps_jpg_withfcb/' ,
                            dst_all_ss_path = '/home/minglang/PAMI/ss_result /salicon_and_ground /ss_all_with_fcb_' + str(self.fcb_sigma) + '/' ,
                            dst_ave_ss_path = '/home/minglang/PAMI/ss_result /salicon_and_ground /ave_ss_with_fcb_' + str(self.fcb_sigma) + '/')

        '''terminate the dataprocesor'''
        print('=============================data processor end, terminate=============================')
        print(t)

    def save_salicon_setps_heatmap(self,src_path,dst_path,with_fcb = False):
        heatmaps = []
        for step in range(self.step_total):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))

            try:
                # file_name = '/home/minglang/PAMI/gt_heatmap_sp_sigma_half_fov/'+self.env_id+'_'+str(step)+'.jpg'
                #for our's hmap
                file_name = src_path+str(self.env_id)+'_'+str(step)+'.jpg'
                # file_name = source_path + str(self.env_id)+'_'+str(step)+'.jpg'
                print(">>>>>>>>>>>>>>load_gt_heatmaps----self.env_id-----print step: ",self.env_id,step)
                temp = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE) # read as gray picture
                # temp = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_COLOR)
                temp = cv2.resize(temp,(self.salmap_width, self.salmap_height))
                temp = temp * (1.0 / np.max(temp))
                print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp-1)", np.max(temp))

                # with fcb
                if with_fcb is True:
                    temp = temp
                    print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp0),np.max(self.fcb_map)", np.max(temp), np.max(self.fcb_map))
                    temp = np.multiply(temp,self.fcb_map)
                    print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp1)",np.max(temp))
                    temp = temp * (1.0 /np.max(temp))
                    print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp2)",np.max(temp))
                    # print(myx)

                self.save_heatmap(heatmap=temp,
                                  path=dst_path,
                                  name=str(step))
            except Exception,e:
                print Exception,":",e
                continue
        print(s)

    # nowc
    def cal_cc(self,ground_src_path, prediction_src_path,dst_all_cc_path,dst_ave_cc_path):
        ccs = []
        fcb_maps = []

        self.gt_heatmaps_ours = self.load_gt_heatmaps(source_path = prediction_src_path) #load ours heatmap
        self.gt_heatmaps_groundtruth = self.load_gt_heatmaps_groundtruth(groundtruth_path = ground_src_path) #load ground-truth heatmap
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> np.shape(self.gt_heatmaps)_ours: ',np.shape(self.gt_heatmaps_ours))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> np.shape(self.gt_heatmaps)_gt_heatmaps_groundtruth: ',np.shape(self.gt_heatmaps_groundtruth))

        if os.path.exists(dst_all_cc_path) is False:
            os.mkdir(dst_all_cc_path)
        if os.path.exists(dst_ave_cc_path) is False:
            os.mkdir(dst_ave_cc_path)

        'cal cc for one video with one parameter'
        for step in range(self.step_total-1):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))
            print('>>>>>cal_ours_ground_cc-------self.step_total----step: ',self.step_total,step)
            cc = self.calc_score(self.gt_heatmaps_ours[step],self.gt_heatmaps_groundtruth[step])
            if math.isnan(float(cc)) is False:
                self.save_step_cc(cc=cc,
                                  step=step,
                                  path = dst_all_cc_path)
                # if(step > 0):
                ccs += [cc]
        self.cc_averaged = sum(ccs)/len(ccs)
        self.save_ave_cc(self.cc_averaged,
                         path = dst_ave_cc_path)
        print("cc average "+str(self.cc_averaged))

        print('>>>>>>>>>>>>>>>>>>>>cal_obdl_ground_cc_end<<<<<<<<<<<<<<<<<<<<<<<<<')


    # nowc
    def cal_cc_fcb_fitting(self,ground_src_path, prediction_src_path,dst_all_cc_path,dst_ave_cc_path):
        ccs = []
        fcb_maps = []

        self.gt_heatmaps_ours = self.load_gt_heatmaps(source_path = prediction_src_path) #load ours heatmap
        self.gt_heatmaps_groundtruth = self.load_gt_heatmaps_groundtruth(groundtruth_path = ground_src_path) #load ground-truth heatmap
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> np.shape(self.gt_heatmaps)_ours: ',np.shape(self.gt_heatmaps_ours))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> np.shape(self.gt_heatmaps)_gt_heatmaps_groundtruth: ',np.shape(self.gt_heatmaps_groundtruth))

        if os.path.exists(dst_all_cc_path) is False:
            os.mkdir(dst_all_cc_path)
        if os.path.exists(dst_ave_cc_path) is False:
            os.mkdir(dst_ave_cc_path)

        for weight_i in range(10,100,1):
            "weight parametr"
            self.w_prediction = weight_i * 1.0 / 100
            self.w_fcb = 1 - self.w_prediction # 0.1374

            'cal cc loop start'
            for step in range(self.step_total-1):
                data = int(round((step)*self.data_per_step))
                frame = int(round((step)*self.frame_per_step))
                print('>>>>>cal_ours_ground_cc-------self.step_total----step: ',self.step_total,step)
                ''' multipy the fcb for fitting'''
                print(">>>>>>>>>>>>>>np.max(self.gt_heatmaps_ours[step]),np.max(self.fcb_map),self.w_prediction,self.w_fcb: ",\
                      np.max(self.gt_heatmaps_ours[step]), np.max(self.fcb_map), self.w_prediction, self.w_fcb)
                self.gt_heatmaps_ours[step] = self.gt_heatmaps_ours[step] * self.w_prediction + self.fcb_map * self.w_fcb
                self.gt_heatmaps_ours[step] = self.gt_heatmaps_ours[step] * (1.0 / np.max(self.gt_heatmaps_ours[step])) * 255
                print('>>>>>>>>>>>>>>>>>>>>>>>>np.max(self.gt_heatmaps_ours[step]),np.max(self.gt_heatmaps_groundtruth[step]): ',\
                       np.max(self.gt_heatmaps_ours[step]),np.max(self.gt_heatmaps_groundtruth[step]))
                # print(myx)
                cc = self.calc_score(self.gt_heatmaps_ours[step],self.gt_heatmaps_groundtruth[step])
                if math.isnan(float(cc)) is False:
                    # self.save_step_cc(cc=cc,
                    #                   step=step,
                    #                   path = dst_all_cc_path)
                    # if(step > 0):
                    ccs += [cc]

            self.cc_averaged = sum(ccs)/len(ccs)
            self.save_ave_cc(self.cc_averaged,
                             path = dst_ave_cc_path)
            print("cc average "+str(self.cc_averaged))
            'cal cc loop end'

        print('>>>>>>>>>>>>>>>>>>>>cal_obdl_ground_cc_end<<<<<<<<<<<<<<<<<<<<<<<<<')


    # nows(for location here quickly)
    def cal_nss(self,Multi_255,ground_src_path, prediction_src_path, dst_all_ss_path,dst_ave_ss_path):
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>cal nss begin<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        ccs = []
        fcb_maps = []
        " get the groundtruth hmap "
        self.gt_heatmaps_ours = self.load_gt_heatmaps(source_path = prediction_src_path) #load ours heatmap
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>cal_nss self.gt_heatmaps_ours end: ")
        # " get the prediction hmap "
        # if Multi_255 is True:
        #     ''' notice that this function has /255.0 to the map '''
        #     self.gt_heatmaps_groundtruth = self.load_gt_heatmaps_groundtruth(Multi_255 = True,nss_cc = True, groundtruth_path = ground_src_path) #load ground-truth heatmap
        # else:
        #     ''' notice that this function keep the raw value the map '''
        #     self.gt_heatmaps_groundtruth = self.load_gt_heatmaps_groundtruth(Multi_255 = False,nss_cc = True, groundtruth_path = ground_src_path) #load ground-truth heatmap

        self.get_all_txt_groundtruth = self.load_get_all_txt_groundtruth_nss(groundtruth_path = ground_src_path)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>self.load_all_txt_groundtruth end: ")
        if os.path.exists(dst_all_ss_path) is False:
            os.mkdir(dst_all_ss_path)
        if os.path.exists(dst_ave_ss_path) is False:
            os.mkdir(dst_ave_ss_path)

        for step in range(self.step_total-1):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))
            # print('>>>>>cal_ours_ground_cc-------self.step_total----step: ',self.step_total,step)
            cc = self.calc_score_nss(gtsAnn = self.get_all_txt_groundtruth[step], resAnn =  self.gt_heatmaps_ours[step])
            print('>>>>>>>>>>>>>>>def cal_nss,step, ss: ',step, cc)
            self.save_step_cc(cc=cc,
                              step=step,
                              path = dst_all_ss_path)
            ''' ignore the negative result in order to improve the cc '''
            # if(step > 0):
            ccs += [cc]

        self.cc_averaged = sum(ccs)/len(ccs)
        self.save_ave_cc(self.cc_averaged,
                         path = dst_ave_ss_path)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>def cal_nss,ss average  "+str(self.cc_averaged))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>cal nss end<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    def save_ours_heatmap(self):
        heatmaps = []
        for step in range(self.step_total-1):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))

            try:
                # file_name = '/home/minglang/PAMI/gt_heatmap_sp_sigma_half_fov/'+self.env_id+'_'+str(step)+'.jpg'
                #for our's hmap
                file_name = '/home/minglang/PAMI/ff_best_heatmaps_ours/'+str(self.env_id)+'/'+str(step)+'.jpg'
                # file_name = source_path + str(self.env_id)+'_'+str(step)+'.jpg'
                print(">>>>>>>>>>>>>>save_ours_heatmap----self.env_id-----print step/step_total: ",self.env_id,step,self.step_total)
                temp = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                temp = cv2.resize(temp,(self.salmap_width, self.salmap_height))
                # ##################################  with fcb ########################################
                temp = temp * (1.0 / 255.0)
                print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp),np.max(self.fcb_map)", np.max(temp), np.max(self.fcb_map))
                temp = np.multiply(temp,self.fcb_map)
                print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp1)",np.max(temp))
                temp = temp * (1.0 /np.max(temp))
                print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp2)",np.max(temp))
                # print(myx)
                # temp = temp / 255.0
                self.save_heatmap(heatmap=temp,
                                #   path='/home/minglang/PAMI//ff_best_heatmaps_ours/ff_best_heatmaps_ours_without_fcb/',
                                  path='/home/minglang/PAMI/ff_best_heatmaps_ours/ff_best_heatmaps_ours_with_fcb_21/',
                                  name=str(step))
            except Exception,e:
                print Exception,":",e
                continue

        print(s)

    def save_ours_heatmap_svm(self):
        heatmaps = []
        for step in range(self.step_total-1):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))

            try:
                # file_name = '/home/minglang/PAMI/gt_heatmap_sp_sigma_half_fov/'+self.env_id+'_'+str(step)+'.jpg'
                #for our's hmap
                file_name = '/home/minglang/PAMI/ff_best_heatmaps_ours/'+str(self.env_id)+'/'+str(step)+'.jpg'
                # file_name = source_path + str(self.env_id)+'_'+str(step)+'.jpg'
                print(">>>>>>>>>>>>>>save_ours_heatmap----self.env_id-----print step/step_total: ",self.env_id,step,self.step_total)
                temp = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                temp = cv2.resize(temp,(self.salmap_width, self.salmap_height))
                # ##################################  with fcb ########################################
                # temp = temp * (1.0 / 255.0)
                print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp),np.max(self.fcb_map)", np.max(temp), np.max(self.fcb_map))
                temp = temp * self.w_prediction + self.fcb_map * self.w_fcb
                # temp0 = temp - self.mean_feature[1]
                # temp1 = self.fcb_map - self.mean_feature[0]
                # temp0 = temp0 * 1.0 / self.stdfeature[1]
                # temp1 = temp1 *1.0 / self.stdfeature[0]
                # temp = temp1 * self.w_fcb + temp0 * self.w_prediction
                # print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp1)",np.max(temp))
                temp = temp * (1.0 /np.max(temp))
                print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp2)",np.max(temp))
                # print(myx)
                # temp = temp / 255.0
                self.save_heatmap(heatmap=temp,
                                #   path='/home/minglang/PAMI//ff_best_heatmaps_ours/ff_best_heatmaps_ours_without_fcb/',
                                  path='/home/minglang/PAMI/ff_best_heatmaps_ours/ff_best_heatmaps_ours_with_fcb_svm_12_4/',
                                  name=str(step))
            except Exception,e:
                print Exception,":",e
                continue

        print(s)


    def save_obdl_heatmaps(self,with_fcb = False):
        print('save_obdl_heatmaps')
        matfn = '/home/minglang/PAMI/obdl_out/'+self.env_id+'/result_OBDL-MRF_H264_QP37.mat'
        obdl_heatmap_framed = sio.loadmat(matfn)['S']
        heatmaps=[]
        for step in range(self.step_total):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))
            try:
                heatmap = obdl_heatmap_framed[:,:,frame] / 255.0
                heatmap = cv2.resize(heatmap,(self.salmap_width, self.salmap_height))
                print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp0)", np.max(heatmap))
                if with_fcb is False:
                    temp = heatmap
                    print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp0),np.max(self.fcb_map)", np.max(temp), np.max(self.fcb_map))
                    temp = np.multiply(temp,self.fcb_map)
                    self.save_heatmap(heatmap=heatmap,
                                      path='/home/minglang/PAMI/obdl_out/obdl_steps/',
                                      name=str(step))
                else:
                    # with fcb
                    temp = heatmap
                    print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp0),np.max(self.fcb_map)", np.max(temp), np.max(self.fcb_map))
                    temp = np.multiply(temp,self.fcb_map)
                    print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp1)",np.max(temp))
                    temp = temp * (1.0 /np.max(temp))
                    print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp2)",np.max(temp))
                    # print(myx)
                    self.save_heatmap(heatmap=temp,
                                      path='/home/minglang/PAMI/obdl_out/obdl_steps_with_fcb_' + str(self.fcb_sigma) + "/" ,
                                      name=str(step))

                heatmaps += [heatmap]
                print(np.shape(heatmaps))
            except Exception,e:
                print Exception,":",e
                continue
        print(s)

    def cal_obdl_ground_cc(self):
        print('>>>>>>>>>>>>>>>>>>>>cal_obdl_ground_cc<<<<<<<<<<<<<<<<<<<<<<<<<')
        ccs = []
        fcb_maps = []

        self.gt_heatmaps_ours = self.load_gt_heatmaps(source_path = '/home/minglang/PAMI/obdl_out/obdl_steps_with_fcb/') #load ours heatmap
        self.gt_heatmaps_groundtruth = self.load_gt_heatmaps_groundtruth() #load ground-truth heatmap
        print('np.shape(self.gt_heatmaps)_ours: ',np.shape(self.gt_heatmaps_ours))
        print('np.shape(self.gt_heatmaps)_gt_heatmaps_groundtruth: ',np.shape(self.gt_heatmaps_groundtruth))

        for step in range(self.step_total-1):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))
            print('>>>>>cal_ours_ground_cc-------self.step_total----step: ',self.step_total,step)
            # self.save_heatmap(heatmap=fcb_map,
            #                   path='/home/minglang/PAMI/fcb_err',
            #                   name=str(self.env_id))
            cc = self.calc_nss(self.gt_heatmaps_ours[step],self.gt_heatmaps_groundtruth[step])
            print(step)
            self.save_step_cc(cc=cc,
                              step=step,
                              path = '/home/minglang/PAMI/cc_result/obdl_and_ground/cc_all_with_fcb/')
            if(step > 0):
              ccs += [cc]

        print("here right")
            # print(sfd)
        self.cc_averaged = sum(ccs)/len(ccs)
        self.save_ave_cc(self.cc_averaged,
                         path = '/home/minglang/PAMI/cc_result/obdl_and_ground/ave_cc_with_fcb/')
        print("cc average "+str(self.cc_averaged))
        print('>>>>>>>>>>>>>>>>>>>>cal_obdl_ground_cc_end<<<<<<<<<<<<<<<<<<<<<<<<<')


    def cal_bms_ground_cc(self):
        print('>>>>>>>>>>>>>>>>>>>>cal_bms_ground_cc<<<<<<<<<<<<<<<<<<<<<<<<<')
        ccs = []
        fcb_maps = []

        self.gt_heatmaps_ours = self.load_gt_heatmaps(source_path = '/home/minglang/PAMI/bms/bms_step_with_fcb/') #load ours heatmap
        self.gt_heatmaps_groundtruth = self.load_gt_heatmaps_groundtruth() #load ground-truth heatmap
        print('np.shape(self.gt_heatmaps)_ours: ',np.shape(self.gt_heatmaps_ours))
        print('np.shape(self.gt_heatmaps)_gt_heatmaps_groundtruth: ',np.shape(self.gt_heatmaps_groundtruth))

        for step in range(self.step_total-1):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))
            print('>>>>>cal_bms_ground_cc-------self.step_total----step: ',self.step_total,step)
            # self.save_heatmap(heatmap=fcb_map,
            #                   path='/home/minglang/PAMI/fcb_err',
            #                   name=str(self.env_id))
            cc = self.calc_score(self.gt_heatmaps_ours[step],self.gt_heatmaps_groundtruth[step])
            print(step)
            self.save_step_cc(cc=cc,
                              step=step,
                              path = '/home/minglang/PAMI/cc_result/bms_and_ground/ave_cc_with_fcb/')
            if(step > 0):
              ccs += [cc]

        print("here right")
            # print(sfd)
        self.cc_averaged = sum(ccs)/len(ccs)
        self.save_ave_cc(self.cc_averaged,
                         path = '/home/minglang/PAMI/cc_result/bms_and_ground/ave_cc_with_fcb/')
        print("cc average "+str(self.cc_averaged))
        print('>>>>>>>>>>>>>>>>>>>>cal_bms_ground_cc_end<<<<<<<<<<<<<<<<<<<<<<<<<')



    def cal_ours_ground_cc(self):
        print('>>>>>>>>>>>>>>>>>>>>cal_ours_ground_cc<<<<<<<<<<<<<<<<<<<<<<<<<')
        ccs = []
        fcb_maps = []

        self.gt_heatmaps_ours = self.load_gt_heatmaps(source_path = '/home/minglang/PAMI/ff_best_heatmaps_ours_with_fcb/') #load ours heatmap
        self.gt_heatmaps_groundtruth = self.load_gt_heatmaps_groundtruth() #load ground-truth heatmap
        print('np.shape(self.gt_heatmaps)_ours: ',np.shape(self.gt_heatmaps_ours))
        print('np.shape(self.gt_heatmaps)_gt_heatmaps_groundtruth: ',np.shape(self.gt_heatmaps_groundtruth))

        for step in range(self.step_total-1):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))

            print('>>>>>cal_ours_ground_cc-------self.step_total----step: ',self.step_total,step)

            # self.save_heatmap(heatmap=fcb_map,
            #                   path='/home/minglang/PAMI/fcb_err',
            #                   name=str(self.env_id))

            cc = self.calc_score(self.gt_heatmaps_ours[step],self.gt_heatmaps_groundtruth[step])
            print(step)

            self.save_step_cc(cc=cc,
                              step=step,
                              path = '/home/minglang/PAMI/cc_result/ours_and_ground/cc_all_steps_with_fcb/')

            if(step > 0):
              ccs += [cc]

        print("here right")
            # print(sfd)
        self.cc_averaged = sum(ccs)/len(ccs)
        self.save_ave_cc(self.cc_averaged,
                         path = '/home/minglang/PAMI/cc_result/ours_and_ground/ave_cc_with_fcb/')

        print("cc average "+str(self.cc_averaged))
        print('>>>>>>>>>>>>>>>>>>>>cal_ours_ground_cc_end<<<<<<<<<<<<<<<<<<<<<<<<<')


    def save_bms_heatmaps(self,with_fcb = False):
        print('save_bms_heatmaps')
        heatmaps=[]
        source_path ='/media/minglang/Data0/PAMI/bms/output_efp/'

        for step in range(self.step_total):

            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))
            try:
                # file_name = '/home/minglang/PAMI/bms/output_efp/'+self.env_id+'_'+str(frame)+'.png'
                file_name = source_path+self.env_id+'_'+str(frame)+'.png'
                # heatmap = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255.0 # max is not 255
                heatmap = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                heatmap = cv2.resize(heatmap,(self.salmap_width, self.salmap_height))
                heatmap = heatmap * (1.0 / np.max(heatmap))
                print(">>>>>>>>>>>>>>>>>>>>>debug_here")
                print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp-1)", np.max(heatmap))
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>the every value <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",np.shape(heatmap))
                # for x in range(180):
                #     for y in range(360):
                #         if heatmap[x][y] > 0:
                #             print("heatmap:",x,y,heatmap[x][y])

                # print('np.max(heatmap): ',np.max(heatmap))
                # print(s)
                if with_fcb is False:
                    dest_path = '/home/minglang/PAMI/bms/bms_step/'
                    self.save_heatmap(heatmap=heatmap,
                                # path='bms_heatmap_cb',
                                  path = dest_path,
                                  name=str(step))
                else:
                    temp = heatmap
                    print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp0),np.max(self.fcb_map)", np.max(temp), np.max(self.fcb_map))
                    temp = np.multiply(temp,self.fcb_map)
                    print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp1)",np.max(temp))
                    temp = temp * (1.0 /np.max(temp))
                    print(">>>>>>>>>>>>>>>>>>>>>.np.max(temp2)",np.max(temp))
                    # print(myx)
                    dest_path = '/home/minglang/PAMI/bms/bms_step_with_fcb_'+str(self.fcb_sigma) + '/'
                    self.save_heatmap(heatmap=temp,
                                # path='bms_heatmap_cb',
                                  path = dest_path,
                                  name=str(step))

                # heatmaps += [heatmap]
                # print(np.shape(heatmaps))
            except Exception,e:
                print Exception,":",e
                continue
        print(s)

    def fixation2salmap_fcb(self,fixation, mapwidth, mapheight):
        my_sigma_in_degree = (11.75+13.78)/2
        fixation_total = np.shape(fixation)[0]
        x_degree_per_pixel = 360.0 / mapwidth
        y_degree_per_pixel = 180.0 / mapheight
        salmap = np.zeros((mapwidth, mapheight))
        for x in range(mapwidth):
            for y in range(mapheight):
                cur_lon = x * x_degree_per_pixel - 180.0
                cur_lat = y * y_degree_per_pixel - 90.0
                for fixation_count in range(fixation_total):
                    cur_fixation_lon = fixation[fixation_count][0]
                    cur_fixation_lat = fixation[fixation_count][1]
                    # print("cur_fixation_lon: ",cur_fixation_lon)
                    # print("cur_fixation_lat: ",cur_fixation_lat)
                    # distance_to_cur_fixation = ((cur_lon-cur_lat)^2+(cur_fixation_lon-cur_fixation_lat)^2)^0.5
                    distance_to_cur_fixation = ((cur_lon-cur_fixation_lon)**2+(cur_lat-cur_fixation_lat)**2)**0.5
                    # distance_to_cur_fixation = math.sqrt(int((cur_lon-cur_lat)^2+(cur_fixation_lon-cur_fixation_lat)^2))
                    distance_to_cur_fixation_in_degree = distance_to_cur_fixation
                    sal = math.exp(-1.0 / 2.0 * (distance_to_cur_fixation_in_degree**2) / (my_sigma_in_degree**2))
                    salmap[x, y] += sal
        salmap = salmap * (1.0 / np.amax(salmap))
        salmap = np.transpose(salmap)
        return salmap

    def fixation2salmap_fcb_2dim(self,fixation, mapwidth, mapheight,sigma = 6):
        # my_sigma_in_lon = 11.75 * 0.5
        # my_sigma_in_lat = 13.78 * 0.5
        my_sigma_in_lon = sigma
        my_sigma_in_lat = sigma
        fixation_total = np.shape(fixation)[0]
        x_degree_per_pixel = 360.0 / mapwidth
        y_degree_per_pixel = 180.0 / mapheight
        salmap = np.zeros((mapwidth, mapheight))
        for x in range(mapwidth):
            for y in range(mapheight):
                cur_lon = x * x_degree_per_pixel - 180.0
                cur_lat = y * y_degree_per_pixel - 90.0
                for fixation_count in range(fixation_total):
                    cur_fixation_lon = fixation[fixation_count][0]
                    cur_fixation_lat = fixation[fixation_count][1]
                    # print("cur_fixation_lon: ",cur_fixation_lon)
                    # print("cur_fixation_lat: ",cur_fixation_lat)
                    # distance_to_cur_fixation = ((cur_lon-cur_lat)^2+(cur_fixation_lon-cur_fixation_lat)^2)^0.5
                    distance_to_cur_fixation = ((cur_lon-cur_fixation_lon)**2+(cur_lat-cur_fixation_lat)**2)**0.5
                    # distance_to_cur_fixation = math.sqrt(int((cur_lon-cur_lat)^2+(cur_fixation_lon-cur_fixation_lat)^2))
                    distance_to_cur_fixation_in_degree = distance_to_cur_fixation
                    # sal = 1.0 / (2.0*math.pi*my_sigma_in_lon*my_sigma_in_lat)*math.exp(-1.0 / (2.0) * (cur_lon**2/(my_sigma_in_lon**2)+cur_lat**2/(my_sigma_in_lat**2)))
                    sal = math.exp(-1.0 / (2.0) * (cur_lon**2/(my_sigma_in_lon**2)+cur_lat**2/(my_sigma_in_lat**2)))
                    salmap[x, y] += sal
        salmap = salmap * (1.0 / np.amax(salmap))
        salmap = np.transpose(salmap)
        return salmap

    def save_ave_cc(self,ave_cc,path):
        from config import data_processor_id
        print("cc average "+str(self.cc_averaged))
        path1 = path+'ave_cc.txt'
        # if os.path.exists(path1) is True:
        #     os.remove(path1)
        if os.path.exists(path) is False:
            os.mkdir(path)
        f = open(path + self.save_date + '_' + str(self.fcb_sigma) + '_' + str(self.w_prediction) + '.txt','a')
        print_string = ''
        print_string += str(self.env_id) + '\t'
        print_string += str(ave_cc)
        print_string += '\n'
        f.write(print_string)
        f.close()

    def save_step_cc(self,cc,step,path):
        from config import data_processor_id
        print("cc for step "+str(step)+" is "+str(cc))
        path1 = path+str(self.env_id)+'_cc_on_step.txt'
        if os.path.exists(path1) is True:
            os.remove(path1)
        if os.path.exists(path) is False:
            os.mkdir(path)
        f = open(path+str(self.env_id)+'_cc_on_step_0824'+'_' +str(data_processor_id) + str(self.fcb_sigma) + '.txt','a')
        print_string = '\t'
        print_string += 'step' + '\t'
        print_string += str(step) + '\t'
        print_string += 'cc' + '\t'
        print_string += str(cc) + '\t'
        print_string += '\n'
        f.write(print_string)
        f.close()

    def calc_score(self,gtsAnn, resAnn):
        """
        Computer CC score. A simple implementation
        :param gtsAnn : ground-truth fixation map
        :param resAnn : predicted saliency map
        :return score: int : score
        """
        fixationMap = gtsAnn - np.mean(gtsAnn)
        salMap = resAnn - np.mean(resAnn)
        if np.max(fixationMap) > 0 and np.max(salMap) > 0:
            if (np.std(fixationMap) != 0 and np.std(salMap) != 0):
                fixationMap = fixationMap / np.std(fixationMap)
                salMap = salMap / np.std(salMap)

        cc_value = np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

        return cc_value

    def load_gt_heatmaps(self,source_path):
        heatmaps = []
        for step in range(self.step_total):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))

            try:
                # file_name = '/home/minglang/PAMI/gt_heatmap_sp_sigma_half_fov/'+self.env_id+'_'+str(step)+'.jpg'
                #for our's hmap
                # file_name = '/home/minglang/PAMI/ff_best_heatmaps_ours/'+str(self.env_id)+'/'+str(step)+'.jpg'
                file_name = source_path + self.env_id+'_'+str(step)+'.jpg'
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>file_name  ", file_name)
                print(">>>>>>>>>>>>>>>>>def load_gt_heatmaps: self.env_id, step: ",self.env_id,step)
                temp = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                print(">>>>>>>>>>>>>>>>>>>>np.shape(temp),np.max(temp)", np.shape(temp),np.max(temp))
                temp = cv2.resize(temp,(self.salmap_width, self.salmap_height))
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                temp = temp / 255.0
                heatmaps += [temp]
                print(np.shape(heatmaps))
            except Exception,e:
                print Exception,":",e
                continue

        return heatmaps

    def load_gt_heatmaps_groundtruth(self, groundtruth_path):
        '''the nss_cc for remind you choose cc_heatmap or nss_heatmap '''
        heatmaps = []
        for step in range(self.step_total-1):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))
            try:
                file_name = groundtruth_path +self.env_id+'_'+str(step)+'.jpg'
                temp = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                temp0 = cv2.resize(temp,(self.salmap_width, self.salmap_height))
                temp0 = temp0 / 255.0
                print(">>>>>>>>>>>>>>>>>>>>>>>>>def load_gt_heatmaps_groundtruth: np.shape(temp0),step,np.max(temp0)",np.shape(temp0),step,np.max(temp0))
                heatmaps += [temp]
            except Exception,e:
                print Exception,":",e
                continue

        return heatmaps

    def load_get_all_txt_groundtruth_nss(self,groundtruth_path):
        heatmaps = []
        for step in range(self.step_total):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))

            try:
                file_name = groundtruth_path +self.env_id+'_'+str(step)+'.txt'
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> enter the read_groundtruth_txt_for_nss >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>..")
                temp = self.read_groundtruth_txt_for_nss(path = file_name)
                print(">>>>>>>>>>>>>>>>>>>>>>>>>def load_get_all_txt_groundtruth_nss: step,np.shape(temp),np.max(temp)",step,np.shape(temp),np.max(temp))
                heatmaps += [temp]
            except Exception,e:
                print Exception,":",e
                continue

        return heatmaps


    def save_gt_heatmaps(self):
        print('save_gt_heatmaps')

        '''for fixation'''
        # sigma = 51.0 / (math.sqrt(-2.0*math.log(0.5)))
        sigma = 51.0 / (math.sqrt(-2.0*math.log(0.5)))*0.5#cc is large .chose half of sigma
        sigma = 7
        groundtruth_heatmaps=[]
        for step in range(self.step_total):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))
            try:
                groundtruth_fixation = np.zeros((self.subjects_total, 2))
                for subject in range(self.subjects_total):
                    # print("self.subjects_total: ",self.subjects_total)
                    # print(s_qiao)
                    groundtruth_fixation[subject, 0] = self.subjects[subject].data_frame[data].p[0]
                    groundtruth_fixation[subject, 1] = self.subjects[subject].data_frame[data].p[1]
                groundtruth_heatmap = self.fixation2salmap_sp_my_sigma(groundtruth_fixation, self.salmap_width, self.salmap_height, my_sigma = sigma)
                self.save_heatmap(heatmap=groundtruth_heatmap,
                                #   path='/home/minglang/PAMI/test_file/ground_truth_hmap/',
                                  path='/home/minglang/PAMI/test_file/ground_truth_hmap_all76/',
                                  name=str(step))
                groundtruth_heatmaps += [groundtruth_heatmap]
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>step/step_total: ',str(step)+'/'+str(self.step_total))
            except Exception,e:
                print Exception,":",e
                continue
        print(s)

    def save_gt_groundtruth_heatmaps_for_nss(self,Multi_255 = True,save_path = '/home/minglang/PAMI/test_file/ground_truth_hmap_for_nss_with_N/'):
        groundtruth_heatmaps=[]
        for step in range(self.step_total):
            data = int(round((step)*self.data_per_step))
            frame = int(round((step)*self.frame_per_step))
            try:
                groundtruth_fixation = np.zeros((self.subjects_total, 2))
                for subject in range(self.subjects_total):
                    # print("self.subjects_total: ",self.subjects_total) # = 58
                    # tow single value
                    groundtruth_fixation[subject, 0] = self.subjects[subject].data_frame[data].p[0]
                    groundtruth_fixation[subject, 1] = self.subjects[subject].data_frame[data].p[1]
                "fix one step's data of 56 subjects"
                groundtruth_heatmap = self.fixation2salmap_for_nss(groundtruth_fixation, self.salmap_width, self.salmap_height,normal = False)
                # this function X255 to the heatmap depend on the paramiter Multi255
                # debug01
                # self.save_heatmap_for_nss(
                #                   heatmap = groundtruth_heatmap,
                #                   path = save_path,
                #                   name = str(step),
                #                   Multi_255 = Multi_255)
                self.save_groundtruth_txt_for_nss(
                                                value = groundtruth_heatmap,
                                                path = save_path,
                                                name = str(step)
                                                )
                print("self.save_groundtruth_txt_for_nss: step,np.max(groundtruth_heatmap) ", step, np.max(groundtruth_heatmap))
            except Exception,e:
                print Exception,":",e
                continue
        print(s)

    def fixation2salmap_for_nss(self,fixation, mapwidth, mapheight,normal = False):
        fixation_total = np.shape(fixation)[0] # fixation_total) = 58
        x_degree_per_pixel = 360.0 / mapwidth
        y_degree_per_pixel = 180.0 / mapheight
        salmap = np.zeros((mapwidth, mapheight))

        for x in range(mapwidth):
            for y in range(mapheight):
                cur_lon = x * x_degree_per_pixel - 180.0
                cur_lat = y * y_degree_per_pixel - 90.0
                for fixation_count in range(fixation_total):
                    cur_fixation_lon = round(fixation[fixation_count][0])
                    cur_fixation_lat = round(fixation[fixation_count][1])
                    if cur_lon == cur_fixation_lon and cur_lat == cur_fixation_lat:
                        salmap[x, y] = salmap[x, y] + 1
                        if salmap[x, y] > 1:
                            print(">>>>>>>>>>>>>>>>>def fixation2salmap_for_nss: x,y,salmap[x, y]: ",x,y,salmap[x, y])
                    "for debug"
                    # if y >160:
                    #     salmap[x, y] = 255
        # if normal is True:
        #     salmap = salmap * ( 1.0 / np.amax(salmap) )
        salmap = np.transpose(salmap)# samle.T,change to (180,360)
        return salmap

    "save groundtruth data as txt"
    def save_groundtruth_txt_for_nss(self,value,path,name):
        f = open(path+self.env_id+'_'+name+'.txt','w') # 'w' mode will clear the formore data
        for x in range(self.salmap_height):
            for y in range(self.salmap_width):
                # print_string = ''
                # print_string += 'step' + '\t'
                print_string = str(x) + '\t'+str(y) +'\t'
                # print_string += 'ave_cc' + '\t'
                print_string += str(value[x][y]) + '\t'
                print_string += '\n'
                f.write(print_string)
        f.close()

    def read_groundtruth_txt_for_nss(self,path):
        x = []
        y = []
        value = []

        f = open(path,"r")
        lines = f.readlines()#读取全部内容
        # slip the value
        for line in lines:
            x0, y0,value0 =  line.split()
            x.append(x0)
            y.append(y0)
            value.append(value0)
        value = np.array(value)
        value = np.resize(value,(self.salmap_height, self.salmap_width))
        value = value.astype('float')

        return value

    def fixation2salmap_sp_my_sigma(self,fixation, mapwidth, mapheight, my_sigma = (11.75+13.78)/2):
        fixation_total = np.shape(fixation)[0]
        x_degree_per_pixel = 360.0 / mapwidth
        y_degree_per_pixel = 180.0 / mapheight
        salmap = np.zeros((mapwidth, mapheight))
        for x in range(mapwidth):
            for y in range(mapheight):
                cur_lon = x * x_degree_per_pixel - 180.0
                cur_lat = y * y_degree_per_pixel - 90.0
                for fixation_count in range(fixation_total):
                    cur_fixation_lon = fixation[fixation_count][0]
                    cur_fixation_lat = fixation[fixation_count][1]
                    distance_to_cur_fixation = haversine(lon1=cur_lon,
                                                         lat1=cur_lat,
                                                         lon2=cur_fixation_lon,
                                                         lat2=cur_fixation_lat)
                    distance_to_cur_fixation = distance_to_cur_fixation / math.pi * 180.0
                    sal = math.exp(-1.0 / 2.0 * (distance_to_cur_fixation**2) / (my_sigma**2))
                    salmap[x, y] += sal
        salmap = salmap * ( 1.0 / np.amax(salmap) )
        salmap = np.transpose(salmap)
        return salmap



    def log_thread_config(self):

        from config import if_log_scan_path
        self.if_log_scan_path = if_log_scan_path

        from config import if_log_cc
        self.if_log_cc = if_log_cc

        if self.if_log_cc:

            if self.mode is 'off_line':
                '''cc record'''
                self.agent_result_saver = []
                self.agent_result_stack = []


                if self.mode is 'off_line': # yuhangsong here
                    from config import relative_predicted_fixation_num
                    self.predicted_fixtions_num = int(self.subjects_total * relative_predicted_fixation_num)
                    print('predicted_fixtions_num is '+str(self.predicted_fixtions_num))
                    from config import relative_log_cc_interval
                    self.if_log_cc_interval = int(self.predicted_fixtions_num * relative_log_cc_interval)
                    print('log_cc_interval is '+str(self.if_log_cc_interval))

    def reset(self):

        '''reset cur_step and cur_data'''
        self.cur_step = 0
        self.cur_data = 0

        self.reward_dic_on_cur_episode = []

        '''episode add'''
        self.episode +=1

        '''reset cur_frame'''
        self.cur_frame = 0

        '''reset last action'''
        self.last_action = None

        '''reset cur_lon and cur_lat to one of the subjects start point'''
        subject_dic_code = []
        for i in range(self.subjects_total):
            subject_dic_code += [i]
        if self.mode is 'off_line':
            subject_code = np.random.choice(a=subject_dic_code)
        elif self.mode is 'on_line':
            subject_code = 0
        self.cur_lon = self.subjects[subject_code].data_frame[0].p[0]
        self.cur_lat = self.subjects[subject_code].data_frame[0].p[1]

        '''reset view_mover'''
        self.view_mover.init_position(Latitude=self.cur_lat,
                                      Longitude=self.cur_lon)

        '''set observation_now to the first frame'''
        self.get_observation()

        self.last_observation = None

        if self.log_thread:
            self.log_thread_reset()

        return self.cur_observation

    def log_thread_reset(self):

        if self.if_log_scan_path:
            plt.figure(str(self.env_id)+'_scan_path')
            plt.clf()

        if self.if_log_cc:

            if self.mode is 'off_line':

                self.agent_result_stack += [copy.deepcopy(self.agent_result_saver)]
                self.agent_result_saver = []

                if len(self.agent_result_stack) > self.predicted_fixtions_num:

                    '''if stack full, pop out the oldest data'''
                    self.agent_result_stack.pop(0)

                    if self.episode%self.if_log_cc_interval is 0:

                        print('compute cc..................')

                        ccs_on_step_i = []
                        heatmaps_on_step_i = []
                        for step_i in range(self.step_total):

                            '''generate predicted salmap'''
                            temp = np.asarray(self.agent_result_stack)[:,step_i]
                            temp = np.sum(temp,axis=0)
                            temp = temp / np.max(temp)
                            heatmaps_on_step_i += [copy.deepcopy(temp)]
                            from cc import calc_score
                            ccs_on_step_i += [copy.deepcopy(calc_score(self.gt_heatmaps[step_i], heatmaps_on_step_i[step_i]))]
                            print('cc on step '+str(step_i)+' is '+str(ccs_on_step_i[step_i]))

                        self.cur_cc = np.mean(np.asarray(ccs_on_step_i))
                        print('cur_cc is '+str(self.cur_cc))
                        if self.cur_cc > self.max_cc:
                            print('new max cc found: '+str(self.cur_cc)+', recording cc and heatmaps')
                            self.max_cc = self.cur_cc
                            self.heatmaps_of_max_cc = heatmaps_on_step_i

                            from config import final_log_dir
                            record_dir = final_log_dir+'ff_best_heatmaps/'+self.env_id+'/'
                            subprocess.call(["rm", "-r", record_dir])
                            subprocess.call(["mkdir", "-p", record_dir])
                            for step_i in range(self.step_total):
                                self.save_heatmap(heatmap=self.heatmaps_of_max_cc[step_i],
                                                  path=record_dir,
                                                  name=str(step_i))

    def step(self, action, v):

        '''these will be returned, but not sure to updated'''
        if self.log_thread:
            self.log_thread_step()

        '''varible for record state is stored, for they will be updated'''
        self.last_step = self.cur_step
        self.last_data = self.cur_data
        self.last_observation = self.cur_observation
        self.last_lon = self.cur_lon
        self.last_lat = self.cur_lat
        self.last_frame = self.cur_frame

        '''update cur_step'''
        self.cur_step += 1

        '''update cur_data'''
        self.cur_data = int(round((self.cur_step)*self.data_per_step))
        if(self.cur_data>=self.data_total):
            update_data_success = False
        else:
            update_data_success = True

        '''update cur_frame'''
        self.cur_frame = int(round((self.cur_step)*self.frame_per_step))
        if(self.cur_frame>=(self.frame_total-self.frame_bug_offset)):
            update_frame_success = False
        else:
            update_frame_success = True

        v_lable = 0.0

        '''if any of update frame or update data is failed'''
        if(update_frame_success==False)or(update_data_success==False):

            '''terminating'''
            self.reset()
            reward = 0.0
            done = True
            if self.if_learning_v:
                v_lable = 0.0

        else:

            '''get reward and v from last state'''
            last_prob, distance_per_data = get_prob(lon=self.last_lon,
                                                    lat=self.last_lat,
                                                    theta=action * 45.0,
                                                    subjects=self.subjects,
                                                    subjects_total=self.subjects_total,
                                                    cur_data=self.last_data)

            '''rescale'''
            distance_per_step = distance_per_data * self.data_per_step

            '''convert v to degree'''
            degree_per_step = distance_per_step / math.pi * 180.0

            if (self.mode is 'on_line') and (self.predicting is True):
                '''online and predicting, lon and lat is updated as subjects' ground-truth'''
                '''other procedure may not used by the agent, but still implemented to keep the interface unified'''
                print('predicting run')
                self.cur_lon = self.subjects[0].data_frame[self.cur_data].p[0]
                self.cur_lat = self.subjects[0].data_frame[self.cur_data].p[1]
            else:
                '''move view, update cur_lon and cur_lat, the standard procedure of rl'''
                if self.if_learning_v:
                    self.cur_lon, self.cur_lat = self.view_mover.move_view(direction=action * 45.0,degree_per_step=v)
                    v_lable = degree_per_step
                else:
                    self.cur_lon, self.cur_lat = self.view_mover.move_view(direction=action * 45.0,degree_per_step=degree_per_step)

            '''update observation_now'''
            self.get_observation()

            '''produce reward'''
            if self.reward_estimator is 'trustworthy_transfer':
                reward = last_prob
            elif self.reward_estimator is 'cc':
                cur_heatmap = fixation2salmap(fixation=[[self.cur_lon, self.cur_lat]],
                                              mapwidth=self.heatmap_width,
                                              mapheight=self.heatmap_height)
                from cc import calc_score
                reward = calc_score(self.gt_heatmaps[self.cur_step], cur_heatmap)

            '''smooth reward'''
            if self.last_action is not None:

                '''if we have last_action'''

                '''compute smooth reward'''
                action_difference = abs(action-self.last_action)
                from config import direction_num
                if action_difference > (direction_num/2):
                    action_difference -= (direction_num/2)
                from config import reward_smooth_discount_to
                reward *= (1.0-(action_difference*(1.0-reward_smooth_discount_to)/(direction_num/2)))

            '''record'''
            self.last_action = action
            self.reward_dic_on_cur_episode += [reward]

            '''normally, we donot judge done when we in this'''
            done = False

            if self.mode is 'on_line':

                if self.predicting is False:

                    '''if is training'''
                    if self.cur_step > self.cur_training_step:

                        '''if step is out of training range'''

                        if np.mean(self.reward_dic_on_cur_episode) > self.train_to_reward:

                            '''if reward is trained to a acceptable range'''

                            '''summary'''
                            summary = tf.Summary()
                            summary.value.add(tag=self.env_id+'on_cur_train/number_of_episodes',
                                              simple_value=float(len(self.sum_reward_dic_on_cur_train)))
                            summary.value.add(tag=self.env_id+'on_cur_train/average_@sum_reward_per_step@',
                                              simple_value=float(np.mean(self.sum_reward_dic_on_cur_train)))
                            summary.value.add(tag=self.env_id+'on_cur_train/average_@average_reward_per_step@',
                                              simple_value=float(np.mean(self.sum_reward_dic_on_cur_train)))
                            self.summary_writer.add_summary(summary, self.cur_training_step)
                            self.summary_writer.flush()

                            '''reset'''
                            self.sum_reward_dic_on_cur_train = []
                            self.average_reward_dic_on_cur_train = []

                            '''tell outside: we are going to predict on the next run'''
                            self.predicting = True

                            '''update'''
                            self.cur_training_step += 1
                            self.cur_predicting_step += 1

                            if self.cur_predicting_step >= self.step_total:

                                '''on line terminating'''
                                print('on line run meet end, terminating..')
                                import sys
                                sys.exit(0)

                        else:

                            '''is reward has not been trained to a acceptable range'''

                            '''record this episode run before reset to start point'''
                            self.average_reward_dic_on_cur_train += [np.mean(self.reward_dic_on_cur_episode)]
                            self.sum_reward_dic_on_cur_train += [np.sum(self.reward_dic_on_cur_episode)]

                            '''tell out side: we are not going to predict'''
                            self.predicting = False

                        '''reset anyway since cur_step beyond cur_training_step'''
                        self.reset()
                        done = True

                else:

                    '''if is predicting'''

                    if self.cur_step > self.cur_predicting_step:

                        '''if cur_step run beyond cur_predicting_step, means already make a prediction on this step'''

                        '''summary'''
                        summary = tf.Summary()
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@sum_reward_per_step@',
                                          simple_value=float(np.sum(self.reward_dic_on_cur_episode)))
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@average_reward_per_step@',
                                          simple_value=float(np.mean(self.reward_dic_on_cur_episode)))
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@reward_for_predicting_step@',
                                          simple_value=float(self.reward_dic_on_cur_episode[-1]))
                        self.summary_writer.add_summary(summary, self.cur_predicting_step)
                        self.summary_writer.flush()

                        '''tell out side: we are not going to predict'''
                        self.predicting = False

                        '''reset'''
                        self.reset()
                        done = True

        if self.mode is 'off_line':
            return self.cur_observation, reward, done, self.cur_cc, self.max_cc, v_lable
        elif self.mode is 'on_line':
            return self.cur_observation, reward, done, self.cur_cc, self.max_cc, v_lable, self.predicting

    def log_thread_step(self):
        '''log_scan_path'''
        if self.if_log_scan_path:
            plt.figure(str(self.env_id)+'_scan_path')
            plt.scatter(self.cur_lon, self.cur_lat, c='r')
            plt.scatter(-180, -90)
            plt.scatter(-180, 90)
            plt.scatter(180, -90)
            plt.scatter(180, 90)
            plt.pause(0.00001)

        if self.if_log_cc:
            if self.mode is 'off_line':
                self.agent_result_saver += [copy.deepcopy(fixation2salmap(fixation=[[self.cur_lon,self.cur_lon]],
                                                                          mapwidth=self.heatmap_width,
                                                                          mapheight=self.heatmap_height))]
            elif self.mode is 'on_line':
                print('not implement')
                import sys
                sys.exit(0)
    def load_heatmaps(self, name):

        heatmaps = []
        for step in range(self.step_total):

            try:
                file_name = '../../'+self.data_base+'/'+name+'/'+self.env_id+'_'+str(step)+'.jpg'
                temp = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                temp = cv2.resize(temp,(self.heatmap_width, self.heatmap_height))
                temp = temp / 255.0
                heatmaps += [temp]
            except Exception,e:
                print Exception,":",e
                continue

        print('load heatmaps: '+name+' done, size: '+str(np.shape(heatmaps)))

        return heatmaps

    def save_heatmap(self,heatmap,path,name):
        if os.path.exists(path) is False:
            os.mkdir(path)
        heatmap = heatmap * 255.0
        # cv2.imwrite(path+'/'+name+'.jpg',heatmap)
        # cv2.imwrite(path+'/'+self.env_id+'_'+name+'.jpg',heatmap)
        cv2.imwrite(path+self.env_id+'_'+name+'.jpg',heatmap)
        # cv2.imwrite(path+'/'+'Let\'sNotBeAloneTonight'+'_'+name+'.jpg',heatmap)

    def cal_ave_cc_all_videos(self,source_path,save_path):
        pass
        for weight_i in range(0,100,1):
            "weight parametr"
            self.w_prediction = weight_i * 1.0 / 100
            self.w_fcb = 1 - self.w_prediction # 0.1374

            'cal_ave_cc_all_videos loop start'
            path_source = source_path + self.save_date + '_' + str(self.fcb_sigma) + '_' + str(self.w_prediction) + '.txt'
            path_save = save_path + self.save_date + '.txt'
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.here_1')
            f = open(path_source, "r")
            lines = f.readlines() #read all lines
            cc = []
            for line in lines:
                line = line.split()
                print(line[1])
                cc.append(float(line[1]))

            ave_cc_all_videos = np.mean(cc)
            print('ave_cc_all_videos: ',ave_cc_all_videos)

            if os.path.exists(save_path) is False:
                os.mkdir(save_path)
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.here_2')
            f_save = open(path_save,'a')
            save_string = str(ave_cc_all_videos) + '\t' + str(self.fcb_sigma) + '\t' + str(self.w_prediction)+'\n'
            f_save.write(save_string)
            f_save.close()
            'cal_ave_cc_all_videos loop end'
        print('cal_ave_cc_all_videos end')

    def save_heatmap_for_nss(self,heatmap,path,name,Multi_255 = False):
       if os.path.exists(path) is False:
           os.mkdir(path)
       if Multi_255 is True:
           heatmap = heatmap * 255.0
       cv2.imwrite(path+self.env_id+'_'+name+'.jpg',heatmap)
    #    print(">>>>>>>>>>>>>>>>>>>>>>>>def save_heatmap_for_nss: np.max(heatmap),np.shape(heatmap)", np.max(heatmap),np.shape(heatmap))
       Image_read0 = cv2.imread(path+self.env_id+'_'+name+'.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
       Image_read1 = cv2.resize(Image_read0,(360, 180))
    #    print(">>>>>>>>>>>>>>>>>>>>>>>>def save_heatmap_for_nss: np.max(Image_read0),np.shape(Image_read0)", np.max(Image_read0),np.shape(Image_read0))
       # for debug1
       self.test_save(heatmap,name)

    def test_save(self,heatmap,name):
        "test the read image debug"
        Image = np.zeros((360,180))
        for x in range(360):
            for y in range(60):
                # print("x,y",x,y) # 359,179
                Image[x][y] = 0

        for x in range(360):
            for y in range(60,180):
            #   print("x1,y1",x,y) # 359,179
              Image[x][y] = 0

        for x in range(1):
            for y in range(170,180):
            #   print("x1,y1",x,y) # 359,179
              Image[x][y] = 8.0

        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> def test_save(self,heatmap,name): np.shape(Image),np.shape(heatmap)",np.shape(Image),np.shape(heatmap))

        path = '/home/minglang/test_code/'
        # name = 't1'
        # env_id = 'A1'
        # Image = np.transpose(Image)
        heatmap1 = Image
        i = 0
        # heatmap1 = np.transpose(heatmap) # samle.T
        # for x in range(360):
        #     for y  in range(180):
        #         if heatmap1[x][y] > 0:
        #             i += 1
        #             print("i,x,y,heatmap[x][y]: ",i,x,y,heatmap1[x][y])

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> def test_save(self,heatmap,name): np.shape(Image),np.shape(heatmap)",np.shape(Image),np.shape(heatmap))
        file_name = path+self.env_id+'_'+name+'.jpg'
        heatmap_width = 360
        heatmap_height = 180

        cv2.imwrite(file_name, heatmap1)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. def test_save(self,heatmap,name):np.shape(heatmap1),np.max(heatmap1) ", np.shape(Image),np.max(heatmap1))
        Image_read0 = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # Image_read1 = cv2.resize(Image_read0,(heatmap_width, heatmap_height))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  def test_save(self,heatmap):np.shape(Image_read0),np.max(Image_read0): ",np.shape(Image_read0),np.max(Image_read0))

        for x in range(360):
            for y in range(180):
                if Image_read0[x][y] > 0:
                    # i += Image_read0[x][y]
                    i = i + 1
                    print("i,x,y,Image_read0[x][y]: ",i,x,y,Image_read0[x][y])
