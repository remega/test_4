import random
import math
# count = -1
# val = []
# for i in range(0,58):
#     count += 1
#     val.append(count)
# val_A  = random.sample(val, 29)
# for i in val_A:
#     val.remove(i)
# val_B = val
#
# print(val_A,val_B)

# sigma = 51.0 / (math.sqrt(-2.0*math.log(0.5))) * 0.126 # 5.45 ----> 55.55
sigma = 51.0 / (math.sqrt(-2.0*math.log(0.5))) * 0.0693   # 3.0 ----> 33.33

status = '_' + '3_3.3' + '/'

path_A = '/media/minglang/Data0/PAMI/Haochen/HA'+status

print(sigma)


            status = '_' + '3_3.3' + '/'
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> status',status)
            # self.save_gt_heatmaps_for_haoceng(
            #                                   path_A = '/media/minglang/Data0/PAMI/Haochen/HA' + status,
            #                                   path_B = '/media/minglang/Data0/PAMI/Haochen/HB' + status
            #                                  )
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> cal cc begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ground_src_path', '/media/minglang/Data0/PAMI/Haochen/HA' + status)
            self.cal_cc(
                    ground_src_path = '/media/minglang/Data0/PAMI/Haochen/HA' + status,
                    prediction_src_path = '/media/minglang/Data0/PAMI/Haochen/HB' + status, # '/media/minglang/Data0/PAMI/Haochen/HB/', '/media/minglang/Data0/PAMI/fcb/'
                    dst_all_cc_path ='/media/minglang/Data0/PAMI/Haochen/CC_DEV/step_all' + status,
                    dst_ave_cc_path ='/media/minglang/Data0/PAMI/Haochen/CC_DEV/ave/' + status)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ground_src_path', '/media/minglang/Data0/PAMI/Haochen/HA' + status)
            # print('>>>>>>>>>>>>>>>>>>>>end<<<<<<<<<<<<<<<<<<<<<<<<<')
