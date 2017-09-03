import numpy as np

f = open('/home/minglang/PAMI/cc_result/ours_and_ground/ave_cc_with_fcb_fitting/ave_cc_0902_12_0.5_minglang_get_ours_groundhp_ss_cc.txt',"r")
lines = f.readlines() #read all lines
cc = []
for line in lines:
    line = line.split()
    print(line[1])
    cc.append(float(line[1]))

ave_cc_all_videos = np.mean(cc)
print('ave_cc_all_videos: ',ave_cc_all_videos)
