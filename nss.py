#!/usr/bin/env python
#
# File Name : nss.py
#
# Description : Computes NSS metric #

# Author : Ming Jiang

import numpy as np
import scipy.ndimage

def calc_score(self, gtsAnn, resAnn):
    """
    Computer NSS score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : NSS score
    """

    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)
    return np.mean([ salMap[y-1][x-1] for y,x in gtsAnn ])

class NSS():
    '''
    Class for computing NSS score for saliency maps

    '''
    def __init__(self,saliconRes):
        self.saliconRes = saliconRes
        self.imgs = self.saliconRes.imgs

    # here, minglang 5, you have to refer to saliconeval/cc/cc.py in this project
    # and the calc_score in support_lib.py in our project, to see how I implement the function from salicon to our project,
    # and try to implement a similar function to calculate nss in our project.
    # name the function to calc_score_nss and keep the interface exactly same as the original calc_score function in support_lib.py in our project
    def calc_score(self, gtsAnn, resAnn):
        """
        Computer NSS score.
        :param gtsAnn : ground-truth annotations
        :param resAnn : predicted saliency map
        :return score: int : NSS score
        """

        salMap = resAnn - np.mean(resAnn)
        if np.max(salMap) > 0:
            salMap = salMap / np.std(salMap)
        return np.mean([ salMap[y-1][x-1] for y,x in gtsAnn ])

    def compute_score(self, gts, res):
        """
        Computes NSS score for a given set of predictions and fixations
        :param gts : dict : fixation points with "image name" key and list of points as values
        :param res : dict : saliency map predictions with "image name" key and ndarray as values
        :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = res.keys()
        score = []
        for id in imgIds:
            img = self.imgs[id]
            fixations  = gts[id]
            height,width = (img['height'],img['width'])
            salMap = self.saliconRes.decodeImage(res[id])
            mapheight,mapwidth = np.shape(salMap)
            salMap = scipy.ndimage.zoom(salMap, (float(height)/mapheight, float(width)/mapwidth), order=3)
            score.append(self.calc_score(fixations,salMap))
        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "NSS"



if __name__=="__main__":
    nss = NSS()
