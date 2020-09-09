from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import dataloader as datasets
import torch
import cv2
from tracker.multitrackercam4 import JDETracker
from tracking_utils.timer import Timer
from tracking_utils import visualization as vis
from PIL import Image
from utils.bb_polygon import load_zone_anno
import numpy as np
import copy
def eval_seq(opt, dataloader,polygon, paths, data_type, result_filename, frame_dir=None,save_dir=None,bbox_dir=None, show_image=True, frame_rate=30):
    count=0
    if save_dir:
        mkdir_if_missing(save_dir)
    if bbox_dir:
        mkdir_if_missing(bbox_dir)
    if frame_dir:
        mkdir_if_missing(frame_dir)
    tracker = JDETracker(opt,polygon, paths, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 1
    
    f = open(opt.input_video.split('/')[-1][:-4] + '.txt', 'w' )

    for path, img, img0 in dataloader:
        img0_clone=copy.copy(img0)
        if frame_id % 1 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
           

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0) if opt.gpus[0]>=0 else torch.from_numpy(img).cpu().unsqueeze(0)
        online_targets,detection_boxes,out_of_polygon_tracklet = tracker.update(blob, img0)
        if len(out_of_polygon_tracklet)>0:
            for track in np.asarray(out_of_polygon_tracklet)[:,2]:
                if track in ['person','bicycle', 'motorcycle']:
                    count+=1
            print('count : '+str(count))
        online_tlwhs = []
        online_ids = []
        
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] > opt.min_box_area :
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
           
        #bbox detection plot        
        box_tlbrs=[]
        box_scores=[]
        box_classes=[]
        box_occlusions=[]
        img_bbox=img0.copy()
        for box in detection_boxes:
            tlbr=box.tlbr
            tlwh=box.tlwh
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area:
                box_tlbrs.append(tlbr)
                box_scores.append(box.score)
                box_classes.append(box.infer_type())
                box_occlusions.append('occ' if box.occlusion_status==True else 'non_occ')

        timer.toc()
        # save results
        for track in out_of_polygon_tracklet:
            frame_idx,id,classes,movement=track
            results.append((opt.input_video.split('/')[-1][:-4],frame_idx , classes, movement))
            f.write(','.join([opt.input_video.split('/')[-1][:-4], str(frame_idx), str(classes), str(movement)])+ '\n')
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time,out_track=out_of_polygon_tracklet)
            bbox_im=vis.plot_detections(img_bbox,box_tlbrs,scores=box_scores,box_occlusion=None,btypes=box_classes)
        if show_image:
            cv2.polylines(online_im,[np.asarray(polygon)],True,(0,255,255))
            cv2.polylines(bbox_im,[np.asarray(polygon)],True,(0,255,255))
            cv2.polylines(img0_clone,[np.asarray(polygon)],True,(0,255,255))
            cv2.imshow('online_im', online_im)
            cv2.imshow('bbox_im',bbox_im)
        if save_dir is not None:
            cv2.polylines(online_im,[np.asarray(polygon)],True,(0,255,255))
            cv2.polylines(bbox_im,[np.asarray(polygon)],True,(0,255,255))
            cv2.polylines(img0_clone,[np.asarray(polygon)],True,(0,255,255))
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
            cv2.imwrite(os.path.join(bbox_dir, '{:05d}.jpg'.format(frame_id)), bbox_im)
            cv2.imwrite(os.path.join(frame_dir, '{:05d}.jpg'.format(frame_id)),img0_clone)

        frame_id += 1
    # save results
    
    return frame_id, timer.average_time, timer.calls

def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    polygon, paths=load_zone_anno(opt.input_meta)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_tracking_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame_tracking')
    bbox_dir  = None if opt.output_format == 'text' else osp.join(result_root, 'bbox_detection')
    frame_dir =  None if opt.output_format == 'text' else osp.join(result_root, 'frame_dir')
    eval_seq(opt, dataloader,polygon, paths, 'mot', result_filename, frame_dir=frame_dir,save_dir=frame_tracking_dir,bbox_dir=bbox_dir, show_image=False, frame_rate=frame_rate)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
