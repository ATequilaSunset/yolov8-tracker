"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

from PIL import Image, ImageDraw, ImageFont
import random
from collections import defaultdict


np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3, det_thresh=0.6):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_thresh = det_thresh  # ByteTrack: 检测框分数阈值
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
    
    def update(self, dets=np.empty((0, 5))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],...]
        Returns:
        similar array with last column as object ID
        """
        self.frame_count += 1
        
        # 获取预测位置
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # ==================== ByteTrack 策略 ====================
        # 1. 将检测框分为高分框和低分框
        if len(dets) > 0:
            # dets[:, 4] 是置信度分数
            high_score_indices = np.where(dets[:, 4] >= self.det_thresh)[0]
            low_score_indices = np.where(dets[:, 4] < self.det_thresh)[0]
            
            dets_high = dets[high_score_indices]
            dets_low = dets[low_score_indices]
        else:
            dets_high = np.empty((0, 5))
            dets_low = np.empty((0, 5))
        
        # 2. 第一次匹配：轨迹与高分框匹配
        matched, unmatched_dets_high, unmatched_trks = \
            associate_detections_to_trackers(dets_high, trks, self.iou_threshold)
        
        # 3. 第二次匹配：未匹配的轨迹与低分框匹配
        # 获取未匹配轨迹对应的 trks
        if len(unmatched_trks) > 0 and len(dets_low) > 0:
            trks_for_second_match = trks[unmatched_trks]
            matched_second, unmatched_dets_low, unmatched_trks_second = \
                associate_detections_to_trackers(dets_low, trks_for_second_match, self.iou_threshold)
            
            # 将第二次匹配的索引映射回原始索引
            matched_second_mapped = []
            for m in matched_second:
                det_idx = m[0]  # dets_low 中的索引
                trk_idx_in_unmatched = m[1]  # unmatched_trks 中的索引
                original_trk_idx = unmatched_trks[trk_idx_in_unmatched]  # 原始 trks 中的索引
                matched_second_mapped.append([det_idx, original_trk_idx])
            
            matched_second = np.array(matched_second_mapped) if matched_second_mapped else np.empty((0, 2))
        else:
            matched_second = np.empty((0, 2))
            unmatched_dets_low = np.arange(len(dets_low))
        
        # ==================== 更新匹配的轨迹 ====================
        # 更新第一次匹配的轨迹（高分框）
        for m in matched:
            det_idx = m[0]
            trk_idx = m[1]
            self.trackers[trk_idx].update(dets_high[det_idx, :4])  # 只传 bbox，不含 score
        
        # 更新第二次匹配的轨迹（低分框）
        for m in matched_second:
            det_idx = m[0]
            trk_idx = m[1]
            self.trackers[trk_idx].update(dets_low[det_idx, :4])
        
        # ==================== 初始化新轨迹 ====================
        # 只为未匹配的高分框创建新轨迹
        for i in unmatched_dets_high:
            trk = KalmanBoxTracker(dets_high[i, :4])
            self.trackers.append(trk)
        
        # 未匹配的低分框直接丢弃（不创建新轨迹）
        # 未匹配的低分框（matched_second 中的 unmatched_dets_low）也丢弃
        
        # ==================== 输出结果 ====================
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # 删除丢失的轨迹
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

#LZ
class TrackingVisualizer:
    """功能更完整的追踪可视化器"""
    
    def __init__(self, font_path: str = None, id_color_seed: int = 42):
        """
        Args:
            font_path: 字体文件路径，None则使用默认字体
            id_color_seed: 随机种子，确保ID颜色一致性
        """
        self.font_path = font_path
        random.seed(id_color_seed)
        
        # 预生成50种颜色
        self.id_colors = self._generate_distinct_colors(50)
    
    def _generate_distinct_colors(self, n: int) -> list:
        """生成区分度高的颜色"""
        colors = []
        for i in range(n):
            hue = (i * 137.5) % 360  # 黄金角分布
            saturation = 0.7 + random.random() * 0.3
            lightness = 0.5 + random.random() * 0.3
            r, g, b = self._hsl_to_rgb(hue, saturation, lightness)
            colors.append((int(r*255), int(g*255), int(b*255)))
        return colors
    
    @staticmethod
    def _hsl_to_rgb(h, s, l):
        """HSL转RGB"""
        c = (1 - abs(2*l - 1)) * s
        x = c * (1 - abs((h/60) % 2 - 1))
        m = l - c/2
        
        if 0 <= h < 60: r, g, b = c, x, 0
        elif 60 <= h < 120: r, g, b = x, c, 0
        elif 120 <= h < 180: r, g, b = 0, c, x
        elif 180 <= h < 240: r, g, b = 0, x, c
        elif 240 <= h < 300: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        
        return r+m, g+m, b+m

    @staticmethod
    def _draw_detection_cross(
        draw: ImageDraw.ImageDraw,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color=(255, 0, 0),
        width: int = 2,
    ) -> None:
        """在检测框中心绘制叉号标记。"""
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half_size = max(4, int(min(x2 - x1, y2 - y1) * 0.15))
        draw.line([(cx - half_size, cy - half_size), (cx + half_size, cy + half_size)], fill=color, width=width)
        draw.line([(cx - half_size, cy + half_size), (cx + half_size, cy - half_size)], fill=color, width=width)
    
    def draw_tracks(
        self,
        image: Image.Image,
        tracks: np.ndarray,
        detections: np.ndarray | None = None,
        categories: dict = None,
        font_size: int = 24,
        line_width: int = 3,
        show_trajectory: bool = False,
        trajectory_length: int = 10
    ) -> Image.Image:
        """
        绘制追踪结果
        
        Args:
            detections: 检测结果，格式为 [x1, y1, x2, y2, score]
            show_trajectory: 是否显示轨迹线
            trajectory_length: 轨迹历史长度
        """
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # 加载字体
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        if detections is not None:
            for det in detections:
                if len(det) < 4:
                    continue
                self._draw_detection_cross(draw, det[0], det[1], det[2], det[3])
        
        # 维护轨迹历史（如果需要显示轨迹）
        if show_trajectory and not hasattr(self, 'trajectory_history'):
            self.trajectory_history = defaultdict(list)
        
        for track in tracks:
            if len(track) < 5: continue
            
            x1, y1, x2, y2, track_id = track[:5]
            track_id = int(track_id)
            category = categories.get(track_id, "") if categories else ""
            
            color = self.id_colors[track_id % len(self.id_colors)]
            
            # 绘制框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            # 绘制轨迹线
            if show_trajectory:
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                self.trajectory_history[track_id].append(center)
                
                if len(self.trajectory_history[track_id]) > trajectory_length:
                    self.trajectory_history[track_id].pop(0)
                
                if len(self.trajectory_history[track_id]) > 1:
                    draw.line(
                        self.trajectory_history[track_id],
                        fill=color + (128,),  # 半透明
                        width=2
                    )
            
            # 绘制标签
            label = f"ID:{track_id}"
            if category: label += f" {category}"
            
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # 标签背景
            bg_coords = [x1, y1 - text_h - 6, x1 + text_w + 8, y1]
            if bg_coords[1] < 0:  # 防止越界
                bg_coords = [x1, y1, x1 + text_w + 8, y1 + text_h + 6]
            
            draw.rectangle(bg_coords, fill=color, outline=color)
            draw.text((bg_coords[0] + 4, bg_coords[1] + 3), label, 
                     fill=(255, 255, 255), font=font)
        
        return vis_image

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
