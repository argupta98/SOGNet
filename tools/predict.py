import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sognet.models import *
from sognet.lib.utils.timer import Timer
from sognet.config.config import config, update_config
from sognet.dataset.base_dataset import BaseDataset
from test_net import im_detect, im_post
import pycocotools.mask as mask_util
import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random
import cv2
from sognet.lib.utils.colormap import colormap
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, help='Path to pretrained model', type=str)
parser.add_argument('--img_path', required=True, help='Path to test_image', type=str)
parser.add_argument('--cfg', required=True, help='Path to configuration file', type=str)
parser.add_argument('--run_timing',  help='Path to test image', action="store_true")

class SOGPredictor(object):
    """ Wrapper class with all functionality to predict on an image and produce the relevant segmentation mask. """
    def __init__(self, model_path, run_timing=False):
        # load the model
        self.model = eval(config.symbol)().cuda()
        self.model.load_state_dict(torch.load(model_path), resume=True)

        self.data_func = BaseDataset()
        self.base_dir = "examples"

        # Number of COCO classes
        self.num_classes = 81

        # Set to eval mode
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def process_image(self, image_path):
        """ Do image processing which was used to train the model. """
        roidb = [{"image": image_path, "flipped": False}]
        im_blob, im_scales = self.data_func.get_image_blob(roidb)
        batch = {'data': im_blob,
                'im_info': np.array([[im_blob.shape[-2],
                                        im_blob.shape[-1],
                                        im_scales[0]]], np.float32)}
        processed = {}
        processed.update({'data': torch.from_numpy(self.data_func.im_list_to_blob([b['data'] for b in batch]))})
        processed.update({'im_info': np.vstack([b['im_info'] for b in batch])})
        return [(processed, None)]

    def predict(self, image_path):
        """ Load the image and run the prediction. """
        batch = self.process_image(image_path)
        im_infos = [_[0]['im_info'][0] for _ in batch]
        output = im_detect(self.model(*batch), batch, im_infos)
        boxes = [[] for _ in range(self.num_classes)]
        masks = [[] for _ in range(self.num_classes)]
        for score, box, mask, cls_idx, im_info in zip(output['scores'], output['boxes'], output['masks'], output['cls_inds'], im_infos):
            im_post(boxes, masks, score, box, mask, cls_idx, self.num_classes, np.round(im_info[:2] / im_info[2]).astype(np.int32))
        self.visualize(boxes, masks, image_path)    
    
    def visualize(self, boxes, masks, image_path):
        """
        visualize all detections in one image
        :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
        :param class_names: list of names in imdb
        """

        os.makedirs(self.base_dir, exist_ok=True)

        color_list = colormap(rgb=True) / 255
        mask_color_id = 0

        im = np.array(Image.open(image_path))
        fig = plt.figure(frameon=False)

        fig.set_size_inches(im.shape[1] / 200, im.shape[0] / 200)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(im)
        for j in range(1, self.num_classes):
            boxes = boxes[j][0]
            segms = masks[j][0]
            if segms == []:
                continue
            masks = mask_util.decode(segms)
            for k in range(boxes.shape[0]):
                score = boxes[k, -1]
                mask = masks[:, :, k]
                if score < 0.5:
                    continue
                bbox = boxes[k, :]
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                    fill=False, edgecolor='g', linewidth=1, alpha=0.5)
                )
                # TODO: uncomment to get class names labelled
                # ax.text(bbox[0], bbox[1] - 2, name + '{:0.2f}'.format(score).lstrip('0'), fontsize=5, family='serif',
                #         bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'), color='white')
                color_mask = color_list[mask_color_id % len(color_list), 0:3]
                mask_color_id += 1
                w_ratio = .4
                for c in range(3):
                    color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio

                _, contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                for c in contour:
                    ax.add_patch(
                        Polygon(
                            c.reshape((-1, 2)),
                            fill=True, facecolor=color_mask, edgecolor='w', linewidth=0.8, alpha=0.5
                        )
                    )

        path = os.path.join(self.base_dir, '{}.png'.format(image_path.split('/')[-1]))
        fig.savefig(path, dpi=200)
        print("[Predict.py] Saving prediction to: {}".format(path))
        plt.close('all')

if __name__ == "__main__":
    args = parser.parse_args()
    update_config(args.cfg)
    predictor = SOGPredictor(model_path=args.model_path)
    predictor.predict(args.img_path)
