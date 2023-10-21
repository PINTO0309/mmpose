# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import csv
import time
import h5py
from argparse import ArgumentParser
from tqdm import tqdm

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id, pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    # print(f'##### pose_results: {pose_results}')
    # pose_results[0].pred_instances[0]
    """
    pred_instances: <InstanceData(

            META INFORMATION

            DATA FIELDS
            bbox_scores: array([1.], dtype=float32)
            bboxes: array([[227.13312, 187.32321, 273.2862 , 240.98148]], dtype=float32)
            keypoints: array(
                [
                    [
                        [251.61187596, 205.23822994],
                        [254.11584059, 202.23347248],
                        [248.90759415, 201.83283815],
                        [258.2223426 , 202.33363106],
                        [244.40045781, 202.43378964],
                        [262.02836884, 215.75488106],
                        [238.09046692, 216.75646689],
                        [267.53709104, 217.05694263],
                        [229.67714575, 224.76915346],
                        [261.92821026, 213.45123368],
                        [237.89014975, 213.65155084],
                        [259.02361128, 236.18723182],
                        [242.29712751, 236.58786615],
                        [266.4353466 , 227.27311801],
                        [230.27809726, 227.37327659],
                        [265.43376075, 235.185646  ],
                        [230.27809726, 235.58628033],
                        [264.63249207, 232.78184003],
                        [266.13487085, 232.28104712],
                        [267.53709104, 230.77866838],
                        [230.47841443, 226.47184935],
                        [266.4353466 , 231.68009562],
                        [230.27809726, 230.97898555],
                        [245.30188508, 202.03315531],
                        [245.30188508, 204.03632695],
                        [245.50220225, 205.73902285],
                        [245.90283659, 207.84235308],
                        [246.40362951, 209.9456833 ],
                        [247.30505678, 211.54822061],
                        [248.60711839, 212.7501236 ],
                        [250.10949718, 213.55139226],
                        [251.71203454, 213.851868  ],
                        [253.4147305 , 213.45123368],
                        [254.81695069, 212.7501236 ],
                        [255.81853655, 211.44806203],
                        [256.61980523, 209.74536614],
                        [257.12059816, 207.54187733],
                        [257.42107391, 205.63886427],
                        [257.5212325 , 203.93616837],
                        [257.5212325 , 202.13331389],
                        [247.00458103, 200.73109374],
                        [247.80584971, 200.33045942],
                        [248.80743557, 200.63093516],
                        [249.80902142, 200.73109374],
                        [250.71044869, 201.03156949],
                        [252.41314464, 201.23188666],
                        [253.31457191, 201.13172807],
                        [254.31615776, 201.03156949],
                        [255.31774362, 200.83125233],
                        [256.21917089, 201.23188666],
                        [251.51171737, 202.33363106],
                        [251.51171737, 203.73585121],
                        [251.61187596, 204.63727845],
                        [251.61187596, 205.73902285],
                        [250.51013152, 206.84076725],
                        [251.21124162, 206.84076725],
                        [251.81219313, 207.04108442],
                        [252.41314464, 206.84076725],
                        [253.01409615, 206.64045009],
                        [247.80584971, 202.13331389],
                        [248.70727698, 201.93299673],
                        [249.50854566, 202.03315531],
                        [250.40997293, 202.43378964],
                        [249.60870425, 202.43378964],
                        [248.70727698, 202.43378964],
                        [252.7136204 , 202.53394822],
                        [253.31457191, 202.23347248],
                        [254.31615776, 202.33363106],
                        [255.51806079, 202.53394822],
                        [254.41631635, 202.83442397],
                        [253.51488908, 202.73426539],
                        [249.60870425, 210.34631763],
                        [250.40997293, 209.34473181],
                        [251.51171737, 208.8439389 ],
                        [252.0125103 , 208.94409748],
                        [252.51330323, 208.8439389 ],
                        [253.71520625, 209.34473181],
                        [254.21599918, 210.14600046],
                        [253.91552342, 211.24774487],
                        [253.21441332, 211.6483792 ],
                        [252.11266889, 211.6483792 ],
                        [250.91076586, 211.54822061],
                        [250.10949718, 211.14758629],
                        [249.80902142, 210.34631763],
                        [250.81060727, 209.64520755],
                        [252.0125103 , 209.44489039],
                        [253.21441332, 209.64520755],
                        [254.01568201, 210.14600046],
                        [253.31457191, 210.94726912],
                        [252.0125103 , 211.0474277 ],
                        [250.71044869, 210.94726912],
                        [259.62456279, 196.52443329],
                        [260.12535572, 196.32411613],
                        [259.42424562, 194.92189598],
                        [258.52281835, 193.92031016],
                        [257.72154967, 210.04584188],
                        [261.02678299, 193.41951725],
                        [248.50695981, 193.21920008],
                        [248.40680122, 193.719993  ],
                        [248.90759415, 198.02681203],
                        [251.41155879, 193.1190415 ],
                        [251.3114002 , 193.21920008],
                        [251.11108303, 193.719993  ],
                        [250.71044869, 198.12697061],
                        [252.61346181, 193.21920008],
                        [252.11266889, 193.41951725],
                        [251.91235171, 193.92031016],
                        [252.0125103 , 198.12697061],
                        [253.71520625, 193.719993  ],
                        [253.4147305 , 193.719993  ],
                        [253.01409615, 194.92189598],
                        [252.81377898, 198.12697061],
                        [247.9060083 , 213.55139226],
                        [247.70569113, 196.52443329],
                        [246.00299517, 196.42427471],
                        [245.70251942, 196.42427471],
                        [253.01409615, 193.82015158],
                        [250.00933859, 194.02046874],
                        [252.81377898, 193.61983441],
                        [253.21441332, 193.92031016],
                        [254.81695069, 196.42427471],
                        [250.20965576, 193.719993  ],
                        [251.71203454, 193.719993  ],
                        [252.21282747, 194.32094449],
                        [252.91393757, 196.82490904],
                        [250.00933859, 193.82015158],
                        [250.81060727, 193.719993  ],
                        [251.81219313, 194.42110307],
                        [252.21282747, 198.82808068],
                        [247.30505678, 194.22078591],
                        [247.50537395, 194.02046874],
                        [247.40521537, 194.02046874],
                        [247.40521537, 198.32728777]
                    ]
                ]
            )
            keypoint_scores: array(
                [
                    [
                        0.39055237, 0.43155706, 0.41397393, 0.37338537, 0.38762686,
                        0.24463128, 0.29308343, 0.20119104, 0.13940978, 0.16255355,
                        0.09547621, 0.21882378, 0.23256376, 0.18470156, 0.17295861,
                        0.18326853, 0.1208386 , 0.11197783, 0.1771224 , 0.14233336,
                        0.08612108, 0.11921826, 0.12507229, 0.3884861 , 0.4082302 ,
                        0.38907042, 0.37214085, 0.35947144, 0.35539433, 0.35601407,
                        0.35218298, 0.35178792, 0.36086583, 0.38859758, 0.3943511 ,
                        0.4089193 , 0.4268634 , 0.4271938 , 0.45279896, 0.4277975 ,
                        0.3787186 , 0.37447548, 0.3828737 , 0.37866735, 0.37268353,
                        0.40685424, 0.42250377, 0.43595237, 0.43002713, 0.43556288,
                        0.41930807, 0.41790098, 0.38748214, 0.3636421 , 0.39571804,
                        0.41265047, 0.39812112, 0.41254133, 0.40030056, 0.40660536,
                        0.40997747, 0.43023294, 0.4245267 , 0.40798444, 0.3982481 ,
                        0.4348843 , 0.44830483, 0.4538659 , 0.45725912, 0.44512475,
                        0.43735933, 0.43378687, 0.44178796, 0.44058   , 0.42406797,
                        0.43769807, 0.44264424, 0.4378112 , 0.4265138 , 0.41602767,
                        0.413225  , 0.41682976, 0.4286958 , 0.4342265 , 0.44965082,
                        0.43578517, 0.44322687, 0.44605756, 0.4297381 , 0.4227852 ,
                        0.42846805, 0.13265114, 0.15622748, 0.14146815, 0.14147949,
                        0.14228635, 0.15691794, 0.18833655, 0.19588485, 0.15251747,
                        0.22387424, 0.24068034, 0.19264829, 0.15316509, 0.23004925,
                        0.2161184 , 0.19235207, 0.17349994, 0.19135052, 0.19622703,
                        0.16837986, 0.16271196, 0.10283282, 0.12821957, 0.10176074,
                        0.09283999, 0.1155706 , 0.16702124, 0.18882564, 0.19086549,
                        0.16335025, 0.20050123, 0.22297484, 0.20578052, 0.17428216,
                        0.18495083, 0.19598973, 0.18114886, 0.16213018, 0.14182708,
                        0.1347925 , 0.12593439, 0.12576255
                    ]
                ], dtype=float32)
        ) at 0x7b085b3c08e0>
    """
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None), pose_results

def make_csv(pose_result, frame_idx, edaban, timestamp_millis) -> list:
    coords = np.asarray(pose_result.pred_instances['keypoints'], dtype=np.float32)
    scores = np.asarray(pose_result.pred_instances['keypoint_scores'], dtype=np.float32)
    coords_scores: list = np.concatenate([coords, scores[..., np.newaxis]], axis=-1).reshape(-1).tolist()
    row_data = [f'{frame_idx:06}'] + [f'{edaban:02}'] + [timestamp_millis] + coords_scores
    return row_data


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--save-images',
        action='store_true',
        default=False,
        help='whether to save predicted images')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument('--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument('--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root, os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]
        if os.path.splitext(args.input)[-1] == '.h5':
            input_type = 'h5'

    print(f'##### input_type: {input_type}')
    pred_instances_list = []

    if input_type == 'image':

        # inference
        pred_instances, pose_results = process_one_image(args, args.input, detector, pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type == 'h5':

        with h5py.File(args.input, 'r') as hf:
            for index in tqdm(range(len(hf['video'])), dynamic_ncols=True):
                # データセットから指定したインデックスのフレームを読み取る
                frame = hf['video'][index]

                # inference
                pred_instances, pose_results = process_one_image(args, frame, detector, pose_estimator, visualizer)
                # print(pose_results[0].pred_instances._data_fields) # {'keypoint_scores', 'bboxes', 'keypoints', 'bbox_scores'}
                # print(np.asarray(pose_results[0].pred_instances['keypoints'], dtype=np.float32))
                # print(np.asarray(pose_results[0].pred_instances['keypoint_scores'], dtype=np.float32))

                frame_idx = index + 1
                edaban = 1
                timestamp_millis = ((frame_idx-1) * 1000) // fps
                for pose_result in pose_results:
                    row_data = make_csv(pose_result, frame_idx, edaban, timestamp_millis)
                    pred_instances_list.append(row_data)
                    edaban += 1

                if args.save_predictions:
                    pred_instances_list.append(dict(frame_id=frame_idx, instances=split_instances(pred_instances)))

                if output_file and args.save_images:
                    img_vis = visualizer.get_image()
                    basename_without_ext = os.path.splitext(os.path.basename(args.input))[0]
                    # ext = os.path.splitext(args.input)[-1]
                    output_file = os.path.join(args.output_root, f'{basename_without_ext}_{index:06}.png')
                    print(f'output_file: {output_file}')
                    mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        video_writer = None
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            # topdown pose estimation
            pred_instances, pose_results = process_one_image(args, frame, detector,  pose_estimator, visualizer, 0.001)

            edaban = 1
            timestamp_millis = ((frame_idx-1) * 1000) // fps
            for pose_result in pose_results:
                row_data = make_csv(pose_result, frame_idx, edaban, timestamp_millis)
                pred_instances_list.append(row_data)
                edaban += 1

            # if args.save_predictions:
            #     # save prediction results
            #     pred_instances_list.append(dict(frame_id=frame_idx, instances=split_instances(pred_instances)))

            # output videos
            if output_file and args.save_images:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            # press ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions and len(pred_instances_list) > 0:
        # with open(args.pred_save_path, 'w') as f:
        #     json.dump(
        #         dict(
        #             meta_info=pose_estimator.dataset_meta,
        #             instance_info=pred_instances_list),
        #         f,
        #         indent='\t')
        basename_without_ext = os.path.splitext(os.path.basename(args.input))[0]
        output_file = os.path.join(args.output_root, f'{basename_without_ext}.csv')
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(pred_instances_list)
        print(f'predictions have been saved at {args.pred_save_path}')


if __name__ == '__main__':
    main()
