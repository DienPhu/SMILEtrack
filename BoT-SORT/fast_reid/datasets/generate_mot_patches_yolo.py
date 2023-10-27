import os
import argparse
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_trajectories_from_yolo(parent_dir):
    trajectories = []
    annotations_file_path = os.path.join(parent_dir, 'Annotations')
    if os.path.exists(annotations_file_path):
        with open(annotations_file_path, 'r') as annotations_file:
            txt_files = annotations_file.read().splitlines()

        for txt_file in txt_files:
            txt_file_path = os.path.join(parent_dir, txt_file)
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r') as f:
                    lines = f.read().split('\n')
                    for line in lines:
                        if line.strip():  # Check if the line is not empty
                            object_class, x_center, y_center, width, height = map(float, line.split())
                            # Convert YOLO format to x1, y1, x2, y2
                            x1 = int((x_center - width / 2) * image_width)  # Assuming image_width is known
                            y1 = int((y_center - height / 2) * image_height)  # Assuming image_height is known
                            x2 = int((x_center + width / 2) * image_width)
                            y2 = int((y_center + height / 2) * image_height)
                            trajectories.append([object_class, x1, y1, x2, y2])
    return np.array(trajectories)


def make_parser():
    parser = argparse.ArgumentParser("MOTChallenge ReID dataset")

    parser.add_argument("--data_path", default="", help="path to MOT data")
    parser.add_argument("--save_path", default="fast_reid/datasets", help="Path to save the MOT-ReID dataset")
    parser.add_argument("--mot", default=17, help="MOTChallenge dataset number e.g. 17, 20")

    return parser


def main(args):

    # Create folder for outputs
    save_path = os.path.join(args.save_path, 'MOT' + str(args.mot) + '-ReID')
    os.makedirs(save_path, exist_ok=True)

    save_path = os.path.join(args.save_path, 'MOT' + str(args.mot) + '-ReID')
    train_save_path = os.path.join(save_path, 'bounding_box_train')
    os.makedirs(train_save_path, exist_ok=True)
    test_save_path = os.path.join(save_path, 'bounding_box_test')
    os.makedirs(test_save_path, exist_ok=True)

    # Get gt data
    data_path = os.path.join(args.data_path, 'MOT' + str(args.mot), 'train')

    if args.mot == '17':
        seqs = [f for f in os.listdir(data_path) if 'FRCNN' in f]
    else:
        seqs = os.listdir(data_path)

    seqs.sort()

    id_offset = 0

    for seq in seqs:
        print(seq)
        print(id_offset)

        parent_dir = '/path/to/your/parent/directory'  # Specify the directory containing train, test, and annotation files

        gt = generate_trajectories_from_yolo(parent_dir)

        images_path = os.path.join(data_path, seq, 'img1')
        img_files = os.listdir(images_path)
        img_files.sort()

        num_frames = len(img_files)
        max_id_per_seq = 0
        for f in range(num_frames):

            img = cv2.imread(os.path.join(images_path, img_files[f]))

            if img is None:
                print("ERROR: Receive empty frame")
                continue

            H, W, _ = np.shape(img)

            det = gt[f + 1 == gt[:, 0], 1:].astype(np.int_)

            for d in range(np.size(det, 0)):
                id_ = det[d, 0]
                x1 = det[d, 1]
                y1 = det[d, 2]
                x2 = det[d, 3]
                y2 = det[d, 4]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, W)
                y2 = min(y2, H)

                # patch = cv2.cvtColor(img[y1:y2, x1:x2, :], cv2.COLOR_BGR2RGB)
                patch = img[y1:y2, x1:x2, :]

                max_id_per_seq = max(max_id_per_seq, id_)

                # plt.figure()
                # plt.imshow(patch)
                # plt.show()

                fileName = (str(id_ + id_offset)).zfill(7) + '_' + seq + '_' + (str(f)).zfill(7) + '_acc_data.bmp'

                if f < num_frames // 2:
                    cv2.imwrite(os.path.join(train_save_path, fileName), patch)
                else:
                    cv2.imwrite(os.path.join(test_save_path, fileName), patch)

        id_offset += max_id_per_seq


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
