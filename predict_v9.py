from tifffile import imread, imwrite
from skimage import filters
import subprocess
import pointnet_predict
import os
from datetime import datetime
import numpy as np
import argparse
import sys
import time
import cc3d


# v4 split clusters with 2 nuclei

# parser = argparse.ArgumentParser()

# parser.add_argument('-b', '--f-data', dest='f_data', type=str, default='b')
# parser.add_argument('-f', dest='f_data', type=str, default=None)

# parser = argparse.ArgumentParser(description='Microglia Cell Segmentation')
#
# parser.add_argument('-f', '--f-data', dest='f_data', type=str, default=None)
# parser.add_argument('-sd', '--save-dir', dest='save_dir', type=str, default='predictions/')
# parser.add_argument('-ds', '--down-sample', dest='ds', nargs='+', type=int, default=None)
# parser.add_argument('-inv', '--invert', dest='inv', type=bool, default=False)
# parser.add_argument('-scale', '--scale', dest='scale', type=bool, default=False)
# parser.add_argument('-delta', '--delta', dest='delta', type=float, default=3)
#
# parser.add_argument('-oc', '--otsu-cell', dest='otsu_cell', type=float, default=0.5)
# parser.add_argument('-on', '--otsu-nuclei', dest='otsu_nuclei', type=float, default=0.5)
# parser.add_argument('-rc', '--radius-cell', dest='radius_cell', type=float, default=1.8)
# parser.add_argument('-rn', '--radius-nuclei', dest='radius_nuclei', type=float, default=1.8)
# parser.add_argument('-pc', '--persi-cell', dest='persi_cell', type=float, default=0.5)
# parser.add_argument('-pn', '--persi-nuclei', dest='persi_nuclei', type=float, default=0.3)

# parser = argparse.ArgumentParser(description='Interactive Persistence-based Clustering')
# parser.add_argument('-a', '--persi-nuclei', dest='persi_nuclei', type=float, default=0.3)
# parser.add_argument('-b', '--persi-nuclei', dest='persi_nuclei', type=float, default=0.3)

# parser.add_argument('-f', '--f-data', dest='f_data', type=str, default=None)

# parser = argparse.ArgumentParser()
# parser.add_argument('--f', type=str, default=None, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet_cls',
#                     help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
# parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 250]')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
# parser.add_argument('--learning_rate', type=float, default=0.00001, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
# parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
# parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
# parser.add_argument('--decay_rate', type=float, default=1, help='Decay rate for lr decay [default: 0.8]')
# FLAGS = parser.parse_args()

# args = vars(parser.parse_args())
# print(args)

# if FLAGS.f_data is None:
#     print('give input path by \'-f path-to-tiff-file\'')
#     sys.exit()


# fname = FLAGS.f_data.split('/')[-1]
#
# FLAGS.otsu_cell = FLAGS.otsu_cell
# FLAGS.otsu_nuclei = FLAGS.otsu_nuclei
# FLAGS.radius_cell = str(FLAGS.radius_cell)
# FLAGS.radius_nuclei = str(FLAGS.radius_nuclei)
# FLAGS.persi_cell = str(FLAGS.persi_cell)
# FLAGS.persi_nuclei = str(FLAGS.persi_nuclei)
#
# print('otsu cell: %.1f' % FLAGS.otsu_cell)
# print('radius cell: %.1f' % FLAGS.radius_cell)
# print('persi cell: %.1f' % FLAGS.persi_cell)


# if not os.path.isdir(FLAGS.save_dir):
#     os.mkdir(FLAGS.save_dir)
#
# if FLAGS.f_data is None:
#     print('please input the input path by \'-f path-to-tiff-file\'')
#     sys.exit()


def down_sample_cc(x, ds):
    """

    :param x: z * c * x * y
    :param ds:
    :return:
    """
    x_ds = np.arange(0, x.shape[0], ds[0])
    y_ds = np.arange(0, x.shape[1], ds[1])
    z_ds = np.arange(0, x.shape[2], ds[2])
    x = x[x_ds, :, :]
    x = x[:, y_ds, :]
    x = x[:, :, z_ds]
    return x


def down_sample(x, ds):
    """

    :param x: z * c * x * y
    :param ds:
    :return:
    """
    x_ds = np.arange(0, x.shape[0], ds[0])
    y_ds = np.arange(0, x.shape[2], ds[1])
    z_ds = np.arange(0, x.shape[3], ds[2])
    x = x[x_ds, :, :, :]
    x = x[:, :, y_ds, :]
    x = x[:, :, :, z_ds]
    return x


def remove_by_num_points(x, num_points):
    values = np.sort(np.reshape(x, [-1]))
    th = values[-num_points+1]
    x[np.where(x < th)] = 0
    return x


def otsu_thresholding_nuclei(x, adj, num_points):
    idx_mid_slice = int(x.shape[0] / 2)
    val = filters.threshold_otsu(x[idx_mid_slice]) * adj
    print('otsu: %.2f' % val)
    idx = np.where(x < val)
    if len(idx[0]) > num_points:
        values = np.sort(np.reshape(x, [-1]))
        th = values[-num_points + 1]
        x[np.where(x < th)] = 0
    else:
        x[idx] = 0
    return x


def otsu_thresholding(x, adj):
    idx_mid_slice = int(x.shape[0] / 2)
    val = filters.threshold_otsu(x[idx_mid_slice]) * adj
    print('otsu: %.2f' % val)
    idx = np.where(x < val)
    x[idx] = 0
    return x


def pad_or_trim(x, n):
    if len(x) < n:
        x_ = np.zeros([n, x.shape[-1]])
        x_[:len(x)] = x
    elif len(x) > n:
        perm = np.random.permutation(len(x))[:n]
        x_ = x[perm]
    else:
        x_ = x
    return x_


def sort_pred_by_radius(pred):
    s = []
    cc_idx = np.unique(pred).tolist()[1:]
    for i in cc_idx:
        idx = np.where(pred == i)
        dx = np.max(idx[1]) - np.min(idx[1])
        dy = np.max(idx[2]) - np.min(idx[2])
        s.append(max(dx, dy))
    # return sorted(cc_idx, key=lambda k: s[k], reverse=True)
    idx_sorted = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    return np.array(cc_idx)[idx_sorted]

# def sort_pred_by_radius_cc(pred):
#     s = []
#     cc_idx = np.unique(pred).tolist()[1:]
#     for i in cc_idx:
#         idx = np.where(pred == i)
#         dx = np.max(idx[1]) - np.min(idx[1])
#         dy = np.max(idx[2]) - np.min(idx[2])
#         s.append(max(dx, dy))
#     return sorted(range(len(s)), key=lambda k: s[k], reverse=True)


def remove_by_size(preds, th):
    ds = [2, 4, 4]
    preds_ds = down_sample_cc(preds, ds)
    cc_idx = np.unique(preds_ds).tolist()[1:]
    idx_list = []
    for idx in cc_idx:
        # print('removing by size %d/%d' % (idx, np.max(cc_idx)))
        size = len(np.where(preds_ds == idx)[0])
        if size > th / np.prod(ds):
            idx_list.append(idx)
    pred_removed = np.zeros(preds.shape)
    for i in range(len(idx_list)):
        pred_removed[np.where(preds == idx_list[i])] = i + 1
    return pred_removed


def remove_small_clusters(pred, max_num_clusters):
    ds = [2, 4, 4]
    pred = down_sample_cc(pred, ds)
    # print('%d clusters' % np.max(pred))
    # sorted_pred_class = np.array(sort_pred(pred)) + 1  # sort the cluster index from large to small
    sorted_pred_class = np.array(sort_pred_by_radius(pred)) # sort the cluster index from large to small
    pred_removed = np.zeros(pred.shape)
    for i in range(max_num_clusters):
        pred_removed[np.where(pred == sorted_pred_class[i])] = i + 1
    return pred_removed


# def remove_small_clusters_cc(pred, max_num_clusters):
#     ds = [2, 4, 4]
#     pred = down_sample_cc(pred, ds)
#     # print('%d clusters' % np.max(pred))
#     # sorted_pred_class = np.array(sort_pred(pred)) + 1  # sort the cluster index from large to small
#     sorted_pred_class = np.array(sort_pred_by_radius(pred)) + 1 # sort the cluster index from large to small
#     pred_removed = np.zeros(pred.shape)
#     for i in range(max_num_clusters):
#         pred_removed[np.where(pred == sorted_pred_class[i])] = i + 1
#     return pred_removed


def preprocessing(x, invert, otsu_coef, scale):

    print(x.shape)

    if np.min(x) != 0:
        x = x - np.min(x)
    if invert:
        x = np.max(x) - x

    x = otsu_thresholding(x, adj=otsu_coef)

    if scale:
        x = x / np.max(x)
    return x


#
# def preprocessing_nuclei(x, invert, otsu_coef, scale):
#
#     print(x.shape)
#     if np.min(x) != 0:
#         x = x - np.min(x)
#     if invert:
#         x = np.max(x) - x
#     x = otsu_thresholding_nuclei(x, adj=otsu_coef, num_points=num_points)
#     if scale:
#         x = x / np.max(x)
#     return x



# def pbc_nuclei(x, f_pointcloud, radius, th, max_num_classes):
#     """
#
#     :param data_path: path to the tiff file
#     :return: point_set_path: path to the point set defined by the user
#     """

    # if ds is not None:
    #     x = down_sample_nomask(x, ds)
    #
    # print(x.shape)
    # if np.min(x) != 0:
    #     x = x - np.min(x)
    # if invert:
    #     x = np.max(x) - x
    # x = otsu_thresholding_nuclei(x, adj=otsu_coef, num_points=num_points)
    # if scale:
    #     x = x / np.max(x)

    # idx = np.where(x != 0)
    # with open(f_pointcloud, 'w') as f:
    #     for i in range(len(idx[0])):
    #         line = str(idx[0][i]) + " " + str(idx[1][i]) + " " + str(idx[2][i]) + " " + str(
    #             x[idx[0][i], idx[1][i], idx[2][i]]) + "\n"
    #         f.write(line)
    #
    # print('pointcloud file generated! %d points' % len(idx[0]))
    # subprocess.run(["./main", f_pointcloud, radius, th])
    # pred_tiff = np.zeros(x.shape)
    # pred = []
    # with open('clusters.txt') as f:
    #     for line in f:
    #         if line[:-1] == 'NaN':
    #             pred.append(0)
    #         else:
    #             pred.append(int(line[:-1]))
    # pred = np.array(pred)
    # x = np.fromfile(f_pointcloud, dtype=np.float32,
    #                 sep=" ")
    # x = np.reshape(x, (-1, 4))
    # for i in range(len(x)):
    #     pred_tiff[int(x[i, 0]), int(x[i, 1]), int(x[i, 2])] = pred[i]
    #
    #
    # num_classes = np.max(pred_tiff)
    #
    # if max_num_classes is not None and num_classes > max_num_classes:
    #     print('removing small clusters...')
    #     pred_tiff = remove_small_clusters(pred_tiff, max_num_clusters=max_num_classes)
    # return pred_tiff


def pbc(x, f_pointcloud, radius, th, max_num_classes):
    """

    :param data_path: path to the tiff file
    :return: point_set_path: path to the point set defined by the user
    """

    idx = np.where(x != 0)
    with open(f_pointcloud, 'w') as f:
        for i in range(len(idx[0])):
            line = str(idx[0][i]) + " " + str(idx[1][i]) + " " + str(idx[2][i]) + " " + str(
                x[idx[0][i], idx[1][i], idx[2][i]]) + "\n"
            f.write(line)

    print('pointcloud file generated! %d points' % len(idx[0]))
    subprocess.run(["./main", f_pointcloud, radius, th])
    pred_tiff = np.zeros(x.shape, dtype=np.int16)
    pred = []
    with open('clusters.txt') as f:
        for line in f:
            if line[:-1] == 'NaN':
                pred.append(0)
            else:
                pred.append(int(line[:-1]))
    pred = np.array(pred)
    x = np.fromfile(f_pointcloud, dtype=np.float32,
                    sep=" ")
    x = np.reshape(x, (-1, 4))
    for i in range(len(x)):
        pred_tiff[int(x[i, 0]), int(x[i, 1]), int(x[i, 2])] = int(pred[i])

    num_classes = np.max(pred_tiff)

    if max_num_classes is not None and num_classes > max_num_classes:
        print('removing small clusters...')
        pred_tiff = remove_small_clusters(pred_tiff, max_num_clusters=max_num_classes)
    return pred_tiff


def iou(set_a, set_b):
    return len(set_a & set_b) / len(set_a.union(set_b))


def intersection(set_a, set_b):
    return len(set_a & set_b) / len(set_a)


def arrange_by_persistence(preds):

    pred_list = []

    while len(preds) > 0:
        pred_cur = preds[0]
        idx_delete_list = [0]
        pred_cell_list = [pred_cur]
        idx_set = set([i for i in range(0, len(preds))])
        for i in range(1, len(preds)):
            set_cur = set(zip(*pred_cur))
            set_i = set(zip(*preds[i]))
            if intersection(set_cur, set_i) > 0:
                pred_cell_list.append(preds[i])
                idx_delete_list.append(i)
        pred_list.append(pred_cell_list)
        idx_set = idx_set.difference(set(idx_delete_list))
        preds = [preds[i] for i in list(idx_set)]
    return pred_list


# def arrange_by_persistence(preds):
#     pred_list = []
#
#     while len(preds) > 0:
#         pred_cur = preds[0]
#         idx_delete_list = [0]
#         pred_cell_list = [pred_cur]
#         idx_set = set([i for i in range(0, len(preds))])
#         for i in range(1, len(preds)):
#             if np.sum(preds[i] * pred_cur) > 0:
#                 pred_cell_list.append(preds[i])
#                 idx_delete_list.append(i)
#         pred_list.append(np.array(pred_cell_list))
#         idx_set = idx_set.difference(set(idx_delete_list))
#         preds = preds[list(idx_set)]
#     return pred_list


# def nuclei_check(pbc_nuclei, pbc_cell_list):
#     for pred_cell_pbc in pbc_cell_list:
#         idx_pred = []
#         temp = np.zeros(pred_cell_pbc.shape)
#         temp[np.where(pred_nuclei_pbc >= 1)] = 1
#         idx_cell_list = np.unique(temp * pred_cell_pbc)[1:]
#         for idx_cell in idx_cell_list:
#             temp = np.zeros(pred_cell_pbc.shape)
#             temp[np.where(pred_cell_pbc == idx_cell)] = 1
#             idx_nuclei_list = np.unique(temp * pred_nuclei_pbc)[1:]
#             is_cell = False
#             for idx_nuclei in idx_nuclei_list:
#                 set_idx = set(zip(*np.where(pred_cell_pbc == idx_cell)))
#                 set_idx_nuclei = set(zip(*np.where(pred_nuclei_pbc == idx_nuclei)))
#                 d_iou = iou(set_idx_nuclei, set_idx)
#                 if d_iou > iou_th_nuclei_check:
#                     print(d_iou)
#                     is_cell = True
#                     break
#             if is_cell:
#                 idx_pred.append(idx_cell)
#
#         pred_cell_pbc_nuclei = np.zeros(pred_cell_pbc.shape)
#         for i in range(len(idx_pred)):
#             pred_cell_pbc_nuclei[np.where(pred_cell_pbc == idx_pred[i])] = i + 1
#         pred_cell_pbc_nuclei_list.append(pred_cell_pbc_nuclei)


def check_non_overlap(pred, nuclei_coor_list):
    for i in range(len(nuclei_coor_list)):
        nuclei_coor = nuclei_coor_list[i]
        pred_i = pred[nuclei_coor]
        pred_i[np.where(pred_i != 0)] -= np.max(pred_i)
        if np.sum(pred_i) != 0:
            return False
    return True


def separation_score_ranked(score_list):

    # score_mat = None
    id_sorted_list, score_sorted_list = [], []
    for i in range(len(score_list[0])):
        temp = np.array(score_list)
        target_col = np.expand_dims(temp[:, i], axis=1)
        rest_col = np.delete(temp, i, 1)
        score = target_col * np.prod(1 - rest_col, axis=1, keepdims=True)
        id_sorted_list.append(np.argsort(-score) + 1)
        score_sorted_list.append(-np.sort(-score))
        # if score_mat is None:
        #     score_mat = score
        # else:
        #     score_mat = np.concatenate((score_mat, score), axis=1)

    return id_sorted_list, score_sorted_list
    # score = np.sum(score_mat, axis=1)
    # return -np.sort(-score), np.argsort(-score)
        # max_score_list.append(np.max(score))
        # max_idx_list.append(np.argmax(score) + 1)


def separation_score(score_list):
    """

    :param score_list: n_cell * n_nuclei
    :return: best score and n_nuclei indices
    """

    max_score_list = []
    max_idx_list = []
    for i in range(len(score_list[0])):
        temp = np.array(score_list)
        target_col = np.expand_dims(temp[:, i], axis=1)
        rest_col = np.delete(temp, i, 1)
        score = target_col * np.prod(1 - rest_col, axis=1, keepdims=True)
        max_score_list.append(np.max(score))
        max_idx_list.append(np.argmax(score) + 1)
    return max_score_list, max_idx_list


# def cc_score(pbc_pred, cluster_id_chosen_list):



# def split_cluster_v2(data, persi_cell_split, nuclei_coor_list, radius_cell, temp_dir):
#
#     pred = pbc(data, 'pointcloud_split.txt', radius=radius_cell, th=persi_cell_split, max_num_classes=None)
#
#     best_score, best_idx, best_idx_list = -1, None, []
#
#     score_list = []
#
#     n_nuclei = len(nuclei_coor_list)
#
#     for j in range(1, int(np.max(pred)) + 1):
#         set_pred = set(zip(*np.where(pred == j)))
#         score_cell_list = []
#         for k in range(len(nuclei_coor_list)):
#             set_nuclei = set(zip(*nuclei_coor_list[k]))
#             score_cell_list.append(intersection(set_nuclei, set_pred))
#         score_list.append(score_cell_list)
#     cluster_id_sorted, cluster_score_sorted = separation_score_ranked(score_list)
#
#     cluster_id_chosen_list = [[]] * n_nuclei
#     for i in range(n_nuclei):
#         j = 0
#         while cluster_score_sorted[j] == 1:
#             cluster_id_chosen_list[i].append(cluster_id_sorted[j])
#
#     for i in range(len(cluster_id_sorted)):
#         cluster
#     # for i in range(len(cluster_id_sorted)):
#     #     if cluster_score_sorted[i] == n_nuclei:
#     #         cluster_id_chosen_list.append(cluster_id_sorted[i])
#     #         continue
#
#
#     for i in range(len(nuclei_coor_list)):
#
#         pred_temp = np.copy(pred_list[best_idx])
#         pred_ids = best_idx_list[best_idx]
#         for j in range(len(nuclei_coor_list)):
#             if j == i: continue
#             pred_temp[np.where(pred_temp == pred_ids[j])] = 0
#         pred_temp[np.where(pred_temp > 0)] = 1
#         cc_i = cc3d.connected_components(pred_temp)
#         imwrite(temp_dir + 'temp_cc' + str(i) + '.tiff', cc_i)
#         pred_temp = np.copy(pred_list[best_idx])
#         pred_temp[np.where(pred_temp != pred_ids[i])] = 0
#         pred_temp[np.where(pred_temp == pred_ids[i])] = 1
#         cc_idx = np.unique(pred_temp * cc_i)[1]
#         pred_split[np.where(cc_i == cc_idx)] = i + 1
#
#     return pred_split


def split_cluster(data, persi_cell_low_list, nuclei_coor_list, radius_cell, temp_dir):

    pred_list = []
    for p in persi_cell_low_list:
        pred_pbc = pbc(data, 'pointcloud_split.txt', radius=radius_cell, th=p, max_num_classes=None)
        pred_list.append(pred_pbc)
        imwrite(temp_dir + 'temp_pbc_split' + p + '.tiff', pred_pbc)

    best_score, best_idx, best_idx_list = -1, None, []
    for i in range(len(pred_list)):
        pred = pred_list[i]
        score_list = []

        for j in range(1, int(np.max(pred)) + 1):
            set_pred = set(zip(*np.where(pred == j)))
            score_cell_list = []
            for k in range(len(nuclei_coor_list)):
                set_nuclei = set(zip(*nuclei_coor_list[k]))
                score_cell_list.append(intersection(set_nuclei, set_pred))
            score_list.append(score_cell_list)
        max_score_list, max_idx_list = separation_score(score_list)
        if np.sum(max_score_list) > best_score:
            best_score = np.sum(max_score_list)
            best_idx = i
            best_idx_list.append(max_idx_list)
    print('best p')
    print(best_idx)
    print('best score')
    print(best_score)
    print('best cell')
    print(best_idx_list[best_idx])
    pred_split = np.zeros(data.shape)

    for i in range(len(nuclei_coor_list)):
        pred_temp = np.copy(pred_list[best_idx])
        pred_ids = best_idx_list[best_idx]
        for j in range(len(nuclei_coor_list)):
            if j == i: continue
            pred_temp[np.where(pred_temp == pred_ids[j])] = 0
        pred_temp[np.where(pred_temp > 0)] = 1
        cc_i = cc3d.connected_components(pred_temp)
        imwrite(temp_dir + 'temp_cc' + str(i) + '.tiff', cc_i)
        pred_temp = np.copy(pred_list[best_idx])
        pred_temp[np.where(pred_temp != pred_ids[i])] = 0
        pred_temp[np.where(pred_temp == pred_ids[i])] = 1
        cc_idx = np.unique(pred_temp * cc_i)[1]
        pred_split[np.where(cc_i == cc_idx)] = i + 1

    return pred_split


# def split_cluster(data, persi_cell_low_list, nuclei_coor_list, radius_cell):
#
#     pred_list = []
#     for p in persi_cell_low_list:
#         pred_pbc = pbc(data, 'pointcloud_split.txt', radius=radius_cell, th=p, max_num_classes=None)
#         pred_list.append(pred_pbc)
#         imwrite('temp/temp_pbc_split' + p + '.tiff', pred_pbc)
#
#     best_iou = []
#     best_id_list = []
#     for i in range(len(pred_list)):
#         pred = pred_list[i]
#         best_pred = 0
#         id_list = []
#         for k in range(len(nuclei_coor_list)):
#             set_nuclei = set(zip(*nuclei_coor_list[k]))
#             best_temp = 0
#             best_cell_id = None
#             for j in range(1, int(np.max(pred)) + 1):
#                 set_pred = set(zip(*np.where(pred == j)))
#                 iou_temp = iou(set_pred, set_nuclei)
#                 if iou_temp > best_temp:
#                     best_temp = iou_temp
#                     best_cell_id = j
#             best_pred += best_temp
#             id_list.append(best_cell_id)
#         best_iou.append(best_pred)
#         best_id_list.append(id_list)
#     print(best_iou)
#     best_idx = np.argmax(best_iou)
#     pred_split = np.zeros(data.shape)
#     for i in range(len(nuclei_coor_list)):
#         pred_temp = np.copy(pred_list[best_idx])
#         pred_ids = best_id_list[best_idx]
#         for j in range(len(nuclei_coor_list)):
#             if j == i: continue
#             pred_temp[np.where(pred_temp == pred_ids[j])] = 0
#         pred_temp[np.where(pred_temp > 0)] = 1
#         cc_i = cc3d.connected_components(pred_temp)
#         imwrite('temp/temp_cc' + str(i) + '.tiff', cc_i)
#         pred_temp = np.copy(pred_list[best_idx])
#         pred_temp[np.where(pred_temp != pred_ids[i])] = 0
#         pred_temp[np.where(pred_temp == pred_ids[i])] = 1
#         cc_idx = np.unique(pred_temp * cc_i)[1]
#         pred_split[np.where(cc_i == cc_idx)] = i + 1
#
#     return pred_split

def rearrange_idx(x):
    idx_list = np.unique(x).tolist()[1:]
    temp = np.zeros(x.shape)
    for i in range(len(idx_list)):
        temp[np.where(x == idx_list[i])] = i + 1
    return temp


def gen_nuclei_mask(nuclei_cc):
    mask = np.zeros(nuclei_cc.shape)
    for i in range(int(np.max(nuclei_cc))):
        idx = np.where(nuclei_cc == i)
        x_max = np.max(idx[1]) + 20
        x_min = np.min(idx[1]) - 20
        y_max = np.max(idx[2]) + 20
        y_min = np.min(idx[2]) - 20
        z_max = np.max(idx[0]) + 10
        z_min = np.min(idx[0]) - 10
        mask[z_min:z_max, x_min:x_max, y_min:y_max] = 1
    return mask


def inference(f_data, persi_cell_list, persi_cell_split_list, iou_th_nuclei_check=0.2, otsu_cell=0.5, radius_cell='3',
              otsu_nuclei=0.5, radius_nuclei='1', persi_nuclei='0.3', n_points=3000, ds=None, nuclei_size_th=None,
              save_dir='predictions_v3/', temp_dir='temp/'):
    """

    :param f_data: path to 4-d input data with data[:, 0, :, :] being the nuclei, data[:, 1, :, :] being the cells.
    :param f_save:
    :param persi_cell:
    :param iou_th:
    :param o_th_cell:
    :param radius_cell:
    :param o_th_nuclei:
    :param radius_nuclei:
    :param persi_nuclei:
    :param n_points:
    :return:
    """
    start_time = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    fname = f_data.split('/')[-1]
    print('processing %s' % str(fname))

    data = imread(f_data)

    if ds is not None:
        data = down_sample(data, ds)

    data_cell = data[:, 1, :, :]
    data_nuclei = data[:, 0, :, :]

    data_nuclei = preprocessing(data_nuclei, invert=False, scale=True, otsu_coef=otsu_nuclei)
    imwrite('temp_data.tiff', data_cell)
    data_cell = preprocessing(data_cell, invert=False, scale=True, otsu_coef=otsu_cell)
    data_nuclei[np.where(data_cell == 0)] = 0
    if len(np.where(data_cell > 0)[0]) > 7000000:
        values = np.sort(np.reshape(data_cell, [-1]))
        th = values[-5000000 + 1]
        data_cell[np.where(data_cell < th)] = 0

    print('*** PBC nuclei ***')
    pred_nuclei_pbc = pbc(x=data_nuclei,
                          f_pointcloud=fname + 'pointcloud.txt',
                          radius=radius_nuclei,
                          th=persi_nuclei,
                          max_num_classes=None)
    imwrite('test_nuclei_pbc.tif', pred_nuclei_pbc)
    pred_nuclei_pbc[np.where(pred_nuclei_pbc > 0 )] = 1
    pred_nuclei_cc = cc3d.connected_components(np.int8(pred_nuclei_pbc))
    pred_nuclei_cc = remove_by_size(pred_nuclei_cc, nuclei_size_th)

    # pred_nuclei_cc = remove_small_clusters(pred_nuclei_cc, 15)

    print('%d components' % np.max(pred_nuclei_cc))
    # pred_cell_pbc_list = []
    # temp_dir = 'temp/'
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    imwrite(temp_dir + fname +'nuclei_cc.tiff', pred_nuclei_cc)
    print('*** PBC cell ***')
    for p in persi_cell_list:
        print('p_cell=%s' % p)
        pred_cell_pbc = pbc(x=data_cell,
                        f_pointcloud=fname + 'pointcloud.txt',
                        radius=radius_cell,
                        th=p,
                        max_num_classes=None)
        # imwrite('test_cell_pbc.tif', pred_cell_pbc)
        imwrite(temp_dir + fname + '_pbc_' + p + '.tiff', pred_cell_pbc)
        # pred_cell_pbc_list.append(pred_cell_pbc)

    # pred_cell_pbc_nuclei_list = []
    cluster_merge_nuclei_dic = {}
    print('checking nuclei...')
    for p in persi_cell_list:
        pred_cell_pbc = imread(temp_dir + fname + '_pbc_' + p + '.tiff')
        idx_pred, idx_split_dic = [], {}
        temp = np.zeros(pred_cell_pbc.shape, dtype=np.int16)
        temp[np.where(pred_nuclei_cc >= 1)] = 1
        idx_cell_list = np.unique(temp * pred_cell_pbc)[1:]
        for idx_cell in idx_cell_list:
            temp = np.zeros(pred_cell_pbc.shape, dtype=np.int16)
            temp[np.where(pred_cell_pbc == idx_cell)] = 1
            idx_nuclei_list = np.unique(temp * pred_nuclei_cc)[1:]
            is_cell = False
            nuclei_hit_list = []
            for idx_nuclei in idx_nuclei_list:
                if idx_nuclei not in cluster_merge_nuclei_dic:
                    cluster_merge_nuclei_dic[idx_nuclei] = []
                set_idx = set(zip(*np.where(pred_cell_pbc == idx_cell)))
                set_idx_nuclei = set(zip(*np.where(pred_nuclei_cc == idx_nuclei)))
                d_iou = intersection(set_idx_nuclei, set_idx)
                if d_iou > iou_th_nuclei_check:
                    cluster_merge_nuclei_dic[idx_nuclei].append(idx_cell)
                    print(d_iou)
                    is_cell = True
                    nuclei_coor = np.where((pred_nuclei_cc == idx_nuclei) & (pred_cell_pbc == idx_cell))
                    nuclei_hit_list.append(nuclei_coor)
            if is_cell:
                idx_pred.append(idx_cell)
                if len(nuclei_hit_list) > 1:
                    idx_split_dic[idx_cell] = nuclei_hit_list

        #*** merge clusters overlapping the same nucleus***
        for idx in cluster_merge_nuclei_dic.keys():
            cell_list = cluster_merge_nuclei_dic[idx]
            if len(cell_list) > 1:
                for i in range(1, len(cell_list)):
                    pred_cell_pbc[np.where(pred_cell_pbc == cell_list[i])] = cell_list[0]

        # *** rearrange index ***
        pred_cell_pbc = rearrange_idx(pred_cell_pbc)
        imwrite(temp_dir + fname + '_pbc_merge_' + p + '.tiff', pred_cell_pbc)


        # no splitting
        pred_cell_pbc_nuclei = np.zeros(pred_cell_pbc.shape, dtype=np.int16)
        for i in range(len(idx_pred)):
            pred_cell_pbc_nuclei[np.where(pred_cell_pbc == idx_pred[i])] = i + 1
        imwrite(temp_dir + fname + 'pbc_nuclei_nosplit' + p + '.tiff', pred_cell_pbc_nuclei)

        # with splitting
        pred_cell_pbc_nuclei = np.zeros(pred_cell_pbc.shape, dtype=np.int16)
        cell_id = 1

        for i in range(len(idx_pred)):
            if idx_pred[i] not in idx_split_dic:
                pred_cell_pbc_nuclei[np.where(pred_cell_pbc == idx_pred[i])] = cell_id
                cell_id += 1
            else:
                print('***splitting cells***')
                nuclei_hit_list = idx_split_dic[idx_pred[i]]
                # data_temp = np.copy(data[:, 1, :, :])
                data_temp = np.copy(data_cell)
                data_temp[np.where(pred_cell_pbc != idx_pred[i])] = 0
                pred_split = split_cluster(data_temp, persi_cell_split_list, nuclei_hit_list, radius_cell, temp_dir)
                for j in range(1, int(np.max(pred_split)) + 1):
                    pred_cell_pbc_nuclei[np.where(pred_split == j)] = cell_id
                    cell_id += 1
        # imwrite('test_split_' + p + '.tiff', pred_cell_pbc_nuclei)
        # pred_cell_pbc_nuclei_list.append(pred_cell_pbc_nuclei)
        imwrite(temp_dir + fname + '_pbc_nuclei_' + p + '.tiff', pred_cell_pbc_nuclei)
    ###***pointnet***
    pred_by_persi_list = []
    for p in persi_cell_list[::-1]:
        pred_cell_pbc_nuclei = imread(temp_dir + fname + '_pbc_nuclei_' + p + '.tiff')
        data_pointnet = []
        for i in range(1, int(np.max(pred_cell_pbc_nuclei) + 1)):
            idx = list(zip(*np.where(pred_cell_pbc_nuclei == i)))
            idx = np.array(idx)
            idx = idx / np.expand_dims(pred_cell_pbc_nuclei.shape, axis=0)
            idx = pad_or_trim(idx, n=n_points)
            data_pointnet.append(idx)
        data_pointnet = np.array(data_pointnet)
        pred_pointnet = pointnet_predict.inference(data=data_pointnet)
        pred_pointnet = np.argmax(pred_pointnet, 1)
        # print(np.sum(pred_pointnet))
        ids_cell = np.where(pred_pointnet == 1)[0] + 1

        pred_cellidx_pbc_nuclei_pointnet_list = []
        for i in range(len(ids_cell)):
            pred_cellidx_pbc_nuclei_pointnet_list.append(np.where(pred_cell_pbc_nuclei == ids_cell[i]))
        pred_by_persi_list += pred_cellidx_pbc_nuclei_pointnet_list

    pred_list = arrange_by_persistence(pred_by_persi_list)
    # save predictions
    if ds is None:
        f_ds = ''
    else:
        f_ds = '_ds_' + str(ds[0]) + str(ds[1]) + str(ds[2])
    for i in range(len(pred_list)):
        fname_settings = '_otsu_c_' + str(otsu_cell) + '_iou_' + str(iou_th_nuclei_check) + f_ds + '_cell_' + str(i+1)
        f_pred = save_dir + fname + fname_settings + '.tiff'
        pred_final = np.zeros([len(pred_list[i])] + list(data_cell.shape), dtype=np.bool_)
        for j in range(len(pred_list[i])):
            pred_final[j][pred_list[i][j]] = 1
        imwrite(f_pred, pred_final.astype(np.int8))

    print('time elapsed: %.2f sec' % (time.time() - start_time))


if __name__ == '__main__':

    inference(
                # f_data='/home/yue/Desktop/data/Microglia_December_2020/20201103-slide6-1-6_slice2_left.tif',
                # f_data='/home/yue/Desktop/data/Microglia_December_2020/20201105-slide9-10-4_slice3_right.tif',
                # f_data='/home/yue/Desktop/data/Microglia_December_2020/20201105-slide7-5-4_slice3_right.tif',
                f_data='/home/yue/Desktop/data/Microglia_December_2020/20201105-slide8-11-3_slice1_left.tif',
                otsu_cell=0.7,
                iou_th_nuclei_check=0.01,
                persi_cell_list=['0.6', '0.7', '0.8'],
                # persi_cell_list=['0.2', '0.3', '0.4'],
                # persi_cell_list=['0.6'],
                persi_cell_split_list=['0.5'],
                # persi_cell_split_list=['0.15'],
                persi_nuclei='0.2',
                radius_cell='1.8',
                radius_nuclei='1.8',
                otsu_nuclei=0.5,
                nuclei_size_th=1000,
                save_dir='/media/yue/Data/PycharmProjects_Backup/microglia/predictions/',
                temp_dir='/media/yue/Data/PycharmProjects_Backup/microglia/temp/',
                ds=(2, 2, 2))
