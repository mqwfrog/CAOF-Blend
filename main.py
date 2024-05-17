import torch
import os
import argparse
import numpy as np
import time
import os.path
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from sklearn.manifold import TSNE
from augmentations import DataTransform
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from collections import Counter


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=['HAR', 'SHAR', 'wisdm', 'Epilepsy', 'PenDigits', 'FingerMovements',
                                              'StandWalkJump', 'PhonemeSpectra'], default='HAR', help=' ')
    parser.add_argument('--p', default=0.5, help='the weight of local outlier factor')
    parser.add_argument('--contamination', default=0.1, help='the proportion of outliers')
    parser.add_argument('--dimension', choices=[2, 3], default=2, help='the dimension of visualization')
    parser.add_argument('--use_tsne', action="store_true", default=False, help='adopt t-SNE to adjust visualization')
    parser.add_argument('--original', action="store_true", default=False, help='original data without data augmentation')
    parser.add_argument('--k_dblof', default=10, help='the number of neighbors when computing dataset-based lof')
    parser.add_argument('--k_iclof', default=10, help='the number of neighbors when computing intra-cluster lof')
    parser.add_argument('--cont_dblof', default=0.1, help='the contamination when computing dataset-based lof')
    parser.add_argument('--cont_iclof', default=0.1, help='the contamination when computing intra-cluster lof')

    args = parser.parse_args()

    data_path = 'data/'
    x_train, y_train, x_test, y_test = read_MTS_dataset(data_path, args.dataset)
    n_classes = len(np.unique(y_train))
    print(f'n_classes:{n_classes}')

    if args.original:
        x_all = x_train
        y_all = y_train.reshape([y_train.size()[0], -1])

    else:
        if os.path.exists('x_all.pt') and os.path.exists('y_all.pt'):
            x_all = torch.load("x_all.pt")
            y_all = torch.load("y_all.pt")
            ranges = [(i * x_train.size()[0], (i + 1) * x_train.size()[0]) for i in range(8)]
            sub_tensors = [x_all[start:end] for start, end in ranges]
            x_jitter = sub_tensors[1]
            x_scaling = sub_tensors[2]
            x_permutation = sub_tensors[3]
            x_rotation = sub_tensors[4]
            x_magnitude_warp = sub_tensors[5]
            x_time_warp = sub_tensors[6]
            x_window_slice = sub_tensors[7]
        else:
            start = time.time()
            x_jitter, x_scaling, x_permutation, x_rotation, x_magnitude_warp, x_time_warp, x_window_slice = \
                    DataTransform(x_train)
            print(f'x_jitter.shape:{x_jitter.shape}')
            print('augmentation time: {:2.2f} seconds'.format(time.time() - start))

            x_all = torch.cat([x_train, x_jitter, x_scaling, x_permutation, x_rotation, x_magnitude_warp, x_time_warp, x_window_slice], dim=0)
            y_all = torch.cat([y_train.reshape([y_train.size()[0], -1])] * 8, dim=0)
            torch.save(x_all, f"data/{args.dataset}/x_all.pt")
            torch.save(y_all, f"data/{args.dataset}/y_all.pt")

    X = x_all.reshape([x_all.size()[0], -1])

    if args.use_tsne:
        print('*********use TSNE***********')
        tsne = TSNE(n_components=args.dimension, perplexity=30, learning_rate=200, random_state=42)
        X = tsne.fit_transform(X)
    print(f'X.shape:{X.shape}')

    """ 1. Perform k-means clustering """
    start2 = time.time()

    kmeans = KMeans(n_clusters=n_classes, random_state=7)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    # Get the distances of each sample to the centroids
    distances = kmeans.transform(X)
    # Get the index of the closest centroid for each sample
    closest_centroids = np.argmin(distances, axis=1)
    # Calculate the average intra-cluster distance for each cluster
    average_intra_cluster_distances = np.zeros(n_classes)
    for i in range(n_classes):
        cluster_samples = X[closest_centroids == i]
        cluster_centroid = kmeans.cluster_centers_[i]
        intra_cluster_distances = np.linalg.norm(cluster_samples - cluster_centroid, axis=1)
        average_intra_cluster_distances[i] = np.mean(intra_cluster_distances)

    print('Clustering Time: {:2.2f} seconds'.format(time.time() - start2))
    print(f'distances:{distances}')
    print(f'average_intra_cluster_distances:{average_intra_cluster_distances}')


    """ 2. Calculate dataset-based lof """
    start3 = time.time()
    clf_db = LocalOutlierFactor(n_neighbors=args.k_dblof, contamination=args.cont_dblof)
    print(f'x_all.shape:{x_all.shape}')
    print(f'type(x_all):{type(x_all)}')
    print(f'X.shape:{X.shape}')
    print(f'type(X):{type(X)}')
    clf_db.fit(X)  # fit the model for outlier detection (default)

    dblof = clf_db.negative_outlier_factor_  # lof of all samples and augmentations
    print(f'dblof:{dblof}')
    np.savetxt(f'dblof.txt', dblof, fmt='%s')
    print('Computing Dataset-based Lof Time: {:2.2f} seconds'.format(time.time() - start3))

    db_lrd = clf_db._lrd
    print(f'db_lrd:{db_lrd}')
    np.savetxt(f'db_lrd.txt', db_lrd, fmt='%s')

    db_y = clf_db.fit_predict(X)

    # plt.figure()
    # sns.histplot(dblof[db_y == 1], label="inlier scores", kde=True, color='#93c47d', edgecolor='none')
    # sns.histplot(dblof[db_y == -1], label="outlier scores", kde=True, color='#6fa8dc', edgecolor='none')
    # plt.title("Distribution of Outlier Scores from DBLOF Detector")
    # plt.legend()
    # plt.xlabel("Outlier Score")
    # plt.savefig('db_outlier_distribution.jpg')

    k_neighbors = clf_db.kneighbors(X)
    k_distances = k_neighbors[0]
    print(f'k_distances:{k_distances}')
    np.savetxt(f'k_distances.txt', k_distances, fmt='%s')
    k_ids = k_neighbors[1]
    print(f'k_ids:{k_ids}')
    k_distance_max = k_distances.max(axis=1)
    print(f'k_distance_max:{k_distance_max}')
    np.savetxt(f'k_distance_max.txt', k_distance_max, fmt='%s')

    """ 3. Calculate intra-cluster lof"""
    iclof = np.zeros_like(labels, dtype=float)
    ic_y = np.zeros_like(labels, dtype=float)

    start4 = time.time()
    for cluster_id in range(n_classes):
        # Filter the data points belonging to the current cluster
        cluster_mask = (labels == cluster_id)
        X_cluster = X[cluster_mask]
        # Compute LOF for the current cluster
        clf_ic = LocalOutlierFactor(n_neighbors=args.k_iclof, contamination=args.cont_iclof)
        clf_ic.fit(X_cluster)
        num_true_values = np.sum(cluster_mask)
        print("Number of True values:", num_true_values)
        print(f'cluster_mask.shape:{cluster_mask.shape}')
        print(f'iclof.shape:{iclof.shape}')
        print(f'clf_ic.negative_outlier_factor_.shape:{clf_ic.negative_outlier_factor_.shape}')

        iclof[cluster_mask] = clf_ic.negative_outlier_factor_
        ic_y[cluster_mask] = clf_ic.fit_predict(X_cluster)

    ic_lrd = clf_ic._lrd
    print(f'ic_lrd:{ic_lrd}')
    np.savetxt(f'ic_lrd.txt', ic_lrd, fmt='%s')

    np.savetxt(f'iclof.txt', iclof, fmt='%s')
    print('Computing Intra-Cluster Lof Time: {:2.2f} seconds'.format(time.time() - start4))

    for cluster_id in range(n_classes):
        cluster_samples = np.where(labels == cluster_id)[0]
        num_samples_in_cluster = len(cluster_samples)
        cluster_outliers = np.where(ic_y[labels == cluster_id] == -1)[0]
        num_outliers_in_cluster = len(cluster_outliers)
        proportion_outliers = num_outliers_in_cluster / num_samples_in_cluster if num_samples_in_cluster > 0 else 0.0
        print(f"Cluster {cluster_id}: {num_outliers_in_cluster} outliers/{num_samples_in_cluster} samples, "
              f"proportion {proportion_outliers:.4f}")
    # Cluster 0: 197 outliers/1963 samples, proportion 0.1004
    # Cluster 1: 177 outliers/1764 samples, proportion 0.1003
    # Cluster 2: 407 outliers/4062 samples, proportion 0.1002
    # Cluster 3: 681 outliers/6821 samples, proportion 0.0998
    # Cluster 4: 124 outliers/1238 samples, proportion 0.1002
    # Cluster 5: 509 outliers/5088 samples, proportion 0.1000

    # plt.figure()
    # sns.histplot(iclof[ic_y == 1], label="inlier scores", kde=True, color='#93c47d', edgecolor='none')
    # sns.histplot(iclof[ic_y == -1], label="outlier scores", kde=True, color='#6fa8dc', edgecolor='none')
    # plt.title("Distribution of Outlier Scores from ICLOF Detector")
    # plt.legend()
    # plt.xlabel("Outlier Score")
    # plt.savefig(f'ic_outlier_distribution.jpg')

    """ 4. Calculate outlier proportion """
    # dblof
    db_count_jitter_outlier = 0
    db_count_jitter_inlier = 0
    db_count_scaling_outlier = 0
    db_count_scaling_inlier = 0
    db_count_permutation_outlier = 0
    db_count_permutation_inlier = 0
    db_count_rotation_outlier = 0
    db_count_rotation_inlier = 0
    db_count_magnitude_warp_outlier = 0
    db_count_magnitude_warp_inlier = 0
    db_count_time_warp_outlier = 0
    db_count_time_warp_inlier = 0
    db_count_window_slice_outlier = 0
    db_count_window_slice_inlier = 0

    for i in range(len(x_train), 2 * len(x_train)):
        if db_y[i] == -1:
            db_count_jitter_outlier = db_count_jitter_outlier + 1
        else:
            db_count_jitter_inlier = db_count_jitter_inlier + 1
    db_count_jitter = db_count_jitter_outlier + db_count_jitter_inlier
    db_jitter_ratio = db_count_jitter_outlier / db_count_jitter

    for i in range(2 * len(x_train), 3 * len(x_train)):
        if db_y[i] == -1:
            db_count_scaling_outlier = db_count_scaling_outlier + 1
        else:
            db_count_scaling_inlier = db_count_scaling_inlier + 1
    db_count_scaling = db_count_scaling_outlier + db_count_scaling_inlier
    db_scaling_ratio = db_count_scaling_outlier / db_count_scaling

    for i in range(3 * len(x_train), 4 * len(x_train)):
        if db_y[i] == -1:
            db_count_permutation_outlier = db_count_permutation_outlier + 1
        else:
            db_count_permutation_inlier = db_count_permutation_inlier + 1
    db_count_permutation = db_count_permutation_outlier + db_count_permutation_inlier
    db_permutation_ratio = db_count_permutation_outlier / db_count_permutation

    for i in range(4 * len(x_train), 5 * len(x_train)):
        if db_y[i] == -1:
            db_count_rotation_outlier = db_count_rotation_outlier + 1
        else:
            db_count_rotation_inlier = db_count_rotation_inlier + 1
    db_count_rotation = db_count_rotation_outlier + db_count_rotation_inlier
    db_rotation_ratio = db_count_rotation_outlier / db_count_rotation


    for i in range(5 * len(x_train), 6 * len(x_train)):
        if db_y[i] == -1:
            db_count_magnitude_warp_outlier = db_count_magnitude_warp_outlier + 1
        else:
            db_count_magnitude_warp_inlier = db_count_magnitude_warp_inlier + 1
    db_count_magnitude_warp = db_count_magnitude_warp_outlier + db_count_magnitude_warp_inlier
    db_magnitude_warp_ratio = db_count_magnitude_warp_outlier / db_count_magnitude_warp

    for i in range(6 * len(x_train), 7 * len(x_train)):
        if db_y[i] == -1:
            db_count_time_warp_outlier = db_count_time_warp_outlier + 1
        else:
            db_count_time_warp_inlier = db_count_time_warp_inlier + 1
    db_count_time_warp = db_count_time_warp_outlier + db_count_time_warp_inlier
    db_time_warp_ratio = db_count_time_warp_outlier / db_count_time_warp

    for i in range(7 * len(x_train), 8 * len(x_train)):
        if db_y[i] == -1:
            db_count_window_slice_outlier = db_count_window_slice_outlier + 1
        else:
            db_count_window_slice_inlier = db_count_window_slice_inlier + 1
    db_count_window_slice = db_count_window_slice_outlier + db_count_window_slice_inlier
    db_window_slice_ratio = db_count_window_slice_outlier / db_count_window_slice

    db_ratio_dict = {}
    db_ratio_dict["db_jitter"] = db_jitter_ratio
    db_ratio_dict["db_scaling"] = db_scaling_ratio
    db_ratio_dict["db_permutation"] = db_permutation_ratio
    db_ratio_dict["db_rotation"] = db_rotation_ratio
    db_ratio_dict["db_magnitude_warp"] = db_magnitude_warp_ratio
    db_ratio_dict["db_time_warp"] = db_time_warp_ratio
    db_ratio_dict["db_window_slice"] = db_window_slice_ratio

    # print(sorted(db_ratio_dict.items(), key=lambda kv: (kv[1], kv[0])))
    # print(f'db_ratio_dict: {db_ratio_dict}')
    # torch.save(db_ratio_dict, "db_ratio_dict.pt")

    # iclof
    ic_count_jitter_outlier = 0
    ic_count_jitter_inlier = 0
    ic_count_scaling_outlier = 0
    ic_count_scaling_inlier = 0
    ic_count_permutation_outlier = 0
    ic_count_permutation_inlier = 0
    ic_count_rotation_outlier = 0
    ic_count_rotation_inlier = 0
    ic_count_magnitude_warp_outlier = 0
    ic_count_magnitude_warp_inlier = 0
    ic_count_time_warp_outlier = 0
    ic_count_time_warp_inlier = 0
    ic_count_window_slice_outlier = 0
    ic_count_window_slice_inlier = 0

    for i in range(len(x_train), 2 * len(x_train)):
        if ic_y[i] == -1:
            ic_count_jitter_outlier = ic_count_jitter_outlier + 1
        else:
            ic_count_jitter_inlier = ic_count_jitter_inlier + 1
    ic_count_jitter = ic_count_jitter_outlier + ic_count_jitter_inlier
    ic_jitter_ratio = ic_count_jitter_outlier / ic_count_jitter

    for i in range(2 * len(x_train), 3 * len(x_train)):
        if ic_y[i] == -1:
            ic_count_scaling_outlier = ic_count_scaling_outlier + 1
        else:
            ic_count_scaling_inlier = ic_count_scaling_inlier + 1
    ic_count_scaling = ic_count_scaling_outlier + ic_count_scaling_inlier
    ic_scaling_ratio = ic_count_scaling_outlier / ic_count_scaling

    for i in range(3 * len(x_train), 4 * len(x_train)):
        if ic_y[i] == -1:
            ic_count_permutation_outlier = ic_count_permutation_outlier + 1
        else:
            ic_count_permutation_inlier = ic_count_permutation_inlier + 1
    ic_count_permutation = ic_count_permutation_outlier + ic_count_permutation_inlier
    ic_permutation_ratio = ic_count_permutation_outlier / ic_count_permutation

    for i in range(4 * len(x_train), 5 * len(x_train)):
        if ic_y[i] == -1:
            ic_count_rotation_outlier = ic_count_rotation_outlier + 1
        else:
            ic_count_rotation_inlier = ic_count_rotation_inlier + 1
    ic_count_rotation = ic_count_rotation_outlier + ic_count_rotation_inlier
    ic_rotation_ratio = ic_count_rotation_outlier / ic_count_rotation

    for i in range(5 * len(x_train), 6 * len(x_train)):
        if ic_y[i] == -1:
            ic_count_magnitude_warp_outlier = ic_count_magnitude_warp_outlier + 1
        else:
            ic_count_magnitude_warp_inlier = ic_count_magnitude_warp_inlier + 1
    ic_count_magnitude_warp = ic_count_magnitude_warp_outlier + ic_count_magnitude_warp_inlier
    ic_magnitude_warp_ratio = ic_count_magnitude_warp_outlier / ic_count_magnitude_warp

    for i in range(6 * len(x_train), 7 * len(x_train)):
        if ic_y[i] == -1:
            ic_count_time_warp_outlier = ic_count_time_warp_outlier + 1
        else:
            ic_count_time_warp_inlier = ic_count_time_warp_inlier + 1
    ic_count_time_warp = ic_count_time_warp_outlier + ic_count_time_warp_inlier
    ic_time_warp_ratio = ic_count_time_warp_outlier / ic_count_time_warp

    for i in range(7 * len(x_train), 8 * len(x_train)):
        if ic_y[i] == -1:
            ic_count_window_slice_outlier = ic_count_window_slice_outlier + 1
        else:
            ic_count_window_slice_inlier = ic_count_window_slice_inlier + 1
    ic_count_window_slice = ic_count_window_slice_outlier + ic_count_window_slice_inlier
    ic_window_slice_ratio = ic_count_window_slice_outlier / ic_count_window_slice

    ic_ratio_dict = {}
    ic_ratio_dict["ic_jitter"] = ic_jitter_ratio
    ic_ratio_dict["ic_scaling"] = ic_scaling_ratio
    ic_ratio_dict["ic_permutation"] = ic_permutation_ratio
    ic_ratio_dict["ic_rotation"] = ic_rotation_ratio
    ic_ratio_dict["ic_magnitude_warp"] = ic_magnitude_warp_ratio
    ic_ratio_dict["ic_time_warp"] = ic_time_warp_ratio
    ic_ratio_dict["ic_window_slice"] = ic_window_slice_ratio


    print(f'db_ratio_dict:{db_ratio_dict}')
    print(f'ic_ratio_dict:{ic_ratio_dict}')
    # print(sorted(ic_ratio_dict.items(), key=lambda kv: (kv[1], kv[0])))
    # print(f'ic_ratio_dict: {ic_ratio_dict}')
    # torch.save(ic_ratio_dict, "ic_ratio_dict.pt")

    """ 5. Reweighting Sum (weights = normalized reciprocals of ratios)"""
    db_reciprocals = {key: 1 / (value+1e-4) for key, value in db_ratio_dict.items()} #+1e4 avoid 0
    db_sum_reciprocals = sum(db_reciprocals.values())
    db_weights_dict = {key: value / db_sum_reciprocals for key, value in db_reciprocals.items()}
    if not os.path.exists(f'{args.dataset}_results/dbk{args.k_dblof}_ick_{args.k_iclof}_results'):
        os.makedirs(f'{args.dataset}_results/dbk{args.k_dblof}_ick_{args.k_iclof}_results')
    torch.save(db_weights_dict, f"{args.dataset}_results/dbk{args.k_dblof}_ick_{args.k_iclof}_results/db_weights_dict.pt")
    print((f'db_weights:', sorted(db_weights_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)))

    start5 = time.time()
    db_mix_sum = x_jitter * db_weights_dict['db_jitter'] + x_scaling * db_weights_dict['db_scaling'] + \
             x_permutation * db_weights_dict['db_permutation'] + x_rotation * db_weights_dict['db_rotation'] \
             + x_magnitude_warp * db_weights_dict['db_magnitude_warp'] + x_time_warp * db_weights_dict['db_time_warp'] \
             + x_window_slice * db_weights_dict['db_window_slice']
    # db_mix_avg = db_mix_sum/7
    print('DB-Augmentation Blending Time: {:2.2f} seconds'.format(time.time() - start5))
    print(f'db_mix_sum.shape:{db_mix_sum.shape}')
    torch.save(db_mix_sum, f"{args.dataset}_results/dbk{args.k_dblof}_ick_{args.k_iclof}_results/db_mix_sum.pt")


    ic_reciprocals = {key: 1 / (value+1e-4) for key, value in ic_ratio_dict.items()}
    ic_sum_reciprocals = sum(ic_reciprocals.values())
    ic_weights_dict = {key: value / ic_sum_reciprocals for key, value in ic_reciprocals.items()}
    torch.save(ic_weights_dict, f"{args.dataset}_results/dbk{args.k_dblof}_ick_{args.k_iclof}_results/ic_weights_dict.pt")
    print((f'ic_weights:', sorted(ic_weights_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)))

    start6 = time.time()
    ic_mix_sum = x_jitter * ic_weights_dict['ic_jitter'] + x_scaling * ic_weights_dict['ic_scaling'] + \
                 x_permutation * ic_weights_dict['ic_permutation'] + x_rotation * ic_weights_dict['ic_rotation'] \
                 + x_magnitude_warp * ic_weights_dict['ic_magnitude_warp'] \
                 + x_time_warp * ic_weights_dict['ic_time_warp'] + x_window_slice * ic_weights_dict['ic_window_slice']
    # ic_mix_avg = ic_mix_sum / 7
    print('IC-Augmentation Blending Time: {:2.2f} seconds'.format(time.time() - start6))
    print(f'ic_mix_sum.shape:{ic_mix_sum.shape}')
    torch.save(ic_mix_sum, f"{args.dataset}_results/dbk{args.k_dblof}_ick_{args.k_iclof}_results/ic_mix_sum.pt")

    y = y_all.squeeze().numpy()
    true_counts = Counter(y)
    pred_counts = Counter(labels)
    print(f'true_counts:{true_counts}')
    print(f'pred_counts:{pred_counts}')


    # """ DBSCAN clustering """
    # # # Perform DBSCAN clustering on the reduced-dimensional data
    # # dbscan = DBSCAN(eps=30, min_samples=10)  # You may need to adjust eps and min_samples based on your data
    # # labels = dbscan.fit_predict(X)
    # # centroids =dbscan.core_sample_indices_
    # # Counter({-1: 2397, 0: 90, 1: 37, 4: 32, 5: 20, 3: 16, 6: 13, 2: 12})

    # """ OPTICS clustering """
    # eps = 200
    # plotpoints = 10000 #len(XX)
    # start = time.time()
    # optics = OPTICS(eps=eps, min_samples=1000)  # eps=3.8
    # optics.fit(X)
    # optics_labels_ = optics.labels_
    # print(optics.core_distances_)
    # print(optics.ordering_)
    # print(optics.reachability_)
    # print(optics_labels_)
    # np.savetxt(f'optics_core_distances_.txt', optics.core_distances_, fmt='%s')
    # np.savetxt(f'optics_ordering_.txt', optics.ordering_, fmt='%s')
    # np.savetxt(f'optics_reachability_.txt', optics.reachability_, fmt='%s')
    # np.savetxt(f'optics_labels_.txt', optics_labels_, fmt='%s')
    # print('OPTICS clustering time: {:2.2f} seconds'.format(time.time() - start))
    # XX = optics.reachability_[optics.ordering_]
    # plt.figure()
    # plt.plot(range(0, plotpoints), XX[0:plotpoints])
    # plt.plot([0, plotpoints], [eps, eps])

    # """ MDS """
    # start = time.time()
    # mds = MDS(n_components=2)
    # embeddings_mds = mds.fit_transform(X)
    # print(embeddings_mds.shape)
    # print(embeddings_mds)
    # np.savetxt(f'embeddings_mds.txt', embeddings_mds, fmt='%s')
    # print('MDS time: {:2.2f} seconds'.format(time.time() - start))


def read_MTS_dataset(data_path, dataset_name):
    dataset_path = os.path.join(data_path, dataset_name)
    train_dataset = torch.load(os.path.join(dataset_path, "train.pt"))
    test_dataset = torch.load(os.path.join(dataset_path, "test.pt"))

    X_train = train_dataset["samples"]
    Y_train = train_dataset["labels"]
    X_test = test_dataset["samples"]
    Y_test = test_dataset["labels"]

    if len(X_train.shape) < 3:
        X_train = X_train.unsqueeze(2)
    if len(X_test.shape) < 3:
        X_test = X_test.unsqueeze(2)

    if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        X_train = X_train.permute(0, 2, 1)
    if X_test.shape.index(min(X_test.shape)) != 1:  # make sure the Channels in second dim
        X_test = X_test.permute(0, 2, 1)

    if isinstance(X_train, np.ndarray):
        x_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(Y_train).long()
    else:
        x_train = X_train
        y_train = Y_train

    if isinstance(X_test, np.ndarray):
        x_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(Y_test).long()
    else:
        x_test = X_test
        y_test = Y_test

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    start_all = time.time()
    main()
    print('The total running time: {:2.2f} seconds'.format(time.time() - start_all))
