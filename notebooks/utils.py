from os.path import exists
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from datashader.bundling import hammer_bundle
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


TSNE_PATH_PREFIX = "tsne"


def show_whole_tsne(X, Y):
    data = StandardScaler().fit_transform(X)
    targets = np.argmax(Y, axis=1)
    points_transformed = (
        TSNE(n_components=2, perplexity=30, random_state=np.random.RandomState(0))
        .fit_transform(data)
        .T
    )
    points_transformed = np.swapaxes(points_transformed, 0, 1)
    show_scatterplot(points_transformed, targets)
    return points_transformed, targets


def show_tsne(
    model_name: str,
    epochs: int,
    X: np.ndarray,
    Y: np.ndarray,
    Y_predicted: Optional[np.ndarray] = None,
    init: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    data = scaler.fit_transform(X)
    targets = np.argmax(Y, axis=1)

    file_path = f"{TSNE_PATH_PREFIX}{model_name}_{epochs}.npy"

    if init is not None:
        tsne = TSNE(n_components=2, perplexity=30, init=init, random_state=0)
    else:
        tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    points_transformed = tsne.fit_transform(data).T
    points_transformed = np.swapaxes(points_transformed, 0, 1)
    np.save(file_path, points_transformed)

    show_scatterplot(points_transformed, targets, Y_predicted)

    return points_transformed, targets


def show_scatterplot(points_transformed, targets, Y_predicted=None):
    palette = sns.color_palette("bright", 10)
    fig, ax = plt.subplots(figsize=(10, 10))
    if Y_predicted is None:
        sns.scatterplot(
            x=points_transformed[:, 0],
            y=points_transformed[:, 1],
            hue=targets,
            legend="full",
            palette=palette,
            ax=ax,
        )
    else:
        Y_diff = targets - Y_predicted
        styles = np.where(Y_diff == 0, "Matched", "Mismatched")
        sns.scatterplot(
            x=points_transformed[:, 0],
            y=points_transformed[:, 1],
            hue=targets,
            style=styles,
            legend="full",
            palette=palette,
            ax=ax,
        )
    plt.show()


def process_activations(activations, Y_test, size):
    arr_activations = np.concatenate(activations, axis=0)
    arr_targets = np.concatenate([Y_test for _ in range(len(activations))], axis=0)

    points_transformed, targets = show_whole_tsne(arr_activations, arr_targets)

    points_lst = list(points_transformed.reshape(len(activations), size, 2))
    return points_lst, targets[:size]


def inter_epoch_evolution(points_lst, targets):
    points_by_digit = [[] for _ in range(10)]
    for n in range(10):
        for points in points_lst:
            ind = targets == n
            points_by_digit[n].append(points[ind])
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    for label, epoch_points in enumerate(points_by_digit):
        activation_lst = []
        for n in range(epoch_points[0].shape[0]):
            epoch_lst = [
                epoch_points[count][n, :] for count in range(len(epoch_points))
            ]
            activation_lst.append(epoch_lst)
        dfs_lst = []
        for activation_count, lst in enumerate(activation_lst, 1):
            dct = {
                f"activation_{activation_count}_epoch_{count * 20}": value
                for count, value in enumerate(lst)
            }
            df = pd.DataFrame(dct)
            dfs_lst.append(df)
        graph = nx.Graph()
        for df in dfs_lst:
            df_dense = df.corr("pearson")
            for edge, _ in df_dense.unstack().items():
                if int(edge[0].split("_")[-1]) + 20 == int(edge[1].split("_")[-1]):
                    graph.add_edge(*edge)
        c_dct = {
            f"activation_{activation_count}_epoch_{epoch_count * 20}": activation_value
            for epoch_count, epoch_value in enumerate(epoch_points)
            for activation_count, activation_value in enumerate(epoch_value, 1)
        }
        nodes = (
            pd.DataFrame(c_dct)
            .T.reset_index()
            .rename(columns={"index": "name", 0: "x", 1: "y"})
        )
        sources = [
            nodes[nodes["name"] == source].index[0] for source, _ in list(graph.edges)
        ]
        targets = [
            nodes[nodes["name"] == target].index[0] for _, target in list(graph.edges)
        ]
        edges = pd.DataFrame({"source": sources, "target": targets})
        hb = hammer_bundle(nodes, edges)
        hb.plot(x="x", y="y", figsize=(10, 10), ax=ax, alpha=0.7, label=label)
