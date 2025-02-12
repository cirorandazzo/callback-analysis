# %%
# breaths-umap.py
#
# Trying to tease out categories of breath responses in callback experiments from a UMAP embedding
#

import glob
from itertools import product
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.colors
import matplotlib.pyplot as plt

import umap

import hdbscan

# %%
# load umap, all_trials data

embedding_name = "embedding34"
fs = 44100

all_trials_path = Path(r".\data\umap\all_trials.pickle")

umap_pickle_path = Path(rf".\data\umap\{embedding_name}.pickle")


# ALL TRIALS
with open(all_trials_path, "rb") as f:
    all_trials = pickle.load(f)

all_insps = np.vstack(all_trials["ii_first_insp"]).T

# UMAP
with open(umap_pickle_path, "rb") as f:
    data = pickle.load(f)

model = data["model"]
embedding = data["embedding"]

del data

model

# %%

# kwargs consistent across
scatter_kwargs = dict(
    x=embedding[:, 0],
    y=embedding[:, 1],
    s=4,
    alpha=0.8,
)

set_kwargs = dict(
    xlabel="UMAP1",
    ylabel="UMAP2",
)

# %%
# PUTATIVE CALL

fig, ax = plt.subplots()

title = f"{embedding_name}: putative call"

# PLOT
colors = np.array(all_trials["putative_call"]).astype(int)
sc = ax.scatter(
    **scatter_kwargs,
    c=colors,
    cmap="Dark2",
)

ax.set(
    **set_kwargs,
    title=title,
)

handles, labels = sc.legend_elements()

label_map = {
    "$\\mathdefault{0}$": "No call",
    "$\\mathdefault{1}$": "Call",
}

legend = ax.legend(handles=handles, labels=[label_map[x] for x in labels])

# %%
# INSP OFFSET

fig, ax = plt.subplots()

title = f"{embedding_name}: insp OFFSET"

offsets_ms = all_insps[1, :] / fs * 1000

sc = ax.scatter(
    **scatter_kwargs,
    c=offsets_ms,
    cmap="magma_r",
)

ax.set(
    **set_kwargs,
    title=title,
)

cbar = fig.colorbar(sc, label="insp offset (ms, stim-aligned)")

# %%
# INSP ONSET

fig, ax = plt.subplots()

title = f"{embedding_name}: insp ONSET"

onsets_ms = all_insps[0, :] / fs * 1000

sc = ax.scatter(
    **scatter_kwargs,
    c=onsets_ms,
    cmap="RdYlGn",
)

ax.set(
    **set_kwargs,
    title=title,
)

cbar = fig.colorbar(sc, label="insp onset (ms, stim-aligned)")

# %%
# INSP DURATION

fig, ax = plt.subplots()

title = f"{embedding_name}: insp DURATION"

duration_ms = (all_insps[1, :] - all_insps[0, :]) / fs * 1000

sc = ax.scatter(
    **scatter_kwargs,
    c=duration_ms,
    cmap="cool",
)

ax.set(
    **set_kwargs,
    title=title,
)

cbar = fig.colorbar(sc, label="insp duration (ms)")

# %%
# BY BIRD

fig, ax = plt.subplots()

title = f"{embedding_name}: bird id"

birdnames = pd.Categorical(all_trials.index.get_level_values("birdname"))

# cmap =
# colors = [cmap[bird] for bird in ]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["#14342B", "#F5BB00", "#FF579F"]
)

sc = ax.scatter(
    **scatter_kwargs,
    c=birdnames.codes,
    cmap=cmap,
)

ax.set(
    **set_kwargs,
    title=title,
)

handles, labels = sc.legend_elements()

label_map = {
    "$\\mathdefault{" + str(i) + "}$": birdnames.categories[i]
    for i in range(len(birdnames.categories))
}

ax.legend(handles=handles, labels=[label_map[x] for x in labels])

# %%
# clustering

clusterer = hdbscan.HDBSCAN(
    metric="l1",
    min_cluster_size=20,
    min_samples=1,
    cluster_selection_method="leaf",
    gen_min_span_tree=True,
    cluster_selection_epsilon=0.5,
)

clusterer.fit(embedding)

cluster_embeddings = {
    i_cluster: embedding[clusterer.labels_ == i_cluster]
    for i_cluster in np.unique(clusterer.labels_)
}

fig, ax = plt.subplots()

title = f"{embedding_name}: hdbscan clustering"

cmap = plt.get_cmap("jet", len(cluster_embeddings.keys()))

for i_cluster, cluster_points in cluster_embeddings.items():
    if i_cluster == -1:
        color = "k"
    else:
        color = cmap(i_cluster)

    ax.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"{i_cluster}",
        facecolor=color,
        s=5,
        alpha=0.5,
    )

ax.set(
    **set_kwargs,
    title=title,
)

ax.legend()

# %%
# plot traces by cluster

cluster_set_kwargs = dict(
    ylabel="amplitude",
    ylim=[-1.05, 0.05],
)

# =========PICK TRACE TYPE=========#
# any column in all_trials

# trace_type = "insps_interpolated"
# trace_type = "insps_padded_right_zero"
# trace_type = "breath_first_cycle"
# trace_type = "breath_first_cycle_unpadded"
trace_type = "insps_unpadded"


# =========STACK TRACES=========#

# use traces as-is (requires same length)
# traces = all_trials[trace_type]

# pad at end (keep alignment)
max_len = max(all_trials[trace_type].apply(len))
traces = all_trials[trace_type].apply(lambda x: np.pad(x, [0, max_len - len(x)]))

# align to insp offset - pad at beginning & end
# max_len_insp = max(all_trials[trace_type].apply(len))
# traces = all_trials.apply(
#     lambda trial: np.pad(
#         trial[trace_type], [max_len_insp - len(trial["insps_unpadded"]), 0]
#     ),
#     axis=1,
# )
# max_len = max(traces.apply(len))
# traces = traces.apply(lambda x: np.pad(x, [0, max_len - len(x)]))

# stack all traces in a numpy array
all_traces = np.vstack(traces)


# =========PLOT SUBSET=========#

label = "all"
cluster_data = {
    i_cluster: all_traces[(clusterer.labels_ == i_cluster)]
    for i_cluster in np.unique(clusterer.labels_)
}

# label = "call"
# cluster_traces = {
#     i_cluster: all_traces[
#         (clusterer.labels_ == i_cluster)
#         & np.array(all_trials["putative_call"]).astype(bool)
#     ]
#     for i_cluster in np.unique(clusterer.labels_)
# }

# label = "no call"
# cluster_traces = {
#     i_cluster: all_traces[
#         (clusterer.labels_ == i_cluster)
#         & ~np.array(all_trials["putative_call"]).astype(bool)
#     ]
#     for i_cluster in np.unique(clusterer.labels_)
# }

# =========GET X=========#

# insp_interpolated x: [0, 1]
x = np.linspace(0, 1, all_traces.shape[1])
cluster_set_kwargs["xlabel"] = "normalized inspiration duration"

# insp offset aligned (ms)
# x = (np.arange(max_len) - max_len_insp) / fs * 1000
# cluster_set_kwargs["xlabel"] = "time (ms, insp offset-aligned)"

# onset aligned (ms)
# x = np.arange(all_traces.shape[1]) / fs * 1000
# cluster_set_kwargs["xlabel"] = "time (ms, insp onset aligned)"

# stim-aligned (ms): requires "window" var from insp_categories
# buffer_ms = 500
# buffer_fr = int(buffer_ms * fs / 1000) + 1
# all_insps = np.vstack(all_trials["ii_first_insp"]).T
# window = (all_insps.min() - buffer_fr, all_insps.max() + buffer_fr)
# x = np.arange(*window) / fs * 1000
# cluster_set_kwargs["xlabel"] = "time (ms, stim-aligned)"


# =========PLOT=========#

for i_cluster, traces in cluster_data.items():
    fig, ax = plt.subplots()

    if i_cluster == -1:
        title_color = "k"
    else:
        title_color = cmap(i_cluster)

    # plot trials
    ax.plot(x, traces.T, color="k", alpha=0.7, linewidth=0.5)

    # plot mean
    ax.plot(x, traces.T.mean(axis=1), color="r", linewidth=1)

    # title n
    if label == "all":
        n = sum(clusterer.labels_ == i_cluster)
    else:
        n = f"{traces.shape[0]}/{sum(clusterer.labels_ == i_cluster)}"

    ax.set_title(
        f"cluster {i_cluster} traces {label} (n={n})",
        color=title_color,
    )

    ax.set(**cluster_set_kwargs)

# %%
# 3d with call duration

embedding_plus = np.vstack([embedding.T, duration_ms])

x, y, z = np.split(embedding_plus, 3, axis=0)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(x, y, z)

ax.set(xlabel="UMAP1", ylabel="UMAP2", zlabel="insp duration (ms)")


# %%
# VIOLIN PLOT BY CLUSTER

data = duration_ms
set_kwargs = dict(
    ylabel="insp duration (ms)",
    title="insp duration",
    xlabel="cluster",
)

# data = onsets_ms
# set_kwargs = dict(
#     title="insp onset (stim-aligned)",
#     ylabel="insp onset (ms)",
#     xlabel="cluster",
# )

# data = offsets_ms
# set_kwargs = dict(
#     title="insp offset (stim-aligned)",
#     ylabel="insp offset (ms)",
#     xlabel="cluster",
# )

cluster_data = {
    i_cluster: data[(clusterer.labels_ == i_cluster)]
    for i_cluster in np.unique(clusterer.labels_)
}

labels, data = cluster_data.keys(), cluster_data.values()


fig, ax = plt.subplots()

ax.violinplot(data, showextrema=False)
ax.set_xticks(ticks=range(1, 1 + len(labels)), labels=labels)

ax.set(**set_kwargs)
