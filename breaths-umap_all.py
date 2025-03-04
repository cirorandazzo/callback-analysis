# %%
# breaths-umap_all.py
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

from utils.file import parse_birdname

# %%
# load umap, all_breaths data

embedding_name = "embedding035-exp"
fs = 44100

all_breaths_path = Path(r".\data\umap-all_breaths\all_breaths.pickle")
all_trials_path = Path(r"./data/breath_figs-spline_fit/all_trials.pickle")

umap_pickle_path = Path(rf".\data\umap-all_breaths\{embedding_name}.pickle")

# breath data
with open(all_breaths_path, "rb") as f:
    all_breaths = pickle.load(f)

    # take only type in embedding
    ii_type = all_breaths["type"] == embedding_name.split("-")[-1]
    all_breaths = all_breaths.loc[ii_type]

# trial data
with open(all_trials_path, "rb") as f:

    all_trials = pickle.load(f)

# UMAP
with open(umap_pickle_path, "rb") as f:
    model = pickle.load(f)

embedding = model.embedding_

model


# %%
# add time since stim

def get_time_since_stim(x, all_trials):
    if pd.isna(x.stims_index):
        return pd.NA
    else:
        return (x.start_s - all_trials.loc[(x.name[0], x.stims_index), "trial_start_s"])


all_breaths["time_since_stim_s"] = all_breaths.apply(
    get_time_since_stim,
    axis=1,
    all_trials=all_trials,
)


# %%

# kwargs consistent across
scatter_kwargs = dict(
    x=embedding[:, 0],
    y=embedding[:, 1],
    s=.2,
    alpha=0.5,
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
colors = np.array(all_breaths["putative_call"]).astype(int)
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
# AMPLITUDE

fig, ax = plt.subplots()

title = f"{embedding_name}: AMPLITUDE"

sc = ax.scatter(
    **scatter_kwargs,
    c=all_breaths.amplitude,
    cmap="magma_r",
)

ax.set(
    **set_kwargs,
    title=title,
)

cbar = fig.colorbar(sc, label="amplitude (normalized)")

# %%
# DURATION

fig, ax = plt.subplots()

title = f"{embedding_name}: DURATION"

sc = ax.scatter(
    **scatter_kwargs,
    c=all_breaths.duration_s,
    cmap="magma_r",
    vmax=1.0,  #  >1s all colored the same
)

ax.set(
    **set_kwargs,
    title=title,
)

cbar = fig.colorbar(sc, label="duration (s)")

# %%
# TIME SINCE STIM

fig, ax = plt.subplots()

n_since_stim = all_breaths["trial_index"].fillna(-1)

n_breaths = 8  # 7+ all colored the same
cmap = plt.get_cmap("viridis_r", n_breaths)
cmap.set_bad("k")


title = f"{embedding_name}: BREATHS SINCE STIM"
sc = ax.scatter(
    **scatter_kwargs,
    c=np.ma.masked_equal(n_since_stim, -1),
    cmap=cmap,
    vmin=0,
    vmax=n_breaths,
)

ax.set(
    **set_kwargs,
    title=title,
)

cbar = fig.colorbar(sc, label="breaths since stim")

ticks = [str(int(r)) for r in cbar.get_ticks()]
ticks[-1] = ""
ticks[-2] = f"{ticks[-2]}+"
cbar.set_ticklabels(ticks)

# %%
# BY BIRD

fig, ax = plt.subplots()

title = f"{embedding_name}: bird id"

birdnames = pd.Categorical(all_breaths.apply(lambda x: parse_birdname(x.name[0]), axis=1))

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
    min_samples=10,
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

# legend outside plot bounds
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
# highlight certain clusters

# to_highlight = [1,5,6,10,13,14,15]
# to_highlight = [2,3,4,7,8,9,11,12]
to_highlight = [0, -1]

fig, ax = plt.subplots()

title = f"{embedding_name}: hdbscan clustering"

cmap = plt.get_cmap("jet", len(cluster_embeddings.keys()))

for i_cluster, cluster_points in cluster_embeddings.items():
    if i_cluster in to_highlight:
        color = cmap(i_cluster)
    else:
        color = "k"

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

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
# plot traces by cluster

cluster_set_kwargs = dict(
    ylabel="amplitude",
    ylim=[-1.05, 6.7],
)

# =========PICK TRACE TYPE=========#
# any column in all_breaths

# trace_type = "breath_interpolated"
trace_type = "breath_norm"

# =========STACK TRACES=========#

# use traces as-is (requires same length)
# traces = all_breaths[trace_type]

# pad at end (keep alignment)
def do_pad(x, max_length):
    if len(x) > max_length:
        return x[:max_length]
    else:
        return np.pad(x, [0, max_length - len(x)])


# max_len = max(all_breaths[trace_type].apply(len))
max_len = int(0.7 * fs)
traces = all_breaths[trace_type].apply(do_pad, max_length=max_len)

# align to insp offset - pad at beginning & end
# max_len_insp = max(all_breaths[trace_type].apply(len))
# traces = all_breaths.apply(
#     lambda trial: np.pad(
#         trial[trace_type], [max_len_insp - len(trial["insps_unpadded"]), 0]
#     ),
#     axis=1,
# )
# max_len = max(traces.apply(len))
# traces = traces.apply(lambda x: np.pad(x, [0, max_len - len(x)]))


# align to insp onset
# def get_padded(trial, pre_frames, post_frames):
#     insp_on = trial["ii_first_insp_window_aligned"][0]

#     st = insp_on - pre_frames
#     en = insp_on + post_frames

#     assert (st > 0)
#     assert (en < len(trial["breath_norm"]))

#     return trial["breath_norm"][st:en]


# pre_frames = int(0.5 * fs)
# post_frames = int(0.5 * fs)

# traces = all_breaths.apply(
#     get_padded,
#     args=[pre_frames, post_frames],
#     axis=1
# )

# stack all traces in a numpy array
all_traces = np.vstack(traces)


# =========PLOT SUBSET=========#

label = "all"
cluster_data = {
    i_cluster: all_traces[(clusterer.labels_ == i_cluster)]
    for i_cluster in np.unique(clusterer.labels_)
}

# label = "call"
# cluster_data = {
#     i_cluster: all_traces[
#         (clusterer.labels_ == i_cluster)
#         & np.array(all_breaths["putative_call"]).astype(bool)
#     ]
#     for i_cluster in np.unique(clusterer.labels_)
# }

# label = "no call"
# cluster_data = {
#     i_cluster: all_traces[
#         (clusterer.labels_ == i_cluster)
#         & ~np.array(all_breaths["putative_call"]).astype(bool)
#     ]
#     for i_cluster in np.unique(clusterer.labels_)
# }

# =========GET X=========#

# insp_interpolated x: [0, 1]
# x = np.linspace(0, 1, all_traces.shape[1])
# cluster_set_kwargs["xlabel"] = "normalized duration"

# insp offset aligned (ms)
# x = (np.arange(max_len) - max_len_insp) / fs * 1000
# cluster_set_kwargs["xlabel"] = "time (ms, insp offset-aligned)"

# onset aligned (ms)
x = np.arange(all_traces.shape[1]) / fs * 1000
cluster_set_kwargs["xlabel"] = "time (ms, onset aligned)"

# stim-aligned (ms): requires "window" var from insp_categories
# buffer_ms = 500
# buffer_fr = int(buffer_ms * fs / 1000) + 1
# all_insps = np.vstack(all_breaths["ii_first_insp"]).T
# window = (all_insps.min() - buffer_fr, all_insps.max() + buffer_fr)
# x = np.arange(*window) / fs * 1000
# cluster_set_kwargs["xlabel"] = "time (ms, stim-aligned)"

# insp onset aligned with pre time (ms):
# x = np.linspace(-1 * pre_frames, post_frames, all_traces.shape[1])
# cluster_set_kwargs["xlabel"] = "time (ms, insp onset-aligned)"

# =========PLOT=========#

for i_cluster, traces in cluster_data.items():
    fig, ax = plt.subplots()

    if i_cluster == -1:
        title_color = "k"
    else:
        title_color = cmap(i_cluster)

    # plot trials
    ax.plot(
        x,
        traces.T,
        color="k",
        alpha=0.2,
        linewidth=0.5,
    )

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
# VIOLIN PLOT BY CLUSTER

# data = all_breaths.duration_s
# set_kwargs = dict(
#     title="exp duration",
#     ylabel="exp duration (s)",
#     ylim=[-0.1, 0.7],
# )

# data = all_breaths.amplitude
# set_kwargs = dict(
#     title="amplitude",
#     ylabel="amplitude (normalized)",
# )

data = all_breaths.stims_index
set_kwargs = dict(
    title="breath segs since stim",
    ylabel="breath segs since stim",
)

cluster_data = {
    i_cluster: data[(clusterer.labels_ == i_cluster) & data.notna()]
    for i_cluster in np.unique(clusterer.labels_)
}

labels, data = cluster_data.keys(), cluster_data.values()


fig, ax = plt.subplots()

ax.violinplot(data, showextrema=False)
ax.set_xticks(ticks=range(1, 1 + len(labels)), labels=labels)

ax.set(xlabel="cluster", **set_kwargs)


# %%
# PUTATIVE CALL PERCENTAGE

cluster_data = {
    i_cluster: sum(all_breaths["putative_call"] & (clusterer.labels_ == i_cluster))
    / sum((clusterer.labels_ == i_cluster))
    for i_cluster in np.unique(clusterer.labels_)
}

fig, ax = plt.subplots()

clusters, heights = cluster_data.keys(), cluster_data.values()

ax.bar(clusters, heights)
ax.set_xticks(list(clusters))

ax.set(
    xlabel="cluster",
    ylabel="% of trials with call",
    title="putative call pct",
)

# %%
# CLUSTER SIZE

cluster_data = {
    i_cluster: sum((clusterer.labels_ == i_cluster))
    for i_cluster in np.unique(clusterer.labels_)
}

fig, ax = plt.subplots()

clusters, heights = cluster_data.keys(), cluster_data.values()

ax.bar(clusters, heights)
ax.set_xticks(list(clusters))

ax.set(
    xlabel="cluster",
    ylabel="count (# trials)",
    title="cluster size",
)
