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

from utils.umap import (
    plot_cluster_traces_pipeline,
    plot_embedding_data,
)

# %load_ext autoreload
# %autoreload 1
# %aimport utils.umap

# %%
# load umap, all_trials data

embedding_name = "embedding0"
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
    s=4,
    alpha=0.8,
)

set_kwargs = dict(
    xlabel="UMAP1",
    ylabel="UMAP2",
)

# %%
# PUTATIVE CALL

plot_embedding_data(
    embedding,
    embedding_name,
    all_trials,
    plot_type="putative_call",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# INSP OFFSET

offsets_ms = all_insps[1, :] / fs * 1000

plot_embedding_data(
    embedding,
    embedding_name,
    all_trials,
    plot_type="insp_offset",
    fs=44100,
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# INSP ONSET

onsets_ms = all_insps[0, :] / fs * 1000

plot_embedding_data(
    embedding,
    embedding_name,
    all_trials,
    plot_type="insp_onset",
    fs=44100,
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# INSP DURATION

duration_ms = (all_insps[1, :] - all_insps[0, :]) / fs * 1000

plot_embedding_data(
    embedding,
    embedding_name,
    all_trials,
    plot_type="duration",
    vmax = duration_ms.max() + 50,
    fs=44100,
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# BY BIRD

plot_embedding_data(
    embedding,
    embedding_name,
    all_trials,
    plot_type="bird_id",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# clustering

clusterer = hdbscan.HDBSCAN(
    metric="l1",
    min_cluster_size=200,
    min_samples=1,
    cluster_selection_method="leaf",
    gen_min_span_tree=True,
    cluster_selection_epsilon=0.5,
)

clusterer.fit(embedding)

ax_clusters = plot_embedding_data(
    embedding,
    embedding_name,
    all_trials,
    plot_type="clusters",
    clusterer=clusterer,
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

cluster_cmap = ax_clusters.collections[0].get_cmap()

# %%
# highlight certain clusters

plot_embedding_data(
    embedding,
    embedding_name,
    all_trials,
    plot_type="clusters",
    clusterer=clusterer,
    highlighted_clusters=[-1, 10, 12, 20],
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# plot traces by cluster
plt.close("all")

cluster_set_kwargs = dict(
    ylabel="amplitude",
    ylim=[-1.05, 0.05],
)

# =========SELECTIONS=========#
# which trace to plot : select one trace_kwargs dict

trace_kwargs = dict(
    trace_type="insps_interpolated",
    set_kwargs={**cluster_set_kwargs, "xlim": [-0.05, 1.05]},
)

# trace_kwargs = dict(
#     trace_type="insps_unpadded",
#     aligned_to="insp onset",
#     padding_kwargs=dict(pad_method="end", max_length=None),
#     set_kwargs={**cluster_set_kwargs, },
# )

# trace_kwargs = dict(
#     trace_type="insps_unpadded",
#     aligned_to="insp offset",
#     padding_kwargs=dict(pad_method="beginning", max_length=None),
#     set_kwargs={**cluster_set_kwargs, },
# )

# trace_kwargs = dict(
#     trace_type="insps_padded_right_zero",
#     aligned_to="insp onset",
#     padding_kwargs=dict(pad_method="end", max_length=None),
#     set_kwargs={**cluster_set_kwargs},
# )

# TODO: breath_norm with alignments:
# insp onset
# insp offset
# stim onset

# need window, defined in insp_categories.py
buffer_ms = 500
buffer_fr = int(buffer_ms * fs / 1000) + 1
all_insps = np.vstack(all_trials["ii_first_insp"]).T
window = np.array((all_insps.min() - buffer_fr, all_insps.max() + buffer_fr))

# trace_kwargs = dict(  # this was an accident lol 
#     trace_type="breath_first_cycle",
#     aligned_to="insp onset",
#     padding_kwargs=dict(
#         pad_method="aligned",
#         max_length=abs(window),
#         i_align=all_insps[0, :] - window[0],
#     ),
#     set_kwargs={**cluster_set_kwargs, "ylim": [-1.05, 6], "xlim": [-100, 500]},
# )

trace_kwargs = dict(
    trace_type="breath_first_cycle",
    aligned_to="stim onset",
    padding_kwargs=dict(
        aligned_at=abs(window[0]),
    ),
    set_kwargs={**cluster_set_kwargs, "ylim": [-1.05, 6], "xlim": [-100, 500]},
)


# which trials to plot: select "all", "call", or "no call"
select = "all"
# select = "call"
# select = "no call"


axs_cluster_traces = plot_cluster_traces_pipeline(
    **trace_kwargs,
    df=all_trials,
    fs=fs,
    clusterer=clusterer,
    select=select,
    cluster_cmap=cluster_cmap,
)

# %%
# 3d with call duration

embedding_plus = np.vstack([embedding.T, duration_ms])

x, height, z = np.split(embedding_plus, 3, axis=0)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(x, height, z)

ax.set(xlabel="UMAP1", ylabel="UMAP2", zlabel="insp duration (ms)")


# %%
# VIOLIN PLOT BY CLUSTER

# data = duration_ms
# set_kwargs = dict(
#     ylabel="insp duration (ms)",
#     title="insp duration",
# )

# data = onsets_ms
# set_kwargs = dict(
#     title="insp onset (stim-aligned)",
#     ylabel="insp onset (ms)",
# )

data = offsets_ms
set_kwargs = dict(
    title="insp offset (stim-aligned)",
    ylabel="insp offset (ms)",
    ylim=[-20, 510],
)

# data = -1 * magnitudes["mag_insp"]  # note: load from insp_categories.py
# set_kwargs = dict(
#     title="insp magnitude",
#     ylabel="insp magnitude (normalized)",
# )

# data = magnitudes["mag_exp"]  # note: load from insp_categories.py
# set_kwargs = dict(
#     title="exp magnitude (300ms post-insp)",
#     ylabel="exp magnitude (normalized)",
# )

cluster_data = {
    i_cluster: data[(clusterer.labels_ == i_cluster)]
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
    i_cluster: sum(all_trials["putative_call"] & (clusterer.labels_ == i_cluster))
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
