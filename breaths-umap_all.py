# %%
# breaths-umap_all.py
#

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.colors
import matplotlib.pyplot as plt

import hdbscan

from utils.file import parse_birdname
from utils.umap import get_time_since_stim, loc_relative, plot_embedding_data

# %load_ext autoreload
# %autoreload 1
# %aimport utils.umap

# %%
# load umap, all_breaths data

embedding_name = "embedding003-insp"
fs = 44100

all_breaths_path = Path(rf"M:\public\Ciro\callback-breaths\umap-all_breaths\all_breaths.pickle")
all_trials_path = Path(r"./data/breath_figs-spline_fit/all_trials.pickle")

umap_pickle_path = Path(rf"M:\public\Ciro\callback-breaths\umap-all_breaths\{embedding_name}.pickle")

# breath data
print("loading all breaths data...")
with open(all_breaths_path, "rb") as f:
    all_breaths = pickle.load(f)
print("all breaths data loaded!")

# trial data
print("loading all trials data...")
with open(all_trials_path, "rb") as f:
    all_trials = pickle.load(f)
print("all trials data loaded!")

# UMAP
# note: ensure environment is EXACTLY the same as when the model was trained.
# otherwise, the model may not load.
print("loading umap embedding...")
with open(umap_pickle_path, "rb") as f:
    model = pickle.load(f)
print("umap embedding loaded!")

embedding = model.embedding_

model

# %%
# kwargs consistent across
scatter_kwargs = dict(
    s=.2,
    alpha=0.5,
)

set_kwargs = dict(
    xlabel="UMAP1",
    ylabel="UMAP2",
)

# %%
# add time since stim

all_breaths["time_since_stim_s"] = all_breaths.apply(
    get_time_since_stim,
    axis=1,
    all_trials=all_trials,
)

# %%
# take only type in embedding
ii_type = all_breaths["type"] == embedding_name.split("-")[-1]

other_breaths = all_breaths.loc[~ii_type]
all_breaths = all_breaths.loc[ii_type]

# %%
# indices for next breath

# example usage
ii_next = all_breaths.apply(
    lambda x: loc_relative(*x.name, df=other_breaths, i=1, field="index"),
    axis=1,
)

# %%
# PUTATIVE CALL

plot_embedding_data(
    embedding,
    embedding_name,
    all_breaths,
    plot_type="putative_call",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# AMPLITUDE

plot_embedding_data(
    embedding,
    embedding_name,
    all_breaths,
    plot_type="amplitude",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# DURATION

plot_embedding_data(
    embedding,
    embedding_name,
    all_breaths,
    plot_type="duration",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
    vmin=0,
    vmax=400,
    cmap_name="viridis",
)

# %%
# BREATHS SINCE LAST STIM

plot_embedding_data(
    embedding,
    embedding_name,
    all_breaths,
    plot_type="breaths_since_stim",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
    n_breaths=6,
)

# %%
# BY BIRD

plot_embedding_data(
    embedding,
    embedding_name,
    all_breaths,
    plot_type="bird_id",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# clustering

clusterer = hdbscan.HDBSCAN(
    metric="l1",
    min_cluster_size=130,
    min_samples=1,
    cluster_selection_method="leaf",
    gen_min_span_tree=True,
    cluster_selection_epsilon=0.2,
)

clusterer.fit(embedding)

plot_embedding_data(
    embedding=embedding,
    embedding_name=embedding_name,
    plot_type="clusters",
    clusterer=clusterer,
    set_kwargs=set_kwargs,
    scatter_kwargs=scatter_kwargs,
    set_bad=dict(c="k", alpha=1),
)

# %%
# highlight certain clusters
plot_embedding_data(
    embedding=embedding,
    embedding_name=embedding_name,
    plot_type="clusters",
    clusterer=clusterer,
    set_kwargs=set_kwargs,
    scatter_kwargs=scatter_kwargs,
    masked_clusters=[-1, 5, 10, 12, 13],
    set_bad=dict(c="k", alpha=1),
)

# %%
# plot traces by cluster

cluster_set_kwargs = dict(
    ylabel="amplitude",
    ylim=[-1.05, 0.05],
)

# =========PICK TRACE TYPE=========#
# any column in all_breaths

# trace_type = "breath_interpolated"
trace_type = "breath_norm"

# =========STACK TRACES=========#

# === use traces as-is (requires same length)
# traces = all_breaths[trace_type]

# === pad at end (keep alignment)
def do_pad(x, max_length):
    if len(x) > max_length:
        return x[:max_length]
    else:
        return np.pad(x, [0, max_length - len(x)])


# max_len = max(all_breaths[trace_type].apply(len))
max_len = int(0.4 * fs)
traces = all_breaths[trace_type].apply(do_pad, max_length=max_len)

# === align to insp offset - pad at beginning & end
# max_len_insp = max(all_breaths[trace_type].apply(len))
# traces = all_breaths.apply(
#     lambda trial: np.pad(
#         trial[trace_type], [max_len_insp - len(trial["insps_unpadded"]), 0]
#     ),
#     axis=1,
# )
# max_len = max(traces.apply(len))
# traces = traces.apply(lambda x: np.pad(x, [0, max_len - len(x)]))


# === align to insp onset
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

data = all_breaths.duration_s
violin_set_kwargs = dict(
    title="duration",
    ylabel="duration (s)",
    ylim=[-0.1, 0.7],
)

# data = all_breaths.amplitude
# violin_set_kwargs = dict(
#     title="amplitude",
#     ylabel="amplitude (normalized)",
# )

# data = all_breaths.stims_index
# violin_set_kwargs = dict(
#     title="breath segs since stim",
#     ylabel="breath segs since stim",
# )

cluster_data = {
    i_cluster: data[(clusterer.labels_ == i_cluster) & data.notna()]
    for i_cluster in np.unique(clusterer.labels_)
}

labels, data = cluster_data.keys(), cluster_data.values()


fig, ax = plt.subplots()

ax.violinplot(data, showextrema=False)
ax.set_xticks(ticks=range(1, 1 + len(labels)), labels=labels)

ax.set(xlabel="cluster", **violin_set_kwargs)


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
