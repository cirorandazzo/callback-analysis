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

# kwargs consistenc across
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

bin_width_f = 0.02 * fs

fig, ax = plt.subplots()

edges = (
    np.arange(
        all_insps.ravel().min(), all_insps.ravel().max() + bin_width_f, bin_width_f
    )
    / fs
    * 1000
)

hist, x_edges, y_edges, im = ax.hist2d(
    x=onsets_ms, y=offsets_ms, cmap="hot", bins=edges
)

ax.axis("equal")
plt.box(False)

ax.set(
    xlabel="onsets (ms, stim-aligned)",
    ylabel="offsets (ms, stim-aligned)",
    title="First inspiration timing",
)

fig.colorbar(im, ax=ax, label="Number of breaths")
