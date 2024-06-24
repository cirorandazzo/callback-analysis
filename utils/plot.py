# ./utils/plot.py
# 2024.05.14 CDR
# 
# Plotting functions
# 

callback_raster_stim_kwargs = dict(color='red', alpha=0.5)
callback_raster_call_kwargs = dict(color='black', alpha=0.5)

def plot_callback_raster(
    data,
    ax=None,
    title = None,
    xlabel='Time since stimulus onset (s)',
    ylabel='Trial #',
    plot_stims = True,
    show_legend = True,
    call_kwargs = callback_raster_call_kwargs,
    stim_kwargs = callback_raster_stim_kwargs,
    force_yticks_int = True,
    kwargs={},
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()


    # Construct patch collections for call boxes
    stim_boxes = []
    call_boxes = []

    for i, trial_i in enumerate(data.index):
        stim_boxes.append(Rectangle((0, i), data.loc[trial_i, 'stim_duration_s'], 1))

        calls = data.loc[trial_i, 'call_times_stim_aligned']
        call_boxes += [Rectangle((st, i), en-st, 1) for st,en in calls]

    call_patches = PatchCollection(call_boxes, **call_kwargs)
    ax.add_collection(call_patches)
    
    # "fakers" are proxy artists, only created for legend
    if plot_stims:
        stim_patches = PatchCollection(stim_boxes, **stim_kwargs)
        ax.add_collection(stim_patches)
        stim_faker = Rectangle([0,0], 0, 0, **stim_kwargs)

    call_faker = Rectangle([0,0], 0, 0, **call_kwargs)

    if show_legend:
        ax.legend((stim_faker, call_faker), ('Stimulus', 'Calls'))

    ax.autoscale_view()

    if force_yticks_int:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        **kwargs,
    )

    return ax

def plot_group_hist(
        df,
        field,
        grouping_level,
        group_colors=None,
        density=False,
        ax=None,
        ignore_nan=False,
        alphabetize_legend=False,
        alt_labels=None,
        histogram_kwargs={},
        stair_kwargs={},
):
    """
    TODO: add new parameters to documentation
    
    Plot overlaid histograms for 1 or more groups, given a DataFrame, a fieldname to plot, and an multi-index level by which to group.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    field : str
        The name of the column in `df` for which the histogram is plotted.
    grouping_level : int or str
        Multi-Index level by which to group. Can be int or string (see pandas.Index.unique)
    group_colors : dict or iterable, optional
        A dictionary mapping group names to colors or an iterable of colors for each group.
        If not provided, matplotlib default colors will be used.
    density : bool, optional
        If True, plot density (ie, normalized to count) instead of raw count for each group.
    ax : matplotlib AxesSubplot object, optional
        The axes on which to draw the plot. If not provided, a new figure and axes will be created.
    ignore_nan: bool, optional
        If True, cut nan values out of plotted data. Else, np.histogram throws ValueError if nan values are in data to plot.

    Returns:
    --------
    ax : matplotlib Axes
        The axes on which the plot is drawn.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if ax is None:
        fig, ax = plt.subplots()

    # use same binning for all groups
    if 'range' not in histogram_kwargs.keys():
        histogram_kwargs['range'] = (np.min(df[field]), np.max(df[field]))

    for i_grp, groupname in enumerate(df.index.unique(level=grouping_level)):
        group_data = np.array(df.loc[
            df.index.get_level_values(level=grouping_level) == groupname,
            field
        ])

        if ignore_nan:
            group_data = group_data[~np.isnan(group_data)]

        hist, edges = np.histogram(group_data, **histogram_kwargs)

        # histogram: plot density vs count
        if density:
            hist = hist/len(group_data)
            ax.set(ylabel='density')
        else:
            ax.set(ylabel='count')

        # get group colors
        if isinstance(group_colors, dict):
            color = group_colors[groupname]
        elif isinstance(group_colors, (list, tuple)):
            color = group_colors[i_grp]
        else:
            color = f'C{i_grp}'

        if alt_labels is None:
            label = f'{groupname} ({len(group_data)})'
        else:
            label = f'{alt_labels[groupname]} ({len(group_data)})'

        ax.stairs(hist, edges, label=label, color=color, **stair_kwargs.get(groupname, {}))

    ax.legend()
    if alphabetize_legend:
        handles, labels = plt.gca().get_legend_handles_labels()

        sort_ii = np.argsort(labels)

        plt.legend([handles[i] for i in sort_ii], [labels[i] for i in sort_ii])

    return ax