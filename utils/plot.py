# ./utils/plot.py
# 2024.05.14 CDR
# 
# Plotting functions
# 

callback_raster_stim_kwargs = dict(color='red', alpha=0.5)
callback_raster_call_kwargs = dict(color='black', alpha=0.5)
day_colors = {1: "#a2cffe", 2: "#840000"}

def plot_callback_raster(
    data,
    ax=None,
    title = None,
    xlabel='Time since stimulus onset (s)',
    ylabel='Trial #',
    plot_stim_blocks = True,
    show_legend = True,
    y_offset=0,
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
        height = i + y_offset

        stim_boxes.append(Rectangle((0, height), data.loc[trial_i, 'stim_duration_s'], 1))

        calls = data.loc[trial_i, 'call_times_stim_aligned']
        call_boxes += [Rectangle((st, height), en-st, 1) for st,en in calls]

    call_patches = PatchCollection(call_boxes, **call_kwargs)
    ax.add_collection(call_patches)
    
    call_faker = Rectangle([0,0], 0, 0, **call_kwargs)
    legend_labels = ['Calls']
    legend_entries = [call_faker]

    # "fakers" are proxy artists, only created for legend
    if plot_stim_blocks:
        stim_patches = PatchCollection(stim_boxes, **stim_kwargs)
        ax.add_collection(stim_patches)

        # add stimulus to legend
        stim_faker = Rectangle([0,0], 0, 0, **stim_kwargs)
        legend_labels.append('Stimulus')
        legend_entries.append(stim_faker)

    if show_legend:
        ax.legend(legend_entries, legend_labels)

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
 
 
def plot_callback_raster_multiblock(
        data,
        ax=None,
        plot_hlines=True,
        show_block_axis=True,
        show_legend=False,
        xlim=[-0.1, 3],
        call_kwargs = callback_raster_call_kwargs,
        stim_kwargs = callback_raster_stim_kwargs,
        title = None,
):
    '''
    Plot multiple blocks on the same axis with horizontal lines separating.

    Notes: xlim sets ax.xlim, but also xmin and xmax for horizontal block lines.
    '''
    
    blocks = list(set(data.index.get_level_values(0)))  # get & order blocks
    blocks.sort()

    y_offset = 0
    block_locs = []

    for block in blocks:
        block_locs.append(y_offset)
        data_block = data.loc[block]

        plot_callback_raster(
            data_block,
            ax=ax,
            y_offset=y_offset,
            plot_stim_blocks=True,
            show_legend=show_legend,
            call_kwargs=call_kwargs,
            stim_kwargs=stim_kwargs,
        )

        y_offset += len(data_block)


    if plot_hlines:
        ax.hlines(y=block_locs, xmin=xlim[0], xmax=xlim[1], colors="k", linestyles="solid", linewidths=.5)

    if show_block_axis:
        block_axis = ax.secondary_yaxis(location='right')
        block_axis.set(
            ylabel='Block',
            yticks=block_locs,
            yticklabels=blocks,
        )

    ax.set(
        title=title,
        xlim=xlim,
        ylim=[0, len(data)],
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


def plot_violins_by_block(
    bird_data,
    field,
    days,
    day_colors,
    ax=None,
    width=0.75,
    dropna=False,
):
    '''
    TODO: document utils.plot.plot_violins_by_block
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    for day in days:
        data = bird_data.loc[day]

        if dropna:
            data.dropna(inplace=True)

        blocks = list(set(data.index.get_level_values(0)))  # get blocks

        values_by_block = [list(data.loc[bl, field]) for bl in blocks]

        parts = ax.violinplot(
            values_by_block,
            blocks,
            widths=width,
            showmedians=True,
            showextrema=False,
            # side='low',  # TODO: use matplotlib 3.9 for 'side' parameter
        )

        # customize violin bodies
        for pc in parts['bodies']:
            pc.set(
                facecolor=day_colors[day],
                edgecolor=day_colors[day],
                alpha=0.5,
            )

            # clip half of polygon
            m = np.mean(pc.get_paths()[0].vertices[:, 0])

            if day==1:
                lims = [-np.inf, m]
            elif day==2:
                lims = [m, np.inf]
            else:
                raise Exception('Unknown day.')
            
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], lims[0], lims[1])

        # customize median bars
        parts['cmedians'].set(
            edgecolor=day_colors[day],
        )

    return ax


def plot_pre_post(
    df_day,
    fieldname,
    ax=None,
    color='k',
    bird_id_fieldname="birdname",
    plot_kwargs={},
):
    '''
    TODO: documentation

    Given `df_day` which has field `fieldname`, plot pre/post type line plot. `color` can be a pd.DataFrame with bird names as index containing color info for every bird in column "color", or a single matplotlib color
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd

    if ax is None:
        fig, ax = plt.subplots()

    all_birds = np.unique(df_day.index.get_level_values(bird_id_fieldname))

    for bird in all_birds:
        bird_data = df_day.loc[bird]

        if isinstance(color, pd.DataFrame):
            c = color.loc[bird, "color"]
        else:
            c = color  # any matplotlib color formats should work here.
            
        ax.plot(
            bird_data.index,  # days
            bird_data[fieldname],
            color=c,
            **plot_kwargs,
        )

    return ax