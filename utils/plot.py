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