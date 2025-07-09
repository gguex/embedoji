
# ------------------------------------------
# ---- INTERFACE
# ------------------------------------------

# ---- Plot functions 

# Plot 2D
def plot_res_2d(df, var, dim1, dim2, color_by):
    fig = px.scatter(df, x=f"MDS_{dim1}", y=f"MDS_{dim2}",
                    size="n_tokens", color=color_by,
                    hover_name="speaker_name", 
                    hover_data=["id", "affair_id", "faction", "canton", "role"], 
                    size_max=15,
                    color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig.update_xaxes(title_text=f"Factorial axis {dim1 + 1} : {var[dim1]:.2%}")
    fig.update_yaxes(title_text=f"Factorial axis {dim2 + 1} : {var[dim2]:.2%}")
    fig['layout']['uirevision'] = 'true'
    return fig

# Plot 3D
def plot_res_3d(df, var, dim1, dim2, dim3, color_by):
    fig = px.scatter_3d(df, 
                        x=f"MDS_{dim1}", 
                        y=f"MDS_{dim2}", 
                        z=f"MDS_{dim3}",
                    size="n_tokens", color=color_by,
                    hover_name="speaker_name", 
                    hover_data=["id", "affair_id", "faction", "canton", "role"],
                    size_max=20,
                    color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_traces(marker=dict(opacity=0.5))
    fig.update_layout(scene=dict(
        xaxis_title=f"Factorial axis {dim1 + 1} : {var[dim1]:.2%}",
        yaxis_title=f"Factorial axis {dim2 + 1} : {var[dim2]:.2%}",
        zaxis_title=f"Factorial axis {dim3 + 1} : {var[dim3]:.2%}"))
    fig['layout']['uirevision'] = 'true'
    return fig

# ---- Interface 

with ui.element("div").classes("flex w-full h-screen"):
    # Sidebar
    with ui.element("div").classes("w-[17%] max-w-xs bg-base-100"):
        with ui.row():
            chose_3d = ui.switch("3D", value=False)
            merge_speaker = ui.switch("Speech/Speaker", value=False)
        select_1 = ui.select(all_dim, label='Axis X', value=1)
        select_2 = ui.select(all_dim, label='Axis Y', value=2)
        with ui.element("div").bind_visibility_from(chose_3d, 'value'):
            select_3 = ui.select(all_dim, label='Axis Z', value=3)
        select_color = ui.select(["faction",
                                  "affair_id",
                                  "canton", 
                                  "party", 
                                  "speech_lang", 
                                  "speaker_name",
                                  "role"], 
                                 label='Color by', value="faction")
        alignment = ui.select(["none",
                               "faction",
                               "affair_id",
                               "canton",
                               "party", 
                               "speech_lang", 
                               "speaker_name",
                               "role"],
                              label="Alignment", value="none")
        with ui.row().classes("items-end"):
            query_input = ui.input('Polars filter').classes('w-2/3')
            query_button = ui.button('Run').classes('h-1/2')
        ui.markdown("")
        ui.markdown("**Speech**")
        with ui.scroll_area().classes('max-w-xs h-100 border'):
            text_label = ui.label('')
        ui.markdown("")
        with ui.scroll_area().classes('max-w-xs h-100 border'):
            trsl_label = ui.label('')
    # Main
    with ui.element("div").classes("grow bg-base-100"):
        plotly_fig = ui.plotly(plot_res_2d(first_df, 
                                           explained_var, 0, 1, "faction"))\
            .classes('w-full h-full items-center')
            
# ---- Interactivity 

# The potential query 
polars_query = ""

# To update the plot, we need to use the event system of NiceGUI
def update_plot():
    
    # Which dataset
    if merge_speaker.value:
        w = g_weights
        Ker = gwdt_scalarp_mat
        df = speakers_df
        X_coord = gX_mds
        var = geigenvalues / np.sum(geigenvalues)
    else:
        w = weights
        Ker = wdt_scalarp_mat
        df = merged_df
        X_coord = X_mds
        eigenvalues = np.sum(X_coord.T**2 * w, axis=1)
        var = eigenvalues / np.sum(eigenvalues)
        
    # Alignement 
    if alignment.value != "none":
        dummies = df[alignment.value].to_dummies().to_numpy()
        Ker_memb = kernel_membership_matrix(dummies, w)
        isometry = isometry_matrix(Ker, Ker_memb)
        X_coord = X_coord @ isometry
        lambdas = np.sum(X_coord.T**2 * w, axis=1)
        var = lambdas / np.sum(lambdas)
        
    # Merge the dataframe with the MDS coordinates
    X_df = pl.from_numpy(X_coord).rename(lambda c_n: f"MDS_{c_n.split("_")[1]}")
    df = df.with_columns(X_df)
    
    # Set dimensions 
    max_dim = np.sum(np.round(var, 20) > 0)
    all_dim = (np.arange(max_dim) + 1).tolist()
    select_1.set_options(all_dim)
    select_2.set_options(all_dim)
    select_3.set_options(all_dim)
        
    # Try the query
    if polars_query != "":
        try:
            df = df.filter(eval(polars_query))
        except Exception as e:
            ui.notify(f"Error in query: {e}")
    
    # 2D or 3D
    if chose_3d.value:
        ui.notify("3D plot")
        plotly_fig.update_figure(plot_res_3d(df, var, 
                                             select_1.value - 1, 
                                             select_2.value - 1,
                                             select_3.value - 1,
                                             select_color.value))
        
    else:
        plotly_fig.update_figure(plot_res_2d(df, var, 
                                             select_1.value - 1, 
                                             select_2.value - 1,
                                             select_color.value))
        
def update_query():
    globals()["polars_query"] = query_input.value
    update_plot()

# Function to display the speech
async def get_speech(event):
    point_id = event.args["points"][0]["customdata"][0]
    if merge_speaker.value:
        df = speakers_df
    else:
        df = merged_df
    speech = df.filter(id = point_id)["speech"].item().strip()
    text_label.set_text(speech)
    translation = await run.cpu_bound(my_translate, speech, translator, 5000)
    ui.notify('Translation done')
    trsl_label.set_text(translation)
    
# Add the event handlers to the select elements
select_1.on_value_change(update_plot) 
select_2.on_value_change(update_plot)
select_3.on_value_change(update_plot)
chose_3d.on_value_change(update_plot)
merge_speaker.on_value_change(update_plot)
select_color.on_value_change(update_plot)
alignment.on_value_change(update_plot)
plotly_fig.on('plotly_click', get_speech)
query_button.on('click', update_query)

# ---- Run the app
ui.run(title="Embeddings MDS")



