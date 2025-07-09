import polars as pl
import polars.selectors as cs
import numpy as np
import plotly.express as px
from nicegui import ui, run

# -----------------------------
# --- PARAMETERS
# -----------------------------

MDS_FILE = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/mds_files/mds100_v1.csv"

# -----------------------------
# --- CODE 
# -----------------------------

# ---- Get data

# Load the file
mds_df = pl.read_csv(MDS_FILE)

mds_df = mds_df.with_columns((pl.col("emoji_list") != "").alias("has_emoji"))

# Get the dimensions_
all_dims = [int(col.split("_")[1]) + 1 for col in mds_df.columns if "MDS" in col]

# Get the explained variance
mds_coord = mds_df.select(cs.starts_with("MDS_")).to_numpy()

# Compute the explained variance
variances = np.mean(mds_coord ** 2, axis=0)
variances /= np.sum(variances)

# ---- Plot functions 

# Plot 2D
def plot_res_2d(df, var, dim1, dim2, color_by):
    df = df.sort(color_by, descending=False)
    fig = px.scatter(df, x=f"MDS_{dim1}", y=f"MDS_{dim2}",
                     color=color_by,
                     hover_name="chat_name", 
                     hover_data=["msg_id", "msg_date", "msg_user", "emoji_list", "msg_text"], 
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

# ---- Interface 

with ui.element("div").classes("flex w-full h-screen"):
    # Sidebar
    with ui.element("div").classes("w-[17%] max-w-xs bg-base-100"):
        select_1 = ui.select(all_dims, label='Axis X', value=1)
        select_2 = ui.select(all_dims, label='Axis Y', value=2)
        select_color = ui.select(["chat_name",
                                  "msg_user",
                                  "has_emoji"], 
                                 label='Color by', value="chat_name")
    # Main
    with ui.element("div").classes("grow bg-base-100"):
        plotly_fig = ui.plotly(plot_res_2d(mds_df, 
                                           variances, 0, 1, "chat_name"))\
            .classes('w-full h-full items-center')
            
# ---- Interactivity 

# The potential query 
polars_query = ""

# To update the plot, we need to use the event system of NiceGUI
def update_plot():
        
    plotly_fig.update_figure(plot_res_2d(mds_df, variances, 
                                         select_1.value - 1, 
                                         select_2.value - 1,
                                         select_color.value))
    
    
# Add the event handlers to the select elements
select_1.on_value_change(update_plot) 
select_2.on_value_change(update_plot)
select_color.on_value_change(update_plot)

# ---- Run the app
ui.run(title="Embeddings MDS")



