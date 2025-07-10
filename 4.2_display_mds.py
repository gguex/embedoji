import polars as pl
import polars.selectors as cs
import numpy as np
import plotly.express as px
from nicegui import ui, run
from src.mv_tools import *

# -----------------------------
# --- PARAMETERS
# -----------------------------

MDS_FILE = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/mds_files/mds200_v1.csv"

# -----------------------------
# --- CODE 
# -----------------------------

# ---- Get data

# Load the file
mds_df = pl.read_csv(MDS_FILE)

mds_df = mds_df.with_columns((pl.col("emoji_list") != "").alias("has_emoji"))

# Get the dimensions_
all_dims = [int(col.split("_")[1]) + 1 for col in mds_df.columns if "MDS" in col]

# Get the coordinates and weights
mds_coord = mds_df.select(cs.starts_with("MDS_")).to_numpy()
w = mds_df["n_tokens"].to_numpy()
w = w / np.sum(w) 

# Compute the explained variance
variances = np.sum((mds_coord ** 2).T * w, axis=1)
variances /= np.sum(variances)

# ---- Plot functions 

# Plot 2D
def plot_res_2d(df, var, dim1, dim2, size_by, color_by):
    df = df.sort(color_by, descending=False)
    fig = px.scatter(df, x=f"MDS_{dim1}", y=f"MDS_{dim2}",
                     size=size_by,
                     color=color_by,
                     hover_name="chat_name", 
                     hover_data=["msg_id", "msg_date", "msg_user", "emoji_list", 
                                 "msg_text"],
                     size_max=30,
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
        select_alignment = ui.select(["none",
                                      "chat_name",
                                      "msg_user",
                                      "has_emoji"],
                                     label='Align to', value="none")
        ui.markdown("**Message**")
        with ui.scroll_area().classes('max-w-xs h-100 border'):
            msg_content = ui.html("")
            
    # Main
    with ui.element("div").classes("grow bg-base-100"):
        plotly_fig = ui.plotly(plot_res_2d(mds_df, 
                                           variances, 0, 1, 
                                           "n_tokens", "chat_name"))\
            .classes('w-full h-full items-center')
            
# ---- Interactivity 


# Function to update coordinates
def make_isometry(mds_df, alignement):
    mds_coord = mds_df.select(cs.starts_with("MDS_")).to_numpy()
    w = mds_df["n_tokens"].to_numpy()
    w = w / np.sum(w)
    out_sq_w = np.outer(np.sqrt(w), np.sqrt(w))
    ker = (mds_coord @ mds_coord.T) * out_sq_w
    
    dummies = mds_df[alignement].to_dummies().to_numpy()
    ker_memb = kernel_membership_matrix(dummies, w)
    isometry = isometry_matrix(ker, ker_memb)
    new_coord = mds_coord @ isometry
    
    new_vars = np.sum((new_coord ** 2).T * w, axis=1)
    new_vars = new_vars / np.sum(new_vars)
    
    new_mds_df = mds_df.clone()
    new_mds_df = new_mds_df.drop(cs.starts_with("MDS_"))
    new_coord_df = pl.from_numpy(new_coord).rename(lambda c_n : f"MDS_{c_n.split("_")[1]}")
    new_mds_df = new_mds_df.with_columns(new_coord_df)
    
    return new_mds_df, new_vars
    
    

# To update the plot, we need to use the event system of NiceGUI
async def update_plot():
    
    # Get the data
    if select_alignment.value != "none":
        ui.notify("Isometry is too slow for the moment...", color="info")
        # updated_df, updated_vars = await run.cpu_bound(make_isometry, 
        #                                                mds_df, 
        #                                                select_alignment.value)
        updated_df = mds_df
        updated_vars = variances
    else:
        updated_df = mds_df
        updated_vars = variances
    
    plotly_fig.update_figure(plot_res_2d(updated_df, updated_vars, 
                                         select_1.value - 1, 
                                         select_2.value - 1,
                                         "n_tokens",
                                         select_color.value))
    
    ui.notify("Plot updated", color="success")

# Function to display the message
def get_msg(event):
    point_id = event.args["points"][0]["customdata"][0]
    line = mds_df.filter(pl.col("msg_id") == point_id)
    msg_content.set_content(
        f"<p><strong>Chat: {line['chat_name'].item()}</strong></p>"
        f"<p><strong>User: {line['msg_user'].item()}</strong></p>"
        f"<p><strong>[{line['msg_date'].item()}]</strong></p>"
        f"<p>{line['msg_text'].item()}</p>")
    
# Add the event handlers to the select elements
select_1.on_value_change(update_plot) 
select_2.on_value_change(update_plot)
select_color.on_value_change(update_plot)
select_alignment.on_value_change(update_plot)
plotly_fig.on('plotly_click', get_msg)

# ---- Run the app
ui.run(title="Embeddings MDS")



