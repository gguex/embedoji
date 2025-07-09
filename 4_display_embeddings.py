import os
import polars as pl
import polars.selectors as cs
from sklearn.metrics.pairwise import cosine_similarity
from src.mv_tools import *
import plotly.express as px
from nicegui import ui, run

# -----------------------------
# --- PARAMETERS
# -----------------------------

EMBEDDINGS_FOLDER = "data/data/corpus_embeddings/"
CORPUS_CSV_FOLDER = \
    "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/corpus_csv/"
    
DATE_SPAN = [pl.date(2020, 3, 16) , pl.date(2020, 6, 20) ]

# -----------------------------
# --- CODE 
# -----------------------------

# Get the msg id
chats =  os.listdir(CORPUS_CSV_FOLDER)
chats.sort()

all_msg_ids = []
all_chat_name = []
for chat in chats:
    df = pl.read_csv(CORPUS_CSV_FOLDER + chat, try_parse_dates=True, infer_schema_length=1000)
    df = df.filter((pl.col("msg_date") >= DATE_SPAN[0]) & (pl.col("msg_date") <= DATE_SPAN[1]))
    id_list = df["msg_id"].to_list()
    chat_list = [chat.split(".csv")[0]] * len(id_list)
    all_msg_ids.extend(id_list)
    all_chat_name.extend(chat_list)

# Read all embeddings
embedding_files = os.listdir(EMBEDDINGS_FOLDER)
embedding_files.sort()

embeddings = [embedding_file for embedding_file in embedding_files 
              if "wns_chat_17" in embedding_file]

point_count = []
for file in embeddings:
    df = pl.read_csv(EMBEDDINGS_FOLDER + file, has_header=False, infer_schema=False) 
    size = df.shape[0]
    point_count.append(size)
    
print(f"Total number of points: {np.sum(point_count)}")
print(f"Mean number of points: {np.max(point_count)}")


# Merge the dataframes
merged_df = pl.concat([debate_df, embeddings_df], how="horizontal")

# Keep only the councillors
merged_df = merged_df.filter(
    pl.col("role").is_in(["councillor", "councillor-k", "councillor-b"]))

# Sort it
merged_df = merged_df.sort("speaker_name")

# ---- Compute the Dataset for speakers

# Merge the dataframe
speakers_df = merged_df.group_by("speaker_name").agg(
    pl.col("number").first().alias("id"),
    pl.col("affair_id").mode().first(),
    pl.col("faction").first(),
    pl.col("party").first(),
    pl.col("canton").first(),
    pl.col("n_tokens").sum(),
    pl.col("speech_lang").first(),
    pl.col("speech").str.join(","), 
    pl.col("role").mode().first())

# Compute the one hot encoding fro speakers
speaker_names = merged_df["speaker_name"]
speakers_one_hot = speaker_names.to_dummies()
z_group = 1*speakers_one_hot.to_numpy()
n_g = z_group.shape[1]
    
# ------------------------------------------
# ---- Make the MDS
# ------------------------------------------

# ---- MDS on speeches

# Get the columns starting with "d" as a numpy array from merged_df
embeddings = merged_df.select(cs.starts_with("d")).to_numpy()

# Weights
weights = merged_df["n_tokens"].to_numpy()

# Compute the cosine similarity
cosine_d = 1 - cosine_similarity(embeddings)

# Compute it
scalarp_mat = scalar_product_matrix(cosine_d, weights)
wdt_scalarp_mat = weighted_scalar_product_matrix(scalarp_mat, weights)
X_mds, eigenvalues = weighted_mds(wdt_scalarp_mat, weights=weights)
mds_max_dim = len(eigenvalues) - 1
all_dim = (np.arange(mds_max_dim) + 1).tolist()

# For the first plot
mds_df = pl.from_numpy(X_mds).rename(lambda c_n : f"MDS_{c_n.split("_")[1]}")
first_df = merged_df.with_columns(mds_df)
explained_var = eigenvalues / sum(eigenvalues)

# ---- MDS on speakers

g_weights = speakers_df["n_tokens"].to_numpy()

cosine_g_d = group_distance(cosine_d, z_group, weights)

# Make the mds
gscalarp_mat = scalar_product_matrix(cosine_g_d, g_weights)
gwdt_scalarp_mat = weighted_scalar_product_matrix(gscalarp_mat, g_weights)
gX_mds, geigenvalues = weighted_mds(gwdt_scalarp_mat, weights=g_weights)

# Add the MDS coordinates to the dataframe
gmds_df = pl.from_numpy(gX_mds).rename(lambda c_n : f"MDS_{c_n.split("_")[1]}")
speakers_df = speakers_df.with_columns(gmds_df)

# ------------------------------------------
# ---- INTERFACE
# ------------------------------------------

# ---- Get the translator

translator = GoogleTranslator(source='auto', target='fr')

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



