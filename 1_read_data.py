import os
import re
import emoji
import polars as pl

DATA_FOLDER = "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/plain_text"
RESULT_FOLDER = "../swissubase_2579_1_0/data/wns_corpus_v1.0.0/data/"

file_names = os.listdir(DATA_FOLDER)
file_names.sort()

result_df = pl.Schema({
    "chat_name": pl.String,
    "msg_date": pl.String,
    "msg_user": pl.String,
    "msg_text": pl.String,
    "emoji_list": pl.String
}).to_frame()

for file_name in file_names:
    
    chat_name = file_name.split(".")[0]

    with open(os.path.join(DATA_FOLDER, file_name), "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    msg_date = msg_user = msg_text = None
    chat_res = []
    for line in lines:

        user_pos = re.search(r"_WNS_USER_(\d){3}_", line)
        hour_pos = re.search(r"\d{2}:\d{2}", line)
        
        if user_pos:
            msg_date = line[:user_pos.span()[0]].strip()
            msg_user = line[user_pos.span()[0]:user_pos.span()[1]].strip()
            msg_text = line[user_pos.span()[1]+1:].strip()
        elif hour_pos:
            msg_date = line[:hour_pos.span()[1]].strip()
            msg_user = None
            msg_text = line[hour_pos.span()[1]+1:].strip()
        else:
            msg_date = None if not msg_date else msg_date
            msg_user = None if not msg_user else msg_user
            msg_text = line.strip()
            
        # Find emoji
        emoji_list = ";".join([char for char in msg_text 
                               if emoji.is_emoji(char)])
        
        # Fill the DataFrame
        chat_res.append({
            "chat_name": chat_name,
            "msg_date": msg_date,
            "msg_user": msg_user,
            "msg_text": msg_text,
            "emoji_list": emoji_list
        })
        
    # Add to results
    result_df = pl.concat([result_df, pl.DataFrame(chat_res)])
    
# Save the DataFrame to a Parquet file
result_df.write_csv(os.path.join(RESULT_FOLDER, "wns_corpus.csv"))