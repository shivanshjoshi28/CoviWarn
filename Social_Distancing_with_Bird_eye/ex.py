import base64
import os

# function taken from the discussion.streamlit . it will create a link to download the picture
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


def start():
    return(get_binary_file_downloader_html('frame.jpg', 'Picture'))
