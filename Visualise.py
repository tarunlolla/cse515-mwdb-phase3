import os
import webbrowser as wb
from IPython.display import display, HTML
import helpers
def make_html(folder, image, extras):
    #text='<li><a href="#" style="background-image:"{}""></a><h3><a href="#">{}</a></h3></li>'.format(os.path.join(folder, image),image)
    text='<div style="height:175px;width:175px;border:1px;border-style:solid;border-color:rgb(0,0,0);"><img src="{}" style="display:block;height:120px;width:160px;margin:5px;"/><p style="font: italic smaller sans-serif; text-align:center;">{}<br>{}</p></div>'.format(os.path.join(folder, image),image,extras)
    print(text)
    return text

# dataset_path,metadata_path=helpers.fetchDatasetDetails()
# files=os.listdir(dataset_path)
# text=''
# for x in files:
#     text += make_html(dataset_path, x)
# html_file=open('render.html','w')
# html_file.write('<div style="display: grid; grid-template-columns: repeat(6, 1fr); grid-template-rows: repeat(8, 5vw);grid-gap: 100px;">'+text+'</div>')
# wb.open_new_tab("/home/tarunlolla/MWDB/Phase3/render.html")
#display(HTML(text))
