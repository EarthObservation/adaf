import ipywidgets as widgets
from ipywidgets import HBox, Label
from IPython.display import display
from adaf_inference import main_routine

style = {'description_width': 'initial'}

output = widgets.Output()

button = widgets.Button(
    description="Run ADAF",
    layout=widgets.Layout(width='98%')
)

inp1 = widgets.Text(
    description='DEM path [*.tif / *.vrt]:',
    placeholder="data_folder/file.tif",
    layout=widgets.Layout(width='98%'),
    style=style
    # layout={'width': 'max-content'}
    # value="data/ISA-15_Kilkee/ISA-15_Kilkee_dem_05m.vrt"
)

inp2 = widgets.RadioButtons(
    options=['object detection', 'segmentation'],
    value='segmentation',
    # layout={'width': 'max-content'}, # If the items' names are long
    description='Select ML method:',
    disabled=False
)

inp3 = widgets.Text(
    description='Path to ML model [*.tar]:',
    placeholder="model_folder/saved_model.tar",
    style=style,
    layout=widgets.Layout(width='98%'),
    # value="results/test_save_01"
)

inp4 = widgets.Dropdown(
    description='Tile size [pixels]:',
    options=[256, 512, 1024, 2048],
    value=1024,
    layout=widgets.Layout(width='20%'),
    style=style
)

inp5 = widgets.FloatSlider(
        min=0.3,
        max=0.9,
        step=0.1,
        value=0.5,
        # description='Threshold (predictions probability)',
        orientation='horizontal',
        disabled=False,
        # style=style
    )
# hb5 = HBox([Label('Threshold (predictions probability)'), inp5])

# inp6 = widgets.IntText(
#     description='Number of CPUs:',
#     value=6,
#     layout=widgets.Layout(width='20%'),
#     style=style
# )

main_box = widgets.VBox(
    [inp1, inp2, inp3, inp4, button]
)

# def main_routine(dem_path, ml_type, model_path, tile_size_px, prob_threshold, nr_processes=1):


def on_button_clicked(b):
    fun_output = main_routine(
        inp1.value,
        inp2.value,
        inp3.value,
        inp4.value,
    )
    with output:
        display(fun_output)


button.on_click(on_button_clicked)
display(main_box)
display(output)
