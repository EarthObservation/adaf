import ipywidgets as widgets
from IPython.display import display
from adaf_inference import main_routine

# Define output Context manager
output = widgets.Output()

# Display full text in the description of the widget
style = {'description_width': 'initial', 'description_color': 'red'}

# ~~~~~~~~~~~~~~~~~~~~~~~~ INPUT FILES ~~~~~~~~~~~~~~~~~~~~~~~~
# There are 2 options, switching between the will enable either DEM or Visualizations text_box
input_radio_options = [
    'DEM [path to *.tif or path to *.vrt file]',
    'Visualizations (path to directory)'
]

# The main radio button options (se the list of available options above)
switch_dem_input = widgets.RadioButtons(
    options=input_radio_options,
    value=input_radio_options[0],
    description='Select ML method:',
    disabled=False
)

# This widget sets path to DEM file
dem_input = widgets.Text(
    description='DEM path:',
    placeholder="<my_data_folder/my_DEM_file.tif>",
    layout=widgets.Layout(width='98%'),
    style=style,
    disabled=False
)
# # To grey out Label, use widgets.HTML, and create GRID, to align all the elements
# text = 'DEM path:'
# my_label = widgets.HTML(value=f"<font color='grey'>{text}")
# display(widgets.HBox([my_label, dem_input]))

# This widget sets path to visualizations folder
visualization_input = widgets.Text(
    description='Visualizations folder:',
    placeholder="<my_data_folder>",
    layout=widgets.Layout(width='98%'),
    style=style,
    disabled=True
)

# Define context manager for displaying the output (text box for DEM)
output4 = widgets.Output()
with output4:
    display(dem_input, visualization_input)


# Radio buttons handler (what happens if radio button is changed)
def what_traits_radio(value):
    widgets.Output().clear_output()
    if value['new'] != input_radio_options[0]:
        # VIS option is selected
        dem_input.disabled = True
        visualization_input.disabled = False
    else:
        # DEM option is selected
        dem_input.disabled = False
        visualization_input.disabled = True

    with widgets.Output():
        display(dem_input, visualization_input)


# When radio button trait changes, call the what_traits_radio function
switch_dem_input.observe(what_traits_radio, names='value')


# ~~~~~~~~~~~~~~~~~~~~~~~~ ML SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~
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

# inp4 = widgets.Dropdown(
#     description='Tile size [pixels]:',
#     options=[256, 512, 1024, 2048],
#     value=1024,
#     layout=widgets.Layout(width='20%'),
#     style=style
# )

# inp5 = widgets.FloatSlider(
#         min=0.3,
#         max=0.9,
#         step=0.1,
#         value=0.5,
#         # description='Threshold (predictions probability)',
#         orientation='horizontal',
#         disabled=False,
#         # style=style
#     )
# hb5 = HBox([Label('Threshold (predictions probability)'), inp5])

# inp6 = widgets.IntText(
#     description='Number of CPUs:',
#     value=6,
#     layout=widgets.Layout(width='20%'),
#     style=style
# )


# BUTTON OF DOOM (click to run the app)
button_run_adaf = widgets.Button(
    description="Run ADAF",
    layout=widgets.Layout(width='98%')
)


# Handler for BUTTON OF DOOM
def on_button_clicked(b):
    if switch_dem_input.index == 0:
        # DEM is selected
        vis_exist_ok = False
        dem_path = dem_input.value
    else:
        # Visualization is selected
        vis_exist_ok = True
        dem_path = visualization_input.value

    # def main_routine(dem_path, ml_type, model_path, tile_size_px, prob_threshold, nr_processes=1):
    fun_output = main_routine(
        dem_path=dem_path,
        ml_type=inp2.value,
        model_path=inp3.value,
        vis_exist_ok=vis_exist_ok
    )
    with output:
        display(fun_output)


button_run_adaf.on_click(on_button_clicked)

# ~~~~~~~~~~~~~~~~~~~~~~~~ DISPLAYING WIDGETS ~~~~~~~~~~~~~~~~~~~~~~~~
# Arrange all widgets vertically one-by-one
main_box = widgets.VBox(
    [inp2, inp3, button_run_adaf]
)


display(switch_dem_input, output4)

display(main_box)

# # This will
# display(output)
