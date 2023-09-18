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
    'DEM (*.tif / *.vrt)',
    'Visualization (*.tif / *.vrt)'
]

# The main radio button options (se the list of available options above)
switch_dem_input = widgets.RadioButtons(
    options=input_radio_options,
    value=input_radio_options[0],
    description='Select input type:',
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

chk_batch_process = widgets.Checkbox(
    value=False,
    description='Batch processing',
    disabled=False,
    indent=False
)

# This widget sets path to visualizations folder
visualization_input = widgets.Text(
    description='Visualization path:',
    placeholder="<my_data_folder/my_visualization_file.tif>",
    layout=widgets.Layout(width='98%'),
    style=style,
    disabled=True
)

# Define context manager for displaying the text box for input DEM
output4 = widgets.Output()
with output4:
    display(dem_input)


# # Radio buttons handler (what happens if radio button is changed)
# debug_view = widgets.Output(layout={'border': '1px solid black'})
#
# @debug_view.capture(clear_output=True)
def what_traits_radio(value):
    output4.clear_output()
    if value['new'] != input_radio_options[0]:
        # VIS option is selected
        dem_input.disabled = True
        visualization_input.disabled = False
        with output4:
            display(visualization_input)
        return "VIS"
    else:
        # DEM option is selected
        dem_input.disabled = False
        visualization_input.disabled = True
        with output4:
            display(dem_input)
        return "DEM"

    print(value)


def check_batch(value):
    return value


# When radio button trait changes, call the what_traits_radio function
switch_dem_input.observe(what_traits_radio)  # , names='value')
# chk_batch_process.observe(what_traits_radio)


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

# Checkboxes for classes
class_barrow = widgets.Checkbox(
    value=True,
    description='Barrow',
    disabled=False,
    indent=False
)

class_ringfort = widgets.Checkbox(
    value=True,
    description='Ringfort',
    disabled=False,
    indent=False
)

class_enclosure = widgets.Checkbox(
    value=True,
    description='Enclosure',
    disabled=False,
    indent=False
)

class_all_archaeology = widgets.Checkbox(
    value=False,
    description='All archaeology',
    disabled=False,
    indent=False
)

# ~~~~~~~~~~~~~~~~~~~~~~~~ BUTTON OF DOOM (click to run the app) ~~~~~~~~~~~~~~~~~~~~~~~~
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
# The classes sub-group
classes_box = widgets.HBox([class_barrow, class_ringfort, class_enclosure, class_all_archaeology])

# Second part of widget (ML settings) arranged vertically
main_box = widgets.VBox(
    [widgets.Label("- - - - - - - - - - -"), inp2, classes_box, inp3, button_run_adaf]  #
)

# This controls the overall display elements
display(
    widgets.HBox([switch_dem_input, chk_batch_process]),
    output4,
    main_box
)

# # This will
# display(output)
