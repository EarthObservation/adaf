import ipywidgets as widgets
from IPython.display import display
from adaf_inference import main_routine
from adaf_utils import ADAFInput


# ~~~~~~~~~~~~~~~~~~~~~~~~ INPUT FILES ~~~~~~~~~~~~~~~~~~~~~~~~
# Display full text in the description of the widget
style = {'description_width': 'initial'}

# There are 2 options, switching between them will enable either DEM or Visualizations text_box
rb_input_file_options = [
    'DEM (*.tif / *.vrt)',
    'Visualization (*.tif / *.vrt)'
]
txt_input_file_placeholders = [
    "<my_data_folder/my_DEM_file.tif>",
    "<my_data_folder/my_visualization_file.tif>"
]
txt_input_file_descriptions = [
    'DEM path:',
    'Visualization path:'
]

# The main radio button options (se the list of available options above)
rb_input_file = widgets.RadioButtons(
    options=rb_input_file_options,
    value=rb_input_file_options[0],
    description='Select input file:',
    disabled=False
)

# Textbox for PATH to input file
txt_input_file = widgets.Text(
    description=txt_input_file_descriptions[0],
    placeholder=txt_input_file_placeholders[0],
    layout=widgets.Layout(width='98%'),
    style=style,
    disabled=False
)

chk_save_vis = widgets.Checkbox(
    value=True,
    description='Save visualizations',
    disabled=False,
    indent=False
)

chk_batch_process = widgets.Checkbox(
    value=False,
    description='Batch processing',
    disabled=False,
    indent=False
)


# Radio buttons handler (what happens if radio button is changed)
# # DEBUGGING
# debug_view = widgets.Output(layout={'border': '1px solid black'})
# @debug_view.capture(clear_output=True)
def input_file_handler(value):
    # # DEBUGGING
    # print("RB:", rb_input_file.index)
    # print("CHK:", chk_batch_process.value)

    # Clear any text that was entered by user
    txt_input_file.value = ""

    # 0 for DEM, 1 for VIS
    rb_idx = rb_input_file.index

    if chk_batch_process.value:
        # Batch processing is enabled, change placeholder to TXT
        txt_input_file.description = txt_input_file_descriptions[rb_idx]
        txt_input_file.placeholder = "<list of paths in TXT file!>"
    else:
        # Select DEM or VIS based on RB selection
        txt_input_file.description = txt_input_file_descriptions[rb_idx]
        txt_input_file.placeholder = txt_input_file_placeholders[rb_idx]

    if rb_input_file.index == 0:
        chk_save_vis.disabled = False
    else:
        chk_save_vis.disabled = True


# When radio button trait changes, call the what_traits_radio function
rb_input_file.observe(input_file_handler)
chk_batch_process.observe(input_file_handler)

# ~~~~~~~~~~~~~~~~~~~~~~~~ ML SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~
inp2 = widgets.RadioButtons(
    options=['segmentation', 'object detection'],
    value='segmentation',
    # layout={'width': 'max-content'}, # If the items names are long
    description='Select ML method:',
    disabled=False
)

# Checkboxes for classes
class_barrow = widgets.Checkbox(
    value=True,
    description='Barrow',
    disabled=False,
    indent=False,
    layout=widgets.Layout(flex='1 1 auto', width='auto')
)

class_ringfort = widgets.Checkbox(
    value=True,
    description='Ringfort',
    disabled=False,
    indent=False,
    layout=widgets.Layout(flex='1 1 auto', width='auto')
)

class_enclosure = widgets.Checkbox(
    value=True,
    description='Enclosure',
    disabled=False,
    indent=False,
    layout=widgets.Layout(flex='1 1 auto', width='auto')
)

class_all_archaeology = widgets.Checkbox(
    value=False,
    description='All archaeology',
    disabled=False,
    indent=False,
    layout=widgets.Layout(flex='1 1 auto', width='auto')
)


def img_widget(path):
    # Open image for txtbox
    with open(path, "rb") as src:
        image = src.read()
        img_wid = widgets.Image(
            value=image,
            format='jpg',
            width=80,
            height=80
        )

    return img_wid


img_b = img_widget("media/barrows.jpg")
img_r = img_widget("media/ringfort.jpg")
img_e = img_widget("media/enclosure.jpg")

# ~~~~~~~~~~~~~~~~~~~~~~~~ Custom ML model ~~~~~~~~~~~~~~~~~~~~~~~~

rb_ml_switch = widgets.RadioButtons(
    options=['ADAF model', 'Custom model'],
    value='ADAF model',
    # layout={'width': 'max-content'}, # If the items names are long
    description='Select ML model:',
    disabled=False
)

txt_custom_model = widgets.Text(
    description='Path to custom ML model [*.tar]:',
    placeholder="model_folder/saved_model.tar",
    style=style,
    layout=widgets.Layout(width='98%'),
    disabled=True
)

debug_view = widgets.Output(layout={'border': '1px solid black'})


@debug_view.capture(clear_output=True)
def ml_method_handler(value):
    print(value.new)

    # 0 for ADAF model, 1 for Custom model
    if value.new == 0:
        class_barrow.disabled = False
        class_ringfort.disabled = False
        class_enclosure.disabled = False
        class_all_archaeology.disabled = False
        txt_custom_model.disabled = True
    elif value.new == 1:
        class_barrow.disabled = True
        class_ringfort.disabled = True
        class_enclosure.disabled = True
        class_all_archaeology.disabled = True
        txt_custom_model.disabled = False


# When radio button trait changes, call the what_traits_radio function
rb_ml_switch.observe(ml_method_handler, names="index")

# ~~~~~~~~~~~~~~~~~~~~~~~~ BUTTON OF DOOM (click to run the app) ~~~~~~~~~~~~~~~~~~~~~~~~
button_run_adaf = widgets.Button(
    description="Run ADAF",
    layout={'width': '98%', 'border': '1px solid black'}  # widgets.Layout(width='98%'), 'border': '1px solid black'
)

# Define output Context manager
output = widgets.Output()  # layout={'border': '1px solid black'})


# Handler for BUTTON OF DOOM
def on_button_clicked(b):
    """
    List of available input parameters:
    rb_input_file.index - 0 for DEM, 1 for visualization
    txt_input_file.value
    inp2.value

    model_path - hard coded based on the inp2.value (segmentation or object detection)
    """

    if rb_input_file.index == 0:
        # DEM is selected
        vis_exist_ok = False
    else:
        # Visualization is selected
        vis_exist_ok = True

    # Select classes
    class_selection = [
        class_barrow,
        class_ringfort,
        class_all_archaeology,
        class_ringfort
    ]
    class_selection = [a.description for a in class_selection if a.value]

    my_input = ADAFInput()
    my_input.update(
        dem_path=txt_input_file.value,
        batch_processing=chk_batch_process.value,
        vis_exist_ok=vis_exist_ok,
        ml_type=inp2.value,
        classes_selection=class_selection,
        ml_model_rbt=rb_ml_switch.value,
        ml_model_pth=txt_custom_model.value
    )

    # def main_routine(dem_path, ml_type, model_path, tile_size_px, prob_threshold, nr_processes=1):
    final_adaf_output = main_routine(my_input)
    with output:
        display(final_adaf_output)


button_run_adaf.on_click(on_button_clicked)

# ~~~~~~~~~~~~~~~~~~~~~~~~ DISPLAYING WIDGETS ~~~~~~~~~~~~~~~~~~~~~~~~
# The classes sub-group
cl = widgets.Label("Select classes for inference:")

# classes_box = widgets.HBox([class_barrow, class_ringfort, class_enclosure, class_all_archaeology])
classes_box = widgets.GridBox(
    children=[class_barrow, class_ringfort, class_enclosure, class_all_archaeology, img_b, img_r, img_e],
    layout=widgets.Layout(
        width='80%',
        grid_template_columns='20% 20% 20% 20%',
        grid_template_rows='30px auto',
        grid_gap='1px'
    )
)

ml_methods_row = widgets.HBox([inp2, rb_ml_switch])

# This controls the overall display elements
display(
    widgets.HTML(value=f"<b>Input data options:</b>"),
    widgets.HBox([rb_input_file, chk_batch_process]),
    txt_input_file,
    chk_save_vis,
    widgets.HTML(value=f"<b>ML options:</b>"),
    widgets.VBox([ml_methods_row, cl, classes_box, txt_custom_model, button_run_adaf]),
    output
)
