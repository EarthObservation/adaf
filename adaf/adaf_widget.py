import ipywidgets as widgets
from IPython.display import display
from pathlib import Path
from adaf_inference import batch_routine
from adaf_utils import ADAFInput
import traitlets
from tkinter import Tk, filedialog


# ~~~~~~~~~~~~~~~~~~~~~~~~ INPUT FILES ~~~~~~~~~~~~~~~~~~~~~~~~
class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self):
        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = "Select file"
        self.style.button_color = None
        # Set on click behavior.
        self.on_click(self.select_files)

    @staticmethod
    def select_files(b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected files will be set to b.value
        b.files = filedialog.askopenfilename(
            title="Select input files",
            filetypes=[("GeoTIF", "*.tif;*.tiff"), ("VRT", "*.vrt")],
            multiple=True
        )

        n_files = len(b.files)
        if n_files > 0 and any(element != "" for element in b.files):
            b.description = f"{n_files} File selected" if n_files == 1 else f"{n_files} Files selected"
            b.style.button_color = "lightgreen"


# There are 2 options, switching between them will enable either DEM or Visualizations text_box
rb_input_file_options = [
    'DEM (*.tif / *.vrt)',
    'Visualization (*.tif / *.vrt)'
]

# The main radio button options (se the list of available options above)
rb_input_file = widgets.RadioButtons(
    options=rb_input_file_options,
    value=rb_input_file_options[0],
    description='Select input file:',
    disabled=False
)

# Button - opens dialog window (select file/s)
b_file_select = SelectFilesButton()

chk_save_vis = widgets.Checkbox(
    value=False,
    description='Save visualizations',
    disabled=False,
    indent=False
)


# Radio buttons handler (what happens if radio button is changed)
def input_file_handler(value):
    if rb_input_file.index == 0:
        chk_save_vis.disabled = False
    else:
        chk_save_vis.disabled = True


# When radio button trait changes, call the what_traits_radio function
rb_input_file.observe(input_file_handler)

# ~~~~~~~~~~~~~~~~~~~~~~~~ ML SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~
rb_semseg_or_objdet = widgets.RadioButtons(
    options=['segmentation', 'object detection'],
    value='segmentation',
    # layout={'width': 'max-content'}, # If the items names are long
    description='Select ML method:',
    disabled=False
)

# Checkboxes for classes
class_barrow = widgets.Checkbox(
    value=False,
    description='Barrow',
    disabled=False,
    indent=False,
    layout=widgets.Layout(flex='1 1 auto', width='auto')
)

class_ringfort = widgets.Checkbox(
    value=False,
    description='Ringfort',
    disabled=False,
    indent=False,
    layout=widgets.Layout(flex='1 1 auto', width='auto')
)

class_enclosure = widgets.Checkbox(
    value=False,
    description='Enclosure',
    disabled=False,
    indent=False,
    layout=widgets.Layout(flex='1 1 auto', width='auto')
)

class_all_archaeology = widgets.Checkbox(
    value=True,
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
            width=100,
            height=100
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
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='65%'),
    disabled=True
)

# --------------------------------------------------------
# The classes subgroup
cl = widgets.Label("Select classes for inference:")

# classes_box = widgets.HBox([class_barrow, class_ringfort, class_enclosure, class_all_archaeology])
classes_box = widgets.GridBox(
    children=[
        class_all_archaeology, class_barrow, class_ringfort, class_enclosure,
        widgets.Label(), img_b, img_r, img_e
    ],
    layout=widgets.Layout(
        width='80%',
        grid_template_columns='20% 20% 20% 20%',
        grid_template_rows='30px auto',
        grid_gap='5px'
    )
)

# Stack ADAF on top of Custom
adaf_box = widgets.VBox([
    cl,
    classes_box,
], layout=widgets.Layout(width='100%'))
# --------------------------------------------------------

# --------------------------------------------------------
custom_box = txt_custom_model
# --------------------------------------------------------

stack = widgets.Stack([adaf_box, custom_box], selected_index=0)

# Dropdown for ADAF vs CUSTOM
dropdown = widgets.Dropdown(
    options=['ADAF model', 'Custom model'],
    style={'description_width': 'initial'},
    description='Select model:'
)
widgets.jslink((dropdown, 'index'), (stack, 'selected_index'))


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

# ~~~~~~~~~~~~~~~~~~~~ Checkbox to save ML predictions  ~~~~~~~~~~~~~~~~~~~~

# Checkbox to save ML predictions files
chk_save_predictions_descriptions = [
    "Keep probability masks (raw ML results)",
    "Keep bounding box txt files (raw ML results)"
]
chk_save_predictions = widgets.Checkbox(
    value=False,
    description=chk_save_predictions_descriptions[0],
    disabled=False,
    indent=False
)


def chk_save_predictions_handler(value):
    chk_save_predictions.description = chk_save_predictions_descriptions[rb_semseg_or_objdet.index]

    if value.new == 0:
        # Segmentation
        fs_roundness.disabled = False
        fs_area.disabled = False
    elif value.new == 1:
        # Segmentation
        fs_roundness.disabled = True
        fs_area.disabled = True


rb_semseg_or_objdet.observe(chk_save_predictions_handler)

# ~~~~~~~~~~~~~~~~~~~~~~~~ POST PROCESSING ~~~~~~~~~~~~~~~~~~~~~~~~

fs_area = widgets.FloatSlider(
    value=30,
    min=0,
    max=40,
    step=1,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',

)

fs_roundness = widgets.FloatSlider(
    value=0.75,
    min=0,
    max=0.95,
    step=0.05,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f'
)

# ~~~~~~~~~~~~~~~~~~~~~~~~ BUTTON OF DOOM (click to run the app) ~~~~~~~~~~~~~~~~~~~~~~~~
button_run_adaf = widgets.Button(
    description="Run ADAF",
    layout={'width': '65%', 'border': '1px solid black'},  # widgets.Layout(width='98%'), 'border': '1px solid black'
    tooltip='Description'
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
    with output:
        display(dropdown.value)

    button_run_adaf.disabled = True

    # Check if paths are correct for custom model
    custom_model_pth = Path(txt_custom_model.value)
    if rb_ml_switch.index == 0:
        custom_tar_ok = True
    elif rb_ml_switch.index == 1 and custom_model_pth.is_file():
        custom_tar_ok = True
    else:
        with output:
            display("The specified Custom Model file doesn't exist!")
        custom_tar_ok = False

    run_app = custom_tar_ok

    if run_app:
        # Prepare input parameter for processing visualization
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
            class_enclosure,
            class_all_archaeology,
        ]
        class_selection = [select_class(a) for a in class_selection if a.value]

        # Save visualizations (Only available if DEM is selected)
        if not vis_exist_ok and chk_save_vis.value:
            save_vis = True
        else:
            save_vis = False

        # Save values into input object  # TODO: have a dict that is updated with every event!
        my_input = ADAFInput()
        my_input.update(
            input_file_list=b_file_select.files,  # Input is list of paths
            vis_exist_ok=vis_exist_ok,
            save_vis=save_vis,
            ml_type=rb_semseg_or_objdet.value,
            labels=class_selection,
            ml_model_custom=dropdown.value,
            custom_model_pth=txt_custom_model.value,
            roundness=fs_roundness.value,
            min_area=fs_area.value,
            save_ml_output=chk_save_predictions.value
        )

        with output:
            display("Inputs check complete.")
            display("RUNNING ADAF!")
        # def main_routine(dem_path, ml_type, model_path, tile_size_px, prob_threshold, nr_processes=1):

        # RUN ACTUAL MAIN ROUTINE
        final_adaf_output = batch_routine(my_input)

        with output:
            display(final_adaf_output)

    button_run_adaf.disabled = False


# button_run_adaf.on_click(on_button_clicked(abc=test_upload))
button_run_adaf.on_click(on_button_clicked)


def select_class(chk_widget):
    if chk_widget.value:
        a = chk_widget.description.lower()
        if a == "all archaeology":
            a = "AO"
    else:
        a = None

    return a


# ~~~~~~~~~~~~~~~~~~~~~~~~ DISPLAYING WIDGETS ~~~~~~~~~~~~~~~~~~~~~~~~

post_proc_box = widgets.GridBox(
    children=[widgets.HTML(value='Select min area [m<sup>2</sup>]:'), fs_area,
              widgets.HTML(value='Select min roundness:'), fs_roundness],
    layout=widgets.Layout(
        width='60%',
        grid_template_columns='30% 20%',
        grid_template_rows='auto auto',
        grid_gap='1px',
        # margin='0 0 0 20px'
    )
)

# This controls the overall display elements -- padding or margin = [top/right/bottom/left]
box_layout = widgets.Layout(
    border='solid 1px grey',
    padding='0 5px 5px 5px',
    grid_gap='10px'
    # display='flex',
    # flex_flow='column'
    # align_items='stretch'
)

display(
    widgets.VBox(
        [
            widgets.VBox([
                widgets.HTML(value=f"<b>Input data options:</b>"),
                widgets.VBox(
                    [
                        rb_input_file,
                        b_file_select,
                        chk_save_vis
                    ],
                    layout=box_layout
                ),
            ]),
            widgets.VBox([
                widgets.HTML(value=f"<b>ML options:</b>"),
                widgets.VBox(
                    [
                        rb_semseg_or_objdet,
                        dropdown,
                        stack
                    ],
                    layout=box_layout
                ),
            ]),
            widgets.VBox([
                widgets.HTML(value=f"<b>Post-processing options:</b>"),
                widgets.VBox(
                    [
                        post_proc_box,
                        chk_save_predictions
                    ],
                    layout=box_layout
                )
            ]),
            button_run_adaf,
            output
        ],
        layout=widgets.Layout(grid_gap='5px')
    ),
)
