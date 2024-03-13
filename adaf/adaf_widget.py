from pathlib import Path
from tkinter import Tk, filedialog

import ipywidgets as widgets
import traitlets
from IPython.display import display
from yaspin import yaspin

from adaf.adaf_inference import main_routine
from adaf.adaf_utils import ADAFInput, build_vrt_from_list


# ~~~~~~~~~~~~~~~~~~~~~~~~ INPUT FILES OPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~
class SelectFilesButton(widgets.Button):
    def __init__(self):
        super(SelectFilesButton, self).__init__()

        # Add traits
        self.add_traits(files=traitlets.traitlets.List())

        # Button options
        self.description = "Select file"
        self.style.button_color = None

        # Set on click behavior.
        self.on_click(self.select_files)

    @staticmethod
    def select_files(b):
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)

        selected_files = filedialog.askopenfilename(
            title="Select input files",
            filetypes=[("GeoTIF", "*.tif;*.tiff"), ("VRT", "*.vrt")],
            multiple=True
        )

        n_files = len(selected_files)
        # IF THERE WAS AT LEAST ONE FILE SELECTED
        if n_files > 0 and any(element != "" for element in selected_files):
            # Change button style
            b.description = f"{n_files} File selected" if n_files == 1 else f"{n_files} Files selected"
            b.style.button_color = "lightgreen"

            # Update paths to files and folders
            if b_dir_select.out_is_selected:
                # If save location folder is already selected, only update files list
                b.files = selected_files
            else:
                # If save location was not defined by the user, use the parent dir of selected files as output location
                out_dir = Path(selected_files[0]).parent
                # Update save location label and set flag to true
                out_dir_label.value = f"<i><b>{out_dir}</b></i>"
                b_dir_select.folder = out_dir.as_posix()  # str(out_dir)
                b_dir_select.out_is_selected = True
                # Finally update files list
                b.files = selected_files


class SelectDirButton(widgets.Button):
    def __init__(self):
        super(SelectDirButton, self).__init__()

        # Add traits
        self.add_traits(folder=traitlets.traitlets.Any())
        self.add_traits(out_is_selected=traitlets.traitlets.Bool())
        self.out_is_selected = False

        # Button options
        self.description = "Change output folder"

        # Set on click behavior.
        self.on_click(self.on_button_click)

    @staticmethod
    def on_button_click(b):
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)

        # List of selected files will be set to b.value
        selected_folder = filedialog.askdirectory(
            title="Select output folder"
        )

        # Update the out_dir_label HTML with the selected directory
        if selected_folder:
            path_out_dir = Path(selected_folder)
            out_dir_label.value = f"<i><b>{path_out_dir.as_posix()}</b></i>"
            b.out_is_selected = True
            b.folder = selected_folder


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

# ############################
# ###### OUTPUT OPTIONS ######
b_dir_select = SelectDirButton()

out_dir_label = widgets.HTML(value=f"<i><b>not selected</b></i>")

out_dir_location = widgets.HBox(
    [
        widgets.Label(value="Output folder:"),
        out_dir_label
    ]
)
# ###### OUTPUT OPTIONS ######
# ############################

chk_save_vis = widgets.Checkbox(
    value=False,
    description='Save visualizations',
    disabled=False,
    indent=False
)

chk_tiling = widgets.Checkbox(
    value=False,
    description='Tiles are from same dataset (create VRT)',
    disabled=False,
    indent=False
)

file_selection = widgets.HBox(
    [
        b_file_select,
        chk_tiling
    ]
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


def img_widget(path, driver="jpg", size=100):
    # Open image for txtbox
    with open(path, "rb") as src:
        image = src.read()
        img_wid = widgets.Image(
            value=image,
            format=driver,
            width=size,
            height=size
        )

    return img_wid


img_b = img_widget(Path(__file__).resolve().parent / "media/barrows.jpg")
img_r = img_widget(Path(__file__).resolve().parent / "media/ringfort.jpg")
img_e = img_widget(Path(__file__).resolve().parent / "media/enclosure.jpg")

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
    layout=widgets.Layout(width='90%'),
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
        # width='80%',
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


def ml_method_handler(value):
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

label_round = widgets.Label("Roundness examples:", style=dict(text_color='Grey'))
img_round50 = img_widget(Path(__file__).resolve().parent / "media/roundness_0.50.svg", driver="svg+xml", size=50)
img_round95 = img_widget(Path(__file__).resolve().parent / "media/roundness_0.95.svg", driver="svg+xml", size=50)
lbl_round50 = widgets.Label("0.50", style=dict(text_color='Grey'))
lbl_round90 = widgets.Label("0.95", style=dict(text_color='Grey'))

roundness_box = widgets.GridBox(
    children=[
        label_round, lbl_round50, img_round50, lbl_round90, img_round95
    ],
    layout=widgets.Layout(
        width='70%',
        grid_template_columns='200px 10% auto 10% auto',
        grid_template_rows='auto',
        grid_gap='0px',
        object_fit='contain',
        align_items='center',
        border='solid 1px LightGrey'
        # border_left='solid 1px grey'# display='flex',
    )
)

# roundness_box = widgets.HBox([label_round, lbl_round50, img_round50, lbl_round90, img_round95])

fs_area = widgets.IntSlider(
    value=40,
    min=0,
    max=100,
    step=5,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',

)

fs_roundness = widgets.FloatSlider(
    value=0.5,
    min=0,
    max=0.95,
    step=0.05,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f'
)


# ~~~~~~~~~~~~~~~~~~~~~~~~ PROGRESS BAR ~~~~~~~~~~~~~~~~~~~~~~~~
# Create an Output widget
output_widget = widgets.Output()

# ~~~~~~~~~~~~~~~~~~~~~~~~ BUTTON OF DOOM (click to run the app) ~~~~~~~~~~~~~~~~~~~~~~~~
button_run_adaf = widgets.Button(
    description="Run ADAF",
    layout={'width': 'auto', 'border': '1px solid black'},
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
    output_widget.clear_output()
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
            tiles_to_vrt=chk_tiling.value,
            vis_exist_ok=vis_exist_ok,
            save_vis=save_vis,
            out_dir=b_dir_select.folder,
            ml_type=rb_semseg_or_objdet.value,
            labels=class_selection,
            ml_model_custom=dropdown.value,
            custom_model_pth=txt_custom_model.value,
            roundness=fs_roundness.value,
            min_area=fs_area.value,
            save_ml_output=chk_save_predictions.value
        )

        # RUN ACTUAL MAIN ROUTINE
        # with output_widget:
        #     with yaspin():
        #         final_adaf_output = batch_routine(my_input)

        with output_widget:
            with yaspin() as spin:
                batch_list = b_file_select.files

                if len(batch_list) == 1:
                    spin.write("Started - single image processing")
                elif len(batch_list) > 1 and chk_tiling.value:
                    spin.write("Started - building VRT")
                    vrt_path = Path(b_dir_select.folder) / "virtual_mosaic.vrt"
                    batch_list = [build_vrt_from_list(batch_list, vrt_path)]
                    spin.write("        - single image processing")
                elif len(batch_list) > 1:
                    spin.write("Started - batch processing")
                else:
                    spin.write("NO FILES SELECTED!")

                for file in batch_list:
                    spin.write(f" >>> {file}")
                    my_input.update(dem_path=file)

                    # adaf_output = main_routine(my_input)
                    main_routine(my_input)

                # finalize
                spin.ok("✔ Finished processing")

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
        # width='60%',
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
                        file_selection,
                        out_dir_location,
                        b_dir_select,
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
                        roundness_box,
                        chk_save_predictions,
                    ],
                    layout=box_layout
                )
            ]),
            widgets.Box(
                [button_run_adaf],
                layout=widgets.Layout(
                    display='flex',
                    flex_flow='column',
                    align_items='stretch',
                    width='100%'
                )
            ),
            output_widget,
            output
        ],
        layout=widgets.Layout(grid_gap='5px', width="65%")
    ),
)
