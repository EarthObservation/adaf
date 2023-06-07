import ipywidgets as widgets
from IPython.display import display
from adaf.adaf_inference import create_patches
from ipywidgets import Layout

output = widgets.Output()

button = widgets.Button(
    description="Create patches",
    layout=widgets.Layout(width='98%')
)

inp1 = widgets.Text(
    description='DEM path:',
    placeholder="data_folder/file.tif",
    layout=widgets.Layout(width='98%'),
    # value="data/ISA-15_Kilkee/ISA-15_Kilkee_dem_05m.vrt"
)

inp2 = widgets.IntText(description='Patch size:',value=512)

inp22 = widgets.IntText(description='No. of CPUs:',value=16)

inp3 = widgets.Text(
    description='Save location:',
    placeholder="save_location/data_folder_name",
    layout=widgets.Layout(width='98%'),
    # value="results/test_save_01"
)

main_box = widgets.VBox(
    [inp1, inp2, inp22, inp3, button]
)

# def create_patches(ds_path, patch_size_px, vdm_path):
#     print(pth[:-5])


def on_button_clicked(b):
    fun_output = create_patches(
        inp1.value,
        inp2.value,
        inp3.value,
        inp22.value
    )
    with output:
        display(fun_output)


button.on_click(on_button_clicked)
display(main_box)
display(output)
