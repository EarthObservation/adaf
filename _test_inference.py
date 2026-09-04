from adaf.adaf_utils import ADAFInput
from adaf.adaf_inference import main_routine

my_input = ADAFInput()
my_input.update(
    # dem_path=r"r:\BiH_ALS_2025\BiH_ALS_2025_DMO_05m.vrt",
    dem_path=r"c:\test_adaf_retrain\trail_cliped\clipped_DEM_ws1.tif",
    vis_exist_ok=False,
    save_vis=False,
    out_dir=r"r:\delovno\nejc",
    ml_type="segmentation",
    labels=["barrow"],
    ml_model_custom="Custom model",
    custom_model_pth=r"r:\delovno\nejc\models\HRNet_barrow_stone_02.pth.tar",
    roundness=0,
    min_area=0,
    save_ml_output=True
)

main_routine(my_input)
