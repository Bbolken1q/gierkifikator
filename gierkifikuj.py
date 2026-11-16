from face2face import Face2Face
from media_toolkit import VideoFile, ImageFile, MediaList, MediaDict
import numpy as np

output_folder = "./out"

swapped_img = Face2Face().swap_img_to_img("./gierek.png", "./target.png", enhance_face_model="gfpgan_1.4")
ImageFile().from_np_array(swapped_img).save(f"{output_folder}/output.png")