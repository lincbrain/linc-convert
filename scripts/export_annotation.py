import os
import numpy as np
import webknossos as wk
from tifffile import imwrite

ANNOTATION_ID = '<annotation_id>'
SEGMENT_IDS = [1,2]
MAG = wk.Mag('8-8-1')
annotation_directory='~/example_export'

with wk.client.context.webknossos_context(url=os.environ.get('WK_URL'),token=os.environ.get('WK_TOKEN')):
    dataset = wk.Annotation.open_as_remote_dataset(
        ANNOTATION_ID, 
        annotation_type='Volume',
        webknossos_url=os.environ.get('WK_URL')
    )
    mag_view = dataset.get_segmentation_layers()[0].get_mag(MAG)

z = mag_view.bounding_box.topleft.z
with mag_view.get_buffered_slice_reader() as reader:
    for slice_data in reader:
        print(slice_data)
        slice_data = slice_data[0]  # First channel only
        for segment_id in SEGMENT_IDS:
            segment_mask = (slice_data == segment_id).astype(
                np.uint8
            ) * 255  # Make a binary mask 0=empty, 255=segment
            segment_mask = segment_mask.T  # Tiff likes the data transposed
            imwrite(
                f"{annotation_directory}/seg{segment_id:04d}_mag{MAG}_z{z:04d}.tiff",
                segment_mask,
            )

        print(f'Downloaded z={z:04d}')
        z += MAG.z