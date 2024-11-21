import os
from dandi.dandiapi import DandiAPIClient, DandiInstance

known_instances = {"linc": DandiInstance(
    "linc",
    "https://lincbrain.org",
    "https://api.lincbrain.org/api"
)}


os.environ["DANDI_API_KEY"] = ""


def download_dandiset_file(dandiset_id, file_path, output_dir, instance="linc"):


    
    """
    Downloads a specific file from a DANDI dataset using the specified instance.

    Parameters:
        dandiset_id (str): The ID of the dandiset (e.g., "000004").
        file_path (str): The path of the file within the dandiset (e.g., "derivatives/.../file.tiff").
        output_dir (str): The directory where the file should be downloaded.
        instance (str): The DANDI instance to use. Default is "linc".
    """
    # Retrieve the instance details
    dandi_instance = known_instances.get(instance)
    if not dandi_instance:
        raise ValueError(f"Unknown instance: {instance}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download the file
    with DandiAPIClient(dandi_instance.api, token="021b9298a490b4c4fcfa020f7f672ae571a8f1f2") as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(file_path)
        if not asset:
            raise FileNotFoundError(f"File '{file_path}' not found in dandiset '{dandiset_id}'.")

        print(f"Downloading {file_path} from dandiset {dandiset_id} on instance {instance}...")
        asset.download(os.path.join(output_dir, os.path.basename(file_path)))

    print(f"Download complete. File saved to {output_dir}")


# Example usage:
download_dandiset_file(
    dandiset_id="000004",
    file_path="derivatives/AI7_EH5f1_z01_y01_demixed/AI7_EH5f1_z01_y01_demixed_plane001_c1.tiff",
    output_dir="./downloads",
    instance="linc"
)