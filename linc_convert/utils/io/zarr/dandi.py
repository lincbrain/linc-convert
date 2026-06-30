"""Helper functions to connect and read from Dandi."""


import requests
import zarr
from dandi.dandiapi import DandiAPIClient


def _get_dandi_zarr_id(
    dandiset_id: str,
    asset_path: str,
    api_url: str,
    api_key: str,
    version: str,
) -> str:
    """Resolve a DANDI asset path to a Zarr ID."""
    with DandiAPIClient(api_url=api_url, token=api_key) as client:
        dandiset = client.get_dandiset(dandiset_id, version)
        asset = dandiset.get_asset_by_path(asset_path)

    zarr_id = getattr(asset, "zarr", None)
    if zarr_id is None:
        raise RuntimeError(
            f"Asset '{asset_path}' in dandiset '{dandiset_id}' "
            f"is not a Zarr-backed asset."
        )

    return zarr_id


def _open_lincbrain_zarr(
    zarr_id: str,
    api_url: str,
    api_key: str,
) -> zarr.Group:
    """Open a private LINC Brain Zarr using CloudFront signed cookies."""
    session = requests.Session()
    headers = {"Authorization": f"token {api_key}"}

    token_resp = session.get(f"{api_url}/auth/token", headers=headers)
    if token_resp.status_code != 200:
        raise RuntimeError("Failed to authenticate with LINC Brain API.")

    perms_resp = session.get(f"{api_url}/permissions/s3/", headers=headers)
    if perms_resp.status_code != 200:
        raise RuntimeError("Failed to obtain S3 access permissions.")

    cookies = perms_resp.cookies.get_dict()
    if not cookies:
        raise RuntimeError("No CloudFront cookies returned.")

    store_url = f"https://neuroglancer.lincbrain.org/zarr/{zarr_id}/"

    storage_options = {
        "client_kwargs": {
            "cookies": cookies,
        }
    }

    return zarr.api.synchronous.open_group(
        store_url,
        mode="r",
        zarr_format=3,
        storage_options=storage_options,
    )


def _open_public_dandi_zarr(zarr_id: str) -> zarr.Group:
    """Open a public DANDI Zarr stored on S3."""
    return zarr.api.synchronous.open_group(
        f"s3://dandiarchive/zarr/{zarr_id}/",
        mode="r",
    )


def open_dandi_zarr_group(
    *,
    dandiset_id: str,
    asset_path: str,
    api_key: str,
    api_url: str,
    version: str,
) -> zarr.Group:
    """Open a Zarr group from DANDI or LINC Brain infrastructure."""
    zarr_id = _get_dandi_zarr_id(
        dandiset_id,
        asset_path,
        api_url=api_url,
        api_key=api_key,
        version=version,
    )

    if "lincbrain.org" in api_url:
        return _open_lincbrain_zarr(
            zarr_id,
            api_url=api_url,
            api_key=api_key,
        )

    return _open_public_dandi_zarr(zarr_id)
