import os
import asyncio
import json
import datetime
import aiohttp
import traceback
import argparse

# Assuming artifact_manager is in the same directory or PYTHONPATH is set up
from artifact_manager.core import HyphaConnection, Config

VERIFICATION_LOG_FILE_TEMPLATE = "verification_errors_{}.log"

async def _find_zip_files_recursive(artifact_manager, dataset_alias_full: str, current_dir_path: str = "") -> list[str]:
    """
    Recursively finds all .zip files within a dataset artifact.
    Returns a list of their full paths relative to the artifact root.
    """
    zip_file_paths = []
    normalized_dir_path = current_dir_path.lstrip(os.sep) if current_dir_path else ""

    try:
        items = await artifact_manager.list_files(
            artifact_id=dataset_alias_full,
            dir_path=normalized_dir_path if normalized_dir_path else None
        )
        
        for item in items:
            item_relative_path = os.path.join(normalized_dir_path, item['name'])
            
            if item['type'] == 'file' and item['name'].endswith('.zip'):
                zip_file_paths.append(item_relative_path)
            elif item['type'] == 'directory':
                zip_file_paths.extend(
                    await _find_zip_files_recursive(artifact_manager, dataset_alias_full, item_relative_path)
                )
    except Exception as e:
        print(f"Error listing files in {dataset_alias_full} at path '{normalized_dir_path}': {e}")
        log_file_name = VERIFICATION_LOG_FILE_TEMPLATE.format(dataset_alias_full.split('/',1)[-1].replace('/','_'))
        with open(log_file_name, "a") as f_log:
            f_log.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f_log.write(f"Dataset: {dataset_alias_full}\n")
            f_log.write(f"Error during _find_zip_files_recursive at path '{normalized_dir_path}': {e}\n{traceback.format_exc()}\n\n")
    return zip_file_paths

async def verify_dataset_zips(artifact_manager, workspace_name: str, dataset_name_in_url: str, dataset_alias_full: str):
    """Verifies accessibility of .zip file contents in a committed dataset."""
    print(f"\nStarting verification for dataset: {dataset_alias_full}")

    log_file_name = VERIFICATION_LOG_FILE_TEMPLATE.format(dataset_name_in_url.replace('/', '_'))
    
    try:
        print(f"  Searching for .zip files in dataset {dataset_alias_full}...")
        zip_files_in_artifact = await _find_zip_files_recursive(artifact_manager, dataset_alias_full)
        
        if not zip_files_in_artifact:
            print(f"  No .zip files found in dataset {dataset_alias_full} to verify.")
            return

        print(f"  Found {len(zip_files_in_artifact)} .zip files. Verifying access to their contents...")

        async with aiohttp.ClientSession() as session:
            for zip_path_in_artifact in zip_files_in_artifact:
                url_zip_path = zip_path_in_artifact.replace(os.sep, '/') 
                # Construct the URL for listing zip contents (default is root of zip)
                zip_content_list_url = f"{Config.SERVER_URL}/{workspace_name}/artifacts/{dataset_name_in_url}/zip-files/{url_zip_path}"
                
                error_occurred = False
                error_message = ""

                try:
                    print(f"    Verifying: {zip_content_list_url}")
                    async with session.get(zip_content_list_url, timeout=Config.CONNECTION_TIMEOUT) as response:
                        if response.status == 200:
                            try:
                                content_list = await response.json()
                                if isinstance(content_list, list):
                                    print(f"      OK: Listed {len(content_list)} items in {url_zip_path}")
                                else:
                                    error_occurred = True
                                    error_message = f"Error: Response content for {url_zip_path} is not a list. Type: {type(content_list)}"
                            except aiohttp.ContentTypeError:
                                error_occurred = True
                                response_text = await response.text()
                                error_message = f"Error: Response for {url_zip_path} is not valid JSON. Status: {response.status}. Response text: {response_text[:200]}"
                            except json.JSONDecodeError:
                                error_occurred = True
                                response_text = await response.text()
                                error_message = f"Error: Failed to decode JSON for {url_zip_path}. Status: {response.status}. Response text: {response_text[:200]}"
                        else:
                            error_occurred = True
                            error_message = f"Error: HTTP status {response.status} for {url_zip_path}."
                            try:
                                response_text = await response.text()
                                error_message += f" Response text: {response_text[:200]}"
                            except Exception:
                                error_message += " Could not retrieve response text."
                except asyncio.TimeoutError:
                    error_occurred = True
                    error_message = f"Error: Request timed out for {url_zip_path}."
                except aiohttp.ClientError as e:
                    error_occurred = True
                    error_message = f"Error: Client error for {url_zip_path}: {e}"
                except Exception as e:
                    error_occurred = True
                    error_message = f"Error: An unexpected error occurred for {url_zip_path}: {e}\n{traceback.format_exc()}"
                
                if error_occurred:
                    print(f"      FAIL: {error_message}")
                    with open(log_file_name, "a") as f_log:
                        f_log.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                        f_log.write(f"Dataset: {dataset_alias_full}\n")
                        f_log.write(f"Zip URL: {zip_content_list_url}\n")
                        f_log.write(f"Error: {error_message}\n\n")
        
        log_exists = os.path.exists(log_file_name)
        num_errors = 0
        if log_exists:
            with open(log_file_name, "r") as f_log:
                num_errors = f_log.read().count("Error:") 

        if num_errors > 0:
            print(f"Verification finished for dataset {dataset_alias_full}. {num_errors} error(s) logged to {log_file_name}")
        elif zip_files_in_artifact:
            print(f"Verification finished for dataset {dataset_alias_full}. All {len(zip_files_in_artifact)} zip files verified successfully.")
            if log_exists:
                try: os.remove(log_file_name)
                except: pass

    except Exception as e:
        print(f"  An error occurred during the verification setup for {dataset_alias_full}: {e}")
        with open(log_file_name, "a") as f_log:
            f_log.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f_log.write(f"Dataset: {dataset_alias_full}\n")
            f_log.write(f"General Error during verification: {e}\n{traceback.format_exc()}\n\n")

async def main():
    parser = argparse.ArgumentParser(description="Verify Zarr dataset zip file contents in Hypha.")
    # parser.add_argument("gallery_alias", type=str, help="The alias of the gallery (e.g., 'workspace_name/gallery_name') - currently unused by verify_dataset_zips but kept for context.")
    parser.add_argument("dataset_alias_full", type=str, help="The full alias of the dataset (e.g., 'workspace_name/dataset_name').")
    parser.add_argument("--client_id", type=str, default="reef-client-zip-verifier", help="Client ID for Hypha connection.")
    
    args = parser.parse_args()

    hypha_conn = HyphaConnection(server_url=Config.SERVER_URL, token=Config.WORKSPACE_TOKEN)
    
    try:
        await hypha_conn.connect(client_id=args.client_id)
        if not hypha_conn.artifact_manager:
            print("Failed to connect to Hypha or get artifact manager. Exiting.")
            return

        alias_parts = args.dataset_alias_full.split('/', 1)
        if len(alias_parts) < 2:
            print(f"Error: Could not parse workspace from dataset alias: {args.dataset_alias_full}")
            return
        workspace_name = alias_parts[0]
        dataset_name_for_url = alias_parts[1] # This is dataset_name or dataset_name/subpath if it's nested

        await verify_dataset_zips(
            artifact_manager=hypha_conn.artifact_manager,
            workspace_name=workspace_name,
            dataset_name_in_url=dataset_name_for_url,
            dataset_alias_full=args.dataset_alias_full
        )

    except Exception as e:
        print(f"An error occurred in main: {e}")
        traceback.print_exc()
    finally:
        if hypha_conn.api:
            await hypha_conn.disconnect()
        print("Verification script finished.")

if __name__ == "__main__":
    asyncio.run(main()) 