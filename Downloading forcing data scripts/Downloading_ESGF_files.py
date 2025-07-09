import os
import requests
from tqdm import tqdm
import httpx
import netCDF4
import xarray as xr

# Define parameters
models = ['EC-Earth3', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'TaiESM1']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
variables = ['pr', 'prw', 'tas', 'hfls', 'wap', 'ua', 'va', 'hus', 'sftlf']
variant_label = 'r1i1p1f1'
filename_download_links = 'download_links.txt'

def find_download_links_esgf():
    esgf_nodes = ["https://esgf-node.llnl.gov/esg-search/search",
                  "https://esgf-data.dkrz.de/esg-search/search",
                  "https://esgf.ceda.ac.uk/esg-search/search"]
    all_links = []
    for model in models:
        for scenario in scenarios:
            for variable in variables:
                if variable == 'sftlf': frequency = 'fx'
                else: frequency = 'day'
                page_counter = 1
                offset = 0
                while True:
                    results_added = False
                    params = {
                        "project": "CMIP6",
                        "experiment_id": scenario,
                        "variable_id": variable,
                        "frequency": frequency,
                        "variant_label": variant_label,
                        "source_id": model,
                        "type": "File",
                        "format": "application/solr+json",
                        "limit": 1000,
                        "offset": offset
                    }
                    print(f"Querying: Model={model}, Scenario={scenario}, Variable={variable}, Offset={offset}, Page={page_counter}")
                    for BASE_URL in esgf_nodes:
                        try:
                            response = requests.get(BASE_URL, params=params)
                            response.raise_for_status()
                            data = response.json()
                            results = data.get('response', {}).get('docs', [])
                            for result in results:
                                urls = result.get('url', [])
                                for url in urls:
                                    if "HTTPServer" in url:
                                        all_links.append(url.split('|')[0])
                                        results_added = True
                        except Exception as e:
                            print(f"Error querying {BASE_URL} for {model}-{scenario}-{variable}: {e}")
                    if not results_added:
                        break
                    offset += 1000
                    page_counter += 1
    print(f"\nTotal unique download links found: {len(all_links)}")
    with open(filename_download_links, "w") as f:
        f.write("\n".join(all_links))

def download_file(url, target_directory, node):
    local_filename = os.path.join(target_directory, url.split('/')[-1])
    try:
        with httpx.Client() as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with open(local_filename, "wb") as f, tqdm(
                    desc=f"Downloading {local_filename} From: {node}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_bytes(chunk_size=100000):
                        f.write(chunk)
                        bar.update(len(chunk))
        print(f"\nFile downloaded: {local_filename}\n")
        return True
    except:
        print(f'\nDownload failed from: {url}\n')
        return False

def parse_file(file):
    urls = []
    with open(file, 'r') as file:
        for line in file:
            url = line.strip()
            if url:
                urls.append(url)
    return urls

def start_downloading_files():
    all_urls = parse_file(filename_download_links)
    try:
        for model in models:
            target_directory = fr'F:/Forcing_data_{model}'
            if not os.path.isdir(target_directory):
                os.makedirs(target_directory)
            for scenario in scenarios:
                for variable in variables:
                    if variable == 'prw': frequency = 'Eday'
                    if variable == 'sftlf': frequency = 'fx'
                    else: frequency = 'day'
                    search_key = f'{variable}_{frequency}_{model}_{scenario}_{variant_label}'
                    relevant_urls = [url for url in all_urls if search_key in url]
                    for url in relevant_urls:
                        filename = url.split('/')[-1]
                        full_path = os.path.join(target_directory, filename)
                        if not os.path.exists(full_path):
                            if download_file(url, target_directory, url.split('/')[2]):
                                if filename.endswith('.nc'):
                                    try:
                                        netCDF4.Dataset(full_path, 'r')
                                    except:
                                        try:
                                            xr.open_dataset(full_path)
                                        except:
                                            print(f'Corrupted file: {filename}.')
                                            os.remove(full_path)
                                            continue
                                else:
                                    continue
                            else:
                                continue

    except KeyboardInterrupt:
        if 'full_path' in locals() and os.path.exists(full_path):
            os.remove(full_path)
        print(f"\n\nProcess interrupted by user. Deleted uncompleted file {filename} Exiting gracefully.")
    except Exception as e:
        print(f'Unknown error occurred, potentially out of disc space. Error: {e}')

find_download_links_esgf()
start_downloading_files()