import cdsapi
import xarray as xr
from datetime import datetime
import os
import glob
import zipfile
import shutil
import re
import pandas as pd
import numpy as np
import tempfile
from zipfile import ZipFile
import streamlit as st

def add_absolute_time(ds):
    """Dodaje współrzędną absolute_time na podstawie atrybutu FORECAST.
    Zmienia wartości współrzędnej 'time' na absolutny czas, nie zmieniając jej nazwy."""

    if "FORECAST" in ds.attrs:
        match = re.search(r'(\d{8})\+', ds.attrs['FORECAST'])
        if match:
            base_time = pd.to_datetime(match.group(1), format="%Y%m%d")
            abs_time = base_time + pd.to_timedelta(ds.time.values, unit="h")
            ds = ds.assign_coords(time=abs_time)
    return ds

import io

def get_cams_air_quality(today_str=datetime.utcnow().strftime("%Y-%m-%d")):
    """
    Pobiera CAMS Europe air quality forecasts dla bieżącego dnia i zwraca xarray Dataset.
    Plik nie jest zapisywany na dysku, działa całkowicie w pamięci.
    """
    dataset = "cams-europe-air-quality-forecasts"

    request = {
        "variable": [
            "alder_pollen",
            "ammonia",
            "birch_pollen",
            "carbon_monoxide",
            "grass_pollen",
            "mugwort_pollen",
            "nitrogen_dioxide",
            "nitrogen_monoxide",
            "olive_pollen",
            "ozone",
            "particulate_matter_2.5um",
            "particulate_matter_10um",
            "ragweed_pollen",
            "sulphur_dioxide"
        ],
        "model": ["ensemble"],
        "level": ["0"],
        "date": [f"{today_str}/{today_str}"],
        "type": ["forecast"],
        "time": ["00:00"],
        "leadtime_hour": [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
            "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
            "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
            "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
            "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
            "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
            "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
            "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
            "90", "91", "92", "93", "94", "95", "96"
        ],
        "data_format": "netcdf_zip",
        "area": [56, 7, 47, 26]
    }
    st.info(f"Start trying ...")
    try:
        cds_url = os.environ.get("CDSAPI_URL")
        cds_key = os.environ.get("CDSAPI_KEY")

        client = cdsapi.Client(url=cds_url, key=cds_key)
        print("zmiana jest kurde")
        
        print(f"ℹ️  Klient CAMS zainicjalizowany z URL: {cds_url}")

        zip_path = "/tmp/cams_data.zip"
        st.info(f"🔄 Pobieranie danych CAMS do {zip_path}...")
        
        try:
            client.retrieve(dataset, request).download(zip_path)
        except Exception as e:
            st.error(f"❌ Błąd pobierania danych: {e}")
            return None
        
        # Sprawdzenie czy plik istnieje i jego rozmiaru
        if not os.path.exists(zip_path):
            st.error("❌ Plik ZIP nie został pobrany")
            return None
        
        st.info(f"✅ Plik pobrany, rozmiar: {os.path.getsize(zip_path)} bajtów")
        
        # Rozpakowanie ZIP
        try:
            with ZipFile(zip_path) as zf:
                nc_name = zf.namelist()[0]
                with zf.open(nc_name) as nc_file:
                    ds = xr.open_dataset(nc_file)
        except Exception as e:
            st.error(f"❌ Błąd wczytywania ZIP/NetCDF: {e}")
            return None
        
        st.info("✅ Dane wczytane do xarray w pamięci")
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "cams_data.zip")
            print(f"🔄 Pobieranie danych CAMS do tymczasowego pliku {zip_path}...")
    
            # Pobranie danych do pliku ZIP
            try:
                client.retrieve(dataset, request).download(zip_path)
                print(zip_path.exists())
                print(zip_path.stat().st_size)
            except Exception as e:
                raise RuntimeError(f"❌ Błąd pobierania danych z CDS: {e}")
    
            # Sprawdzenie czy plik został pobrany
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"❌ Plik ZIP nie został pobrany: {zip_path}")
    
            # Rozpakowanie NetCDF z ZIP
            try:
                with ZipFile(zip_path) as zf:
                    namelist = zf.namelist()
                    if len(namelist) == 0:
                        raise ValueError("❌ ZIP jest pusty, brak plików do otwarcia.")
                    nc_name = namelist[0]
                    print(f"📦 Otwieranie pliku NetCDF w ZIP: {nc_name}")
                    with zf.open(nc_name) as nc_file:
                        try:
                            ds = xr.open_dataset(nc_file)
                        except Exception as e:
                            raise RuntimeError(f"❌ Błąd wczytywania NetCDF do xarray: {e}")
            except BadZipFile as e:
                raise RuntimeError(f"❌ Nie udało się otworzyć ZIP: {e}")
    
            print("✅ Dane wczytane do xarray w pamięci")
            '''


        # Selekcja poziomu 0 i squeeze
        ds = ds.sel(level=0).squeeze()

        # Zmiana nazw zmiennych
        var_rename = {
            "apg_conc": "alder_pollen",
            "nh3_conc": "ammonia",
            "bpg_conc": "birch_pollen",
            "co_conc": "carbon_monoxide",
            "gpg_conc": "grass_pollen",
            "mpg_conc": "mugwort_pollen",
            "no2_conc": "nitrogen_dioxide",
            "no_conc": "nitrogen_monoxide",
            "opg_conc": "olive_pollen",
            "o3_conc": "ozone",
            "pm2p5_conc": "particulate_matter_2.5um",
            "pm10_conc": "particulate_matter_10um",
            "rwpg_conc": "ragweed_pollen",
            "so2_conc": "sulphur_dioxide"
        }
        ds = ds.rename(var_rename)

        # Poprawka współrzędnych
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            lats = np.round(ds.latitude.values, 3)
            lons = np.round(ds.longitude.values, 3)
            lons_corrected = np.where(lons > 180, lons - 360, lons)
            ds = ds.assign_coords(latitude=lats, longitude=lons_corrected)

        # Tu możesz wywołać add_absolute_time(ds) jeśli potrzebujesz
        ds = add_absolute_time(ds)
        print(ds)

        print("✅ Dane pobrane i przetworzone w pamięci.")
        return ds

    except Exception as e:
        import traceback
        print(f"❌ Błąd pobierania lub przetwarzania danych CAMS:\n{traceback.format_exc()}")
        return None
