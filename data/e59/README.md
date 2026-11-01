# E59 data

`wind_hornsrev_2015_2024.json`: hourly wind speed at 100 m, 2015-01-01 00:00 to 2024-12-31 23:00 UTC (87,672 values,
none missing), ERA5 reanalysis at the Horns Rev offshore wind farm (requested 55.52°N 7.91°E; the API's grid point is
55.571°N 7.834°E).

- Source: Open-Meteo historical weather API, `https://archive-api.open-meteo.com/v1/archive?latitude=55.52&longitude=7.91&start_date=2015-01-01&end_date=2024-12-31&hourly=wind_speed_100m&timezone=GMT&wind_speed_unit=ms`,
  downloaded 1 Nov 2026 with the user's permission.
- Underlying data: Copernicus Climate Change Service, ERA5 hourly data (Hersbach et al. 2020).
- Licence: CC BY 4.0 (Open-Meteo); attribute "Weather data by Open-Meteo.com" and ERA5/Copernicus.
- SHA-256: `7d55be6177241d07b01ba6afe18d1958104e48b953431ac05dbb90a42daf3f7e`.

The file is read as JSON data only. E59's transform (registered in `experiments/registry/E59.json`): square root, minus
the month-by-hour mean of the training years 2015-2021, divided by the training standard deviation.
