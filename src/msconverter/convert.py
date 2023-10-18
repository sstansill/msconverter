from casacore import tables
import numpy as np
import xarray as xr
from numcodecs import Blosc
import dask.array as da
import shutil
from tqdm import tqdm


# Named data variables (ie data)
# All other columns considered as metadata / coordinates
data_variable_columns = [
    "FLOAT_DATA",
    "DATA",
    "CORRECTED_DATA",
    "WEIGHT_SPECTRUM",
    "WEIGHT",
    "FLAG",
    "UVW",
    "TIME_CENTROID",
    "EXPOSURE",
]


# Rename MS columns to a DataArray/ Dataset variable
column_to_data_variable_names = {
    "FLOAT_DATA": "SPECTRUM",
    "DATA": "VISIBILITY",
    "CORRECTED_DATA": "VISIBILITY_CORRECTED",
    "WEIGHT_SPECTRUM": "WEIGHT",
    "WEIGHT": "WEIGHT",
    "FLAG": "FLAG",
    "UVW": "UVW",
    "TIME_CENTROID": "TIME_CENTROID",
    "EXPOSURE": "EFFECTIVE_INTEGRATION_TIME",
    "U": "U",
    "V": "V",
    "W": "W",
}


# The dimensions of a given column
column_dimensions = {
    "DATA": ("time", "baseline_id", "frequency", "polarization"),
    "CORRECTED_DATA": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT_SPECTRUM": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT": ("time", "baseline_id", "polarization"),
    "FLAG": ("time", "baseline_id", "frequency", "polarization"),
    "UVW": ("time", "baseline_id"),
    "U": ("time", "baseline_id"),
    "V": ("time", "baseline_id"),
    "W": ("time", "baseline_id"),
    "TIME_CENTROID": ("time", "baseline_id"),
    "EXPOSURE": ("time", "baseline_id"),
    "FLOAT_DATA": ("time", "baseline_id", "frequency", "polarization"),
}


# Rename coordinate columns to something sensible
column_to_coord_names = {
    "TIME": "time",
    "ANTENNA1": "baseline_antenna1_id",
    "ANTENNA2": "baseline_antenna2_id",
}


# Polarization IDs to Stokes parameter name
stokes_types = {
    0: "Undefined",
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
    13: "RX",
    14: "RY",
    15: "LX",
    16: "LY",
    17: "XR",
    18: "XL",
    19: "YR",
    20: "YL",
    21: "PP",
    22: "PQ",
    23: "QP",
    24: "QQ",
    25: "RCircular",
    26: "LCircular",
    27: "Linear",
    28: "Ptotal",
    29: "Plinear",
    30: "PFtotal",
    31: "PFlinear",
    32: "Pangle",
}


# Check that the time steps have a single unique value
def _check_interval_consistent(MeasurementSet):
    time_interval = np.unique(MeasurementSet.getcol("INTERVAL"))
    assert len(time_interval) == 1, "Interval is not consistent."
    return time_interval[0]


# Check that the time steps have a single unique value
def _check_exposure_consistent(MeasurementSet):
    exposure_time = np.unique(MeasurementSet.getcol("EXPOSURE"))
    assert len(exposure_time) == 1, "Exposure is not consistent."
    return exposure_time[0]


# Check that the FIELD_ID has a single unique value
def _check_single_field(MeasurementSet):
    field_id = np.unique(MeasurementSet.getcol("FIELD_ID"))
    assert len(field_id) == 1, "More than one field present."
    return field_id[0]


# Create the DataArray coordinates and add various metadata
def create_coordinates(xds, MeasurementSet, baseline_ant1_id, baseline_ant2_id):
    ###############################################################
    # Add metadata
    ###############################################################

    # Field subtable
    field = MeasurementSet.FIELD[0]

    # Assumes only a single FIELD_ID
    delay_direction = {
        "data": list(field.get("DELAY_DIR", [])),
        "attrs": {
            "units": "rad",
            "type": "sky_coord",
            "description": "Direction of delay center in right ascension and declination.",
        },
    }

    phase_direction = {
        "data": list(field.get("PHASE_DIR", [])),
        "attrs": {
            "units": "rad",
            "type": "sky_coord",
            "description": "Direction of phase center in right ascension and declination.",
        },
    }
    reference_direction = {
        "data": list(field.get("REFERENCE_DIR", [])),
        "attrs": {
            "units": "rad",
            "type": "sky_coord",
            "description": "Direction of reference direction in right ascension and declination. Used in single-dish to record the associated reference direction if position-switching has already been applied. For interferometric data, this is the original correlated field center, and may equal delay_direction or phase_direction.",
        },
    }
    name = field["NAME"]
    code = field["CODE"]

    field_id = _check_single_field(MeasurementSet)

    field_info = {
        "name": name,
        "code": code,
        # "delay_direction": delay_direction,
        # "phase_direction": phase_direction,
        # "reference_direction": reference_direction,
        "field_id": field_id,
    }
    xds.attrs["field_info"] = field_info

    ###############################################################
    # Add coordinates
    ###############################################################

    coords = {
        # Ignore time because we populate this 'on the fly'
        # 'time': unique_times,
        "baseline_antenna1_id": ("baseline_id", baseline_ant1_id),
        "baseline_antenna2_id": ("baseline_id", baseline_ant2_id),
        "baseline_id": np.arange(len(baseline_ant1_id)),
    }

    # Add frequency coordinates
    frequencies = MeasurementSet.SPECTRAL_WINDOW[0].get("CHAN_FREQ", [])

    # Remove NaN values
    coords["frequency"] = frequencies[~np.isnan(frequencies)]

    # Add polatization coordinates
    polarizations = MeasurementSet.POLARIZATION[0].get("CORR_TYPE", [])
    coords["polarization"] = np.vectorize(stokes_types.get)(polarizations)

    # Add named coordinates
    xds = xds.assign_coords(coords)

    ###############################################################
    # Add metadata to coordinates
    ###############################################################

    xds.frequency.attrs["reference_frequency"] = MeasurementSet.SPECTRAL_WINDOW[0].get(
        "REF_FREQUENCY", ""
    )
    xds.frequency.attrs["effective_channel_width"] = MeasurementSet.SPECTRAL_WINDOW[
        0
    ].get("EFFECTIVE_BW", "")[0]

    channel_widths = MeasurementSet.SPECTRAL_WINDOW[0].get("CHAN_WIDTH", "")

    if not isinstance(channel_widths, str):
        unique_chan_width = np.unique(
            channel_widths[np.logical_not(np.isnan(channel_widths))]
        )
        xds.frequency.attrs["channel_width"] = {
            "data": np.abs(unique_chan_width[0]),
            "attrs": {"type": "quanta", "units": "Hz"},
        }

    return xds


def add_encoding(xds, compressor):
    for da_name in list(xds.data_vars):
        xds[da_name].encoding = {"compressor": compressor}


# Reshape a column
def reshape_column(
    column_data: np.ndarray,
    cshape: tuple[int],
    time_indices: np.ndarray,
    time,
    baselines: np.ndarray,
):
    # full data is the maximum of the data shape and chunk shape dimensions for each time interval

    fulldata = np.full(cshape + column_data.shape[1:], np.nan, dtype=column_data.dtype)

    fulldata[time_indices, baselines] = column_data

    # Use dask here to ensure xarray knows to use distributed methods on the zarr store
    fulldata = da.from_array(fulldata)

    return fulldata


# All baseline pairs needed to reshape data columns
def get_baselines(MeasurementSet: tables.table) -> np.ndarray:
    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = MeasurementSet.getcol("ANTENNA1", 0, -1), MeasurementSet.getcol(
        "ANTENNA2", 0, -1
    )
    baselines = np.array(
        [
            str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
            for ll in np.unique(np.hstack([ant1[:, None], ant2[:, None]]), axis=0)
        ]
    )

    return baselines


# Need to convert pairs of antennas into indices
# Integer indices required for array broadcasting
def get_baseline_indices(MeasurementSet: tables.table) -> tuple:
    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = MeasurementSet.getcol("ANTENNA1", 0, -1), MeasurementSet.getcol(
        "ANTENNA2", 0, -1
    )
    baselines = np.array(
        [
            str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
            for ll in np.unique(np.hstack([ant1[:, None], ant2[:, None]]), axis=0)
        ]
    )

    baseline_ant1_id, baseline_ant2_id = np.array(
        [tuple(map(int, x.split("_"))) for x in baselines]
    ).T

    return (baseline_ant1_id, baseline_ant2_id)


# Add column metadata for a single column
# Must be called after the DataArrays have been created
def create_attribute_metadata(column_name, MeasurementSet):
    attrs_metadata = {}

    if column_name in ["U", "V", "W"]:
        name = "UVW"
        column_description = MeasurementSet.getcoldesc(name)

        column_info = column_description.get("INFO", {})

        attrs_metadata["type"] = column_info.get("type", "None")
        attrs_metadata["ref"] = column_info.get("Ref", "None")

        attrs_metadata["units"] = column_description.get("keywords", {}).get(
            "QuantumUnits", ["None"]
        )[0]

    else:
        column_description = MeasurementSet.getcoldesc(column_name)

        column_info = column_description.get("INFO", {})

        attrs_metadata["type"] = column_info.get("type", "None")
        attrs_metadata["ref"] = column_info.get("Ref", "None")

        attrs_metadata["units"] = column_description.get("keywords", {}).get(
            "QuantumUnits", ["None"]
        )[0]

    return attrs_metadata


# Separate function to add time attributes. Must happen after the
# time coordinate is created
def add_time(xds, MeasurementSet):
    interval = _check_interval_consistent(MeasurementSet)
    exposure = _check_exposure_consistent(MeasurementSet)
    time_description = MeasurementSet.getcoldesc("TIME")

    xds.time.attrs["type"] = time_description.get("MEASINFO", {}).get("type", "None")

    xds.time.attrs["Ref"] = time_description.get("MEASINFO", {}).get("Ref", "None")

    xds.time.attrs["units"] = time_description.get("keywords", {}).get(
        "QuantumUnits", ["None"]
    )[0]

    xds.time.attrs["time_scale"] = (
        time_description.get("keywords", {}).get("MEASINFO", {}).get("Ref", "None")
    )

    xds.time.attrs["integration_time"] = interval

    xds.time.attrs["effective_integration_time"] = exposure


def MS_chunk_to_zarr(
    xds,
    MeasurementSet_chunk,
    time,
    baseline_ant1_id,
    baseline_ant2_id,
    column_names,
    outfile,
    append,
):
    # Get all baseline pairs
    antenna1, antenna2 = (
        MeasurementSet_chunk.getcol("ANTENNA1"),
        MeasurementSet_chunk.getcol("ANTENNA2"),
    )

    # Get dimensions of data
    time_indices = np.searchsorted([time], MeasurementSet_chunk.getcol("TIME"))
    data_shape = (1, len(baseline_ant1_id))

    baseline_combinations = [
        str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
        for ll in np.hstack([antenna1[:, None], antenna2[:, None]])
    ]

    baselines = get_baselines(MeasurementSet_chunk)

    baseline_indices = np.searchsorted(baselines, baseline_combinations)

    # Must loop over each column to create an xarray DataArray for each
    for column_name in column_names:
        # Only handle certain columns. Others are metadata/coordinates
        if column_name in data_variable_columns:
            column_data = MeasurementSet_chunk.getcol(column_name)

            # UVW column must be split into u, v, and w
            if column_name == "UVW":
                subcolumns = [column_data[:, 0], column_data[:, 1], column_data[:, 2]]
                subcolumn_names = ["U", "V", "W"]

                for data, name in zip(subcolumns, subcolumn_names):
                    reshaped_column = reshape_column(
                        data,
                        data_shape,
                        time_indices,
                        time,
                        baseline_indices,
                    )

                    # Create a DataArray instead of appending immediately to Dataset
                    # so time coordinates can be updated
                    xda = xr.DataArray(
                        reshaped_column,
                        dims=column_dimensions.get(name),
                    ).assign_coords(time=("time", [time]))

                    # Add the DataArray to the Dataset
                    xds[column_to_data_variable_names.get(name)] = xda

            else:
                reshaped_column = reshape_column(
                    column_data,
                    data_shape,
                    time_indices,
                    time,
                    baseline_indices,
                )

                # Create a DataArray instead of appending immediately to Dataset
                # so time coordinates can be updated
                xda = xr.DataArray(
                    reshaped_column,
                    dims=column_dimensions.get(column_name),
                ).assign_coords(time=("time", [time]))

                # Add the DataArray to the Dataset
                xds[column_to_data_variable_names.get(column_name)] = xda

    # Add column metadata at the end
    # Adding metadata to a variable means the variable must already exist
    if not append:
        for column_name in column_names:
            if column_name in data_variable_columns:
                if column_name == "UVW":
                    subcolumn_names = ["U", "V", "W"]

                    for subcolumn_name in subcolumn_names:
                        xds[column_to_data_variable_names[subcolumn_name]].attrs.update(
                            create_attribute_metadata(column_name, MeasurementSet_chunk)
                        )

                else:
                    xds[column_to_data_variable_names[column_name]].attrs.update(
                        create_attribute_metadata(column_name, MeasurementSet_chunk)
                    )

    # Each time value is a new chunk
    # Not clear is there's a way to change this behaviour
    if append:
        xds.to_zarr(store=outfile, append_dim="time", consolidated=True)
    else:
        add_time(xds, MeasurementSet_chunk)
        xds.to_zarr(store=outfile, mode="w", consolidated=True)


# Chunk tuning required for optimal performance
# Dask recommends at least 100MB chunk sizes
# https://docs.dask.org/en/latest/array-best-practices.html
def rechunk(outfile_tmp, outfile, compress):
    xds = xr.open_zarr(outfile_tmp)

    # Manually delete encoding otherwise to_zarr() fails
    # see https://stackoverflow.com/questions/67476513/zarr-not-respecting-chunk-size-from-xarray-and-reverting-to-original-chunk-size
    for var in xds:
        del xds[var].encoding["chunks"]

    # Compression slows down the conversion a lot
    if compress:
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        add_encoding(xds, compressor)

    # Chunks method is number of pieces in the chunk
    # not the number of chunks. -1 gives a single chunk
    # xds = xds.chunk({'time': -1})
    xds = xds.chunk({"time": 100})
    xds.to_zarr(outfile, mode="w", compute=True, consolidated=True)

    shutil.rmtree(f"{outfile_tmp}")


def convert(infile, outfile, compress=True):
    # Create temporary zarr store
    # Eventually clean-up by rechunking the zarr datastore
    outfile_tmp = outfile + ".tmp"

    with tables.table(infile) as MeasurementSet:
        # Initial checks
        time_interval = _check_interval_consistent(MeasurementSet)
        exposure_time = _check_exposure_consistent(MeasurementSet)
        field_id = _check_single_field(MeasurementSet)

        # Get the unique timestamps
        time_values = MeasurementSet.getcol("TIME").flatten()
        unique_time_values = np.unique(time_values)

        # Get unique baseline indices
        baseline_ant1_id, baseline_ant2_id = get_baseline_indices(MeasurementSet)

        # Base Dataset
        xds_base = xr.Dataset()

        # Add dimensions, coordinates and attributes here to prevent
        # repetition. Deep copy made for each loop iteration
        xds_base = create_coordinates(
            xds_base, MeasurementSet, baseline_ant1_id, baseline_ant2_id
        )

        column_names = MeasurementSet.colnames()

        # The xarray Dataset for each time value will become a Zarr chunk
        for time in tqdm(unique_time_values, desc="Converting to Zarr", unit="time values"):
            MS_chunk_to_zarr(
                xds_base.copy(deep=True),
                MeasurementSet.selectrows(np.where(time_values == time)[0]),
                time,
                baseline_ant1_id,
                baseline_ant2_id,
                column_names,
                outfile_tmp,
                append=(time != unique_time_values[0]),
            )
            # delayed_conversions.append(dask.delayed(MS_chunk_to_zarr(xds_base.copy(deep=True), MeasurementSet.query('TIME == $time'), time, time_indices, baseline_ant1_id, baseline_ant2_id, column_names, outfile_tmp, append=True)))

        # dask.compute(delayed_conversions)
        rechunk(outfile_tmp, outfile, compress)
