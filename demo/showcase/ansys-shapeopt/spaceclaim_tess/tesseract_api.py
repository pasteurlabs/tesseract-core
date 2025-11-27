# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import zipfile
from pathlib import Path, WindowsPath
from tempfile import TemporaryDirectory

import numpy as np
import trimesh
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Float32

# Example SpaceClaim .exe and script file Paths
# spaceclaim_exe = "F:\\Ansys installations\\ANSYS Inc\\v241\\scdm\\SpaceClaim.exe"
# spaceclaim_script = "geometry_generation.scscript"  # Relies on being executed in same directory as tesseract_api.py

#
# Schemata
#


class InputSchema(BaseModel):
    """Input schema for bar geometry design and SDF generation."""

    differentiable_parameters: list[
        Array[
            (None,),
            Float32,
        ]
    ] = Field(
        description=(
            "Angular positions around the unit circle for the bar geometry. "
            "The shape is (num_bars+1, 2), where num_bars is the number of bars "
            "and the second dimension has the start then end location of each bar."
            "The first (+1) entry represents the two z height coordinates for the cutting plane which combine with "
            "a third fixed coordinate centered on the grid with z = grid_height / 2"
        )
    )

    non_differentiable_parameters: list[
        Array[
            (None,),
            Float32,
        ]
    ] = Field(
        description=(
            "Flattened array of non-differentiable geometry parameters. "
            "The shape is (2), the first float is the maximum height (mm) of the "
            "grid (pre z-plane cutting). The second is the beam thickness (mm)."
        )
    )

    static_parameters: list[list[int]] = Field(
        description=("List of integers used to construct the geometry."),
        default=[],
    )

    string_parameters: list[str] = Field(
        description=(
            "Two string parameters for geometry construction. "
            "First str is Path to SpaceClaim executable. "
            "Second str is Path to SpaceClaim Script (.scscript)."
        )
    )


class TriangularMesh(BaseModel):
    """Triangular mesh representation with fixed-size arrays."""

    points: Array[(None, 3), Float32] = Field(description="Array of vertex positions.")
    faces: Array[(None, 3), Float32] = Field(
        description="Array of triangular faces defined by indices into the points array."
    )


class OutputSchema(BaseModel):
    """Output schema for generated geometry."""

    meshes: list[TriangularMesh] = Field(
        description="Triangular meshes representing the geometries"
    )


#
# Helper functions
#


def build_geometries(
    differentiable_parameters: list[np.ndarray],
    non_differentiable_parameters: list[np.ndarray],
    static_parameters: list[list[int]],
    string_parameters: list[str],
) -> list[trimesh.Trimesh]:
    """Build SpaceClaim geometries from the parameters by modifying template .scscript.

    Returns a list of trimesh objects.
    """
    spaceclaim_exe = Path(string_parameters[0])
    spaceclaim_script = Path(string_parameters[1])

    with TemporaryDirectory() as temp_dir:
        prepped_script_path = _prep_scscript(
            temp_dir,
            spaceclaim_script,
            differentiable_parameters,
            non_differentiable_parameters,
        )
        run_spaceclaim(spaceclaim_exe, prepped_script_path)

        meshes = []
        for output_stl in sorted(Path(temp_dir).glob("*.stl")):
            mesh = trimesh.load(output_stl)
            meshes.append(mesh)

    return meshes


def _prep_scscript(
    temp_dir: TemporaryDirectory,
    spaceclaim_script: Path,
    differentiable_parameters: list[np.ndarray],
    non_differentiable_parameters: list[np.ndarray],
) -> list[str]:
    """Take Tesseract inputs and place into a temp .scscript that will be used to run SpaceClaim.

    Return the Path location of this script and the output .stl.
    """
    # Define output file name and location
    output_file = os.path.join(
        temp_dir, "grid_fin"
    )  # .stl ending is included in .scscript
    prepped_script_path = os.path.join(temp_dir, os.path.basename(spaceclaim_script))
    shutil.copy(spaceclaim_script, prepped_script_path)

    # Define dict used to input params to .scscript
    # Converts np.float32 to python floats so string substitution is clean
    keyvalues = {}
    keyvalues["__output__"] = output_file
    keyvalues["__params__.zeds"] = [
        [float(geom_params[0]), float(geom_params[1])]
        for geom_params in differentiable_parameters
    ]
    keyvalues["__params__.height"] = non_differentiable_parameters[0][0]
    keyvalues["__params__.thickness"] = non_differentiable_parameters[0][1]

    num_of_batches = len(differentiable_parameters)  # number of geometries requested
    num_of_bars = (
        len(differentiable_parameters[0]) - 2
    ) // 2  # Use first geometry in batch to test number of beams

    assert num_of_bars == 8

    batch_starts = []
    batch_ends = []
    for i in range(num_of_batches):
        geom_starts = []
        geom_ends = []
        for j in range(num_of_bars):
            geom_starts.append(float(differentiable_parameters[i][j * 2 + 2]))
            geom_ends.append(float(differentiable_parameters[i][j * 2 + 3]))

        batch_starts.append(geom_starts)
        batch_ends.append(geom_ends)

    keyvalues["__params__.starts"] = (
        batch_starts  # convert to string to ease injection into file text
    )
    keyvalues["__params__.ends"] = batch_ends

    _find_and_replace_keys_in_archive(prepped_script_path, keyvalues)

    return prepped_script_path


def _safereplace(filedata: str, key: str, value: str) -> str:
    # ensure double backspace in windows path
    if isinstance(value, WindowsPath):
        value = str(value)
        value = value.replace("\\", "\\\\")
    else:
        value = str(value)
    return filedata.replace(key, value)


def _find_and_replace_keys_in_archive(file: Path, keyvalues: dict) -> None:
    # work on the zip in a temporary directory
    with TemporaryDirectory() as temp_dir:
        # extract zip
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # walk through the extracted files/folders
        for foldername, _subfolders, filenames in os.walk(temp_dir):
            for filename in filenames:
                # read in file
                filepath = os.path.join(foldername, filename)
                try:
                    with open(filepath) as f:
                        filedata = f.read()
                except Exception:
                    filedata = None

                # find/replace
                if filedata:
                    for key, value in keyvalues.items():
                        if value is not None:
                            filedata = _safereplace(filedata, key, value)

                    # write to file
                    with open(filepath, "w") as f:
                        f.write(filedata)

        # write out all files back to zip
        with zipfile.ZipFile(file, "w") as zip_ref:
            for foldername, _subfolders, filenames in os.walk(temp_dir):
                for filename in filenames:
                    filepath = os.path.join(foldername, filename)
                    zip_ref.write(
                        filepath,
                        arcname=os.path.relpath(filepath, start=temp_dir),
                    )


def run_spaceclaim(spaceclaim_exe: Path, spaceclaim_script: Path) -> None:
    """Runs SpaceClaim subprocess with .exe and script Path locations.

    Returns the subprocess return code.
    """
    env = os.environ.copy()
    cmd = str(
        f'"{spaceclaim_exe}" /UseLicenseMode=True /Welcome=False /Splash=False '
        + f'/RunScript="{spaceclaim_script}" /ExitAfterScript=True /Headless=True'
    )

    result = subprocess.run(
        cmd,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    return result.returncode


#
# Tesseract endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Create a SpaceClaim geometry based on input parameters.

    Returns TraingularMesh objects.
    """
    trimeshes = build_geometries(
        differentiable_parameters=inputs.differentiable_parameters,
        non_differentiable_parameters=inputs.non_differentiable_parameters,
        static_parameters=inputs.static_parameters,
        string_parameters=inputs.string_parameters,
    )

    return OutputSchema(
        meshes=[
            TriangularMesh(
                points=mesh.vertices.astype(np.float32),
                faces=mesh.faces.astype(np.int32),
            )
            for mesh in trimeshes
        ]
    )
