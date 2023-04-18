import subprocess
import re
import warnings
from typing import Callable

import numpy as np
import vtk
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
import time
import datetime
import asyncio
import os
import shutil

# Set parameters for Gascoigne/problem
niter = 4
Nx = 8
Ny = 8
image_dim = 2 ** (niter + 1) + 1  # Compute image dimensions (with prerefine 2)

# Check if Nx and Ny are bigger than Nyquist frequency
if Nx / 2 > 2 ** niter or Ny / 2 > 2 ** niter:
    warnings.warn("Either Nx or Ny is too large for the grid size")


def set_paramfile(a: np.ndarray, b: np.ndarray, process_id: int = 0):
    """
    Updates the parameters a, b, niter and resultsdir in the Gascoigne parameter file.

    Parameters
    ----------
    a : numpy array
        Array with coefficients for the RHS function
    b : numpy array
        Array with coefficients for the RHS function
    process_id : int
        Thread id used so different threads don't try to access the same files

    """

    # Check if a and b are one dimensional
    assert (a.ndim == 1 and b.ndim == 1)

    # Generate string representation for the Gascoigne parameter file
    a_str = str(a).strip("[] ").replace("\n", "")
    b_str = str(b).strip("[] ").replace("\n", "")

    # If a parameter file for this thread does not exist, make a new copy
    if ~os.path.exists(f'Gascoigne/heat{process_id}.param'):
        shutil.copy2('Gascoigne/heat.param', f'Gascoigne/heat{process_id}.param')

    # Read contents of file
    # Note: if file isn't read and written in byte format, '\r' is added after each line, which Gascoigne cannot handle
    with open(f'Gascoigne/heat{process_id}.param', 'rb') as file:
        data = file.read()

    # Replace values
    data = re.sub(b'\nniter[ \t]*\d', bytes(f'\nniter       {niter}', 'UTF-8)'), data)
    data = re.sub(b'\nresultsdir[ \t]*.*', bytes(f'\nresultsdir       Results{process_id}', 'UTF-8)'), data)
    data = re.sub(b'//Block RHS\n.*\n.*',
                  bytes(f'//Block RHS\na   {len(a)}  {a_str} \nb   {len(b)}  {b_str}', 'UTF-8)'),
                  data)

    # Write changes to file
    with open(f'Gascoigne/heat{process_id}.param', 'wb') as file:
        file.write(data)


def run_Gascoigne(process_id: int = 0):
    """
    Runs the compiled Gascoigne model 'HeatProblem' located in ./Gacoigne/build. The parameter file should be located
    in ./Gascoigne. The 'process_id' specifies the parameter file and resultsdir that should be availablle for this
    subprocess to access.

    Parameters
    ----------
    process_id : int, optional
    """
    subprocess.run(f'wsl (cd Gascoigne ; build/HeatProblem {process_id})', shell=True, stdout=subprocess.DEVNULL)


def generate_sample(process_id: int = 0) -> np.ndarray:
    """
    Generates a sample for the dataset.

    Random coefficients for the RHS are generated and passed to the parameter file. The Gascoigne script is run. The
    results in vtk format are read and transformed into a flattened array. The ordering is C-like: this is the same
    ordering numpy uses to flatten and reshape arrays. The vectors a, b and the flattened array are concatenated for
    easy storage.

    Parameters
    ----------
    process_id : int, optional
        Specifies the parameterfile and resultsdir that should be free for this process

    Returns
    -------
    res : np.array
        One dimensional array with the first Nx elements for the coefficients in a, the next Ny for the b coefficients
        and the rest for the resulting data
    """
    # Compute random vectors a and b of length Nx and Ny, respectively (without normalisation)
    a = np.random.normal(size=Nx)  # * (np.arange(Nx) + 1)
    b = np.random.normal(size=Ny)  # * (np.arange(Ny) + 1)

    # Set parameter file
    set_paramfile(a, b, process_id=process_id)

    # Run Gascoigne script
    run_Gascoigne(process_id=process_id)

    # Collect output
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(f"Gascoigne/Results{process_id}/solve.00004.vtk")
    reader.ReadAllScalarsOn()
    reader.Update()
    vtk_data = reader.GetOutput()

    # Get coordinates from vtk file
    # This gives a n*3 array, where the last variable is for time. Since this is steady-state, we can override this col
    data = VN.vtk_to_numpy(vtk_data.GetPoints().GetData())

    # Get solution at each point
    data[:, 2] = VN.vtk_to_numpy(vtk_data.GetPointData().GetScalars())

    # Reorder data
    data = data[data[:, 0].argsort()]
    data = data[data[:, 1].argsort(kind='mergesort')]

    # Concatenate a, b and the result
    return np.concatenate((a, b, data[:, 2]))


def generate_data(num_samples=1000, process_id=0) -> Callable[[], np.ndarray]:
    """
    Returns a function that runs a loop generating samples with generate_sample. This pattern is needed for asyncio's
    to_thread function, which requires a function

    Parameters
    ----------
    num_samples : int, optional
        Number of samples to generate
    process_id : int, optional
        Specifies the thread this runs on
    Returns
    -------
    func : Callable
    """

    def func():
        # Define result array
        result = np.ndarray((num_samples, Nx + Ny + image_dim ** 2))

        for i in range(num_samples):
            # Generate a sample and put it in a row of the result array
            result[i, :] = generate_sample(process_id=process_id)
        return result

    return func


async def main():
    """
    Generates samples on different threads.

    Returns
    -------
    result : list[np.ndarray]
        List of arrays with data sets

    """
    # Specify the number of samples to be generated and the number of threads to run on
    num_samples = 30000
    num_jobs = 6

    # Generate samples on different threads
    result = await asyncio.gather(
        *[asyncio.to_thread(generate_data(num_samples=int(num_samples / num_jobs), process_id=i)) for i in
          range(num_jobs)])

    # Concatenate the datasets from each thread and return
    return np.concatenate(result)


start = time.time()
dataset = asyncio.run(main())

# Save dataset to as a compressed .gz file
np.savetxt(f"dataset_{Nx}_{Ny}_{image_dim}x{image_dim}_{datetime.datetime.now().strftime('%y%m%d%H%M%S')}.gz", dataset)

print(time.time() - start)
print('done')
