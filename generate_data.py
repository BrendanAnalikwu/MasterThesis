import subprocess
import re
import warnings
import numpy as np
import vtk
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
import time
import datetime
import asyncio
import os
import shutil

# Set parameters
niter = 4
Nx = 8
Ny = 8
image_dim = 2 ** (niter + 1) + 1
if Nx / 2 > 2 ** niter or Ny / 2 > 2 ** niter:
    warnings.warn("Either Nx or Ny is too large for the grid size")


def set_paramfile(a: np.ndarray, b: np.ndarray, process_id=0):
    assert (a.ndim == 1 and b.ndim == 1)
    a_str = str(a).strip("[] ").replace("\n", "")
    b_str = str(b).strip("[] ").replace("\n", "")

    # If a parameter file for this thread does not exist, make a new copy
    if ~os.path.exists(f'Gascoigne/heat{process_id}.param'):
        shutil.copy2('Gascoigne/heat.param', f'Gascoigne/heat{process_id}.param')

    with open(f'Gascoigne/heat{process_id}.param', 'rb') as file:
        data = file.read()
    data = re.sub(b'\nniter[ \t]*\d', bytes(f'\nniter       {niter}', 'UTF-8)'), data)
    data = re.sub(b'\nresultsdir[ \t]*.*', bytes(f'\nresultsdir       Results{process_id}', 'UTF-8)'), data)
    data = re.sub(b'//Block RHS\n.*\n.*',
                  bytes(f'//Block RHS\na   {len(a)}  {a_str} \nb   {len(b)}  {b_str}', 'UTF-8)'),
                  data)
    with open(f'Gascoigne/heat{process_id}.param', 'wb') as file:
        file.write(data)


def run_Gascoigne(process_id=0):
    subprocess.run(f'wsl (cd Gascoigne ; build/HeatProblem {process_id})', shell=True, stdout=subprocess.DEVNULL)


def generate_sample(process_id=0):
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
    data = VN.vtk_to_numpy(vtk_data.GetPoints().GetData())
    data[:, 2] = VN.vtk_to_numpy(vtk_data.GetPointData().GetScalars())
    data = data[data[:, 0].argsort()]
    data = data[data[:, 1].argsort(kind='mergesort')]

    return np.concatenate((a, b, data[:, 2]))


def generate_data(num_samples=1000, process_id=0):
    def func():
        result = np.ndarray((num_samples, Nx + Ny + image_dim ** 2))

        for i in range(num_samples):
            result[i, :] = generate_sample(process_id=process_id)
        return result

    return func


async def main():
    num_samples = 100
    num_jobs = 7
    result = await asyncio.gather(
        *[asyncio.to_thread(generate_data(num_samples=int(num_samples / num_jobs), process_id=i)) for i in
          range(num_jobs)])
    return result


start = time.time()
res = asyncio.run(main())
dataset = np.concatenate(res)
np.savetxt(f"dataset_{Nx}_{Ny}_{image_dim}x{image_dim}_{datetime.datetime.now().strftime('%y%m%d%H%M%S')}.gz", dataset)
print(time.time() - start)
print('done')
