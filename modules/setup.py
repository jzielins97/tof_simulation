from setuptools import setup

setup(name='tof_simulation_modules',
      version='0.2.0',
      author='Jakub Zielinski',
      py_modules=["constants","FragmentsDataLoader","electrode","trap","geometry","field_solver","plasma","tracker","tof_tracking"],
      package_data={'':["trap_setup.json","ElectrodesPotentialMap.txt"]})
