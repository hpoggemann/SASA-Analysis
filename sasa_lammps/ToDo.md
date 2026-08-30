# ToDo

- Generalize the postprocessing: More adaptable graphs, less hardcoded  labels etc... (this may not be my decision though)
  - Go over the necessary files and check if an invocation of `postprocessing` is possible without a previous `compute` (in case someone wants to do postprocessing without re-computing the whole thing). 
  - Throw a proper exception in case files are missing for the postprocessing
- Provide a proper warning message in case the `lammps_exe` variable is provided but points to a non-existent file
- Refactor all `str` that represent paths to be of type `pathlib.Path`. Convert the strings specified by the user directly in the respective method
- Adapt a unified coding style (especially in the namings)
- Go over all the tests
  - Refactor the fixtures and share paths, the resource-path in particular
  - Sort out unnecessary tests and maybe add new ones 