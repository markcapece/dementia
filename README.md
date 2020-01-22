[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/markcapece/dementia/master)
## Predicting Dementia in Elderly Patients Using Machine Learning
<p style='text-align: justify'>This repository is a report based on data from <a href="https://www.kaggle.com/jboysen/mri-and-alzheimers">Kaggle</a> regarding cross-sectional and longitudinal MRI studies of patients with and without <a href="https://wikipedia.org/wiki/dementia">dementia</a>. This data is available free to the public by <a href="https://oasis-brains.org">OASIS</a>. To read the report, download the repository and run Dementia.ipynb in a <a href="https://jupyter.org">Jupyter</a> notebook. Readers are encouraged to further explore the data using this notebook.</p>

### Contents
* `Dementia.ipynb` is a technical report written in the style of a research article. Code cells within
the notebook depend on the included `dementia/` module, CSV files within `data/`, and the folder hierarchy
 of `PKLs/`.
* `Data/` contains two CSV files of the OASIS datasets.
* `dementia/` is a local python module containing the following files.
    *  `analysis.py` contains the Analysis object for statistical and machine learning steps of the report.
    * `param_map.py` contains the PARAM_MAP dictionary for default machine learning parameter exploration.
    This dictionary can be modified, replaced, or removed by the user.
    * `utilities.py` contains the transform_data() and principal_component_transformation() functions for data
    processing steps of the report
* `PKLs/` contains folders `old/` and `latest/` for storing saved machine learning result tables
* `WIP_notebooks/` contains an unsupported previous version of the report with additional work-in-progress 
analysis

### A Note on PKL Files
Complete feature exploration by machine learning using the Analysis.ml_table() method can examine hundreds of 
combinations of features across several algorithms. The parameters used in this study required approximately 10 hours 
of runtime to execute this method. So that the reader can access these results without devoting significant time and 
resources to replicate the output, the results tables were saved as PKL files stored in `PKLs/latest/`. If files named 
`*_pc.pkl`, `*_pc_mean.pkl`, `*_raw.pkl`, and `*_raw_mean.pkl` exist in `PKLs/latest/`, these files will be loaded 
without needing to rerun Analysis.ml_table(). To force Analysis.ml_table() to execute, such as to test different 
parameters, simply move `*_pc.pkl`, `*_pc_mean.pkl`, `*_raw.pkl`, and `*_raw_mean.pkl` to `PKLs/old/`. The notebook will 
automatically save your new results of Analysis.ml_table() to `PKLs/latest/` as file names `{date}_{pc or raw}.pkl` and 
`{date}_{pc or raw}_mean.pkl`.
