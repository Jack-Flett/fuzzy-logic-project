# Data

`raw/water-treatment.data` and `raw/water-treatment.names` are the unmodified files of the
**UCI Machine Learning Repository – Water Treatment Plant** dataset
(https://archive.ics.uci.edu/dataset/106/water+treatment+plant, DOI 10.24432/C5FS4C),
created by Manel Poch (Universitat Autònoma de Barcelona), donated by J. Béjar and U. Cortés, 1993.
Licence: Creative Commons Attribution 4.0 (CC BY 4.0).

* 527 daily records (1 Jan 1990 – 30 Oct 1991) of an urban wastewater treatment plant, 38 numeric attributes, missing values marked `?`.
* Target used in this project: `DBO-S` (effluent biological oxygen demand, mg/L).
* The private 1,564-sample dataset of the reference paper is not public; this public dataset is the one
  the paper itself uses for external validation (its Table 6).

All preprocessing is done in code (`src/data.py`) from the raw file; nothing in `raw/` is edited by hand.
