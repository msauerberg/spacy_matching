# SubstanceFinder

![py3.10](https://img.shields.io/badge/python-3.10_|_3.11_|_3.12-blue.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHlZZWxsb3ciIGdyYWRpZW50VHJhbnNmb3JtPSJyb3RhdGUoNDUpIj4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iI2ZlNSIgb2Zmc2V0PSIwLjYiLz4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iI2RhMSIgb2Zmc2V0PSIxIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJweUJsdWUiIGdyYWRpZW50VHJhbnNmb3JtPSJyb3RhdGUoNDUpIj4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iIzY5ZiIgb2Zmc2V0PSIwLjQiLz4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iIzQ2OCIgb2Zmc2V0PSIxIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogIDwvZGVmcz4KCiAgPHBhdGggZD0iTTI3LDE2YzAtNyw5LTEzLDI0LTEzYzE1LDAsMjMsNiwyMywxM2wwLDIyYzAsNy01LDEyLTExLDEybC0yNCwwYy04LDAtMTQsNi0xNCwxNWwwLDEwbC05LDBjLTgsMC0xMy05LTEzLTI0YzAtMTQsNS0yMywxMy0yM2wzNSwwbDAtM2wtMjQsMGwwLTlsMCwweiBNODgsNTB2MSIgZmlsbD0idXJsKCNweUJsdWUpIi8+CiAgPHBhdGggZD0iTTc0LDg3YzAsNy04LDEzLTIzLDEzYy0xNSwwLTI0LTYtMjQtMTNsMC0yMmMwLTcsNi0xMiwxMi0xMmwyNCwwYzgsMCwxNC03LDE0LTE1bDAtMTBsOSwwYzcsMCwxMyw5LDEzLDIzYzAsMTUtNiwyNC0xMywyNGwtMzUsMGwwLDNsMjMsMGwwLDlsMCwweiBNMTQwLDUwdjEiIGZpbGw9InVybCgjcHlZZWxsb3cpIi8+CgogIDxjaXJjbGUgcj0iNCIgY3g9IjY0IiBjeT0iODgiIGZpbGw9IiNGRkYiLz4KICA8Y2lyY2xlIHI9IjQiIGN4PSIzNyIgY3k9IjE1IiBmaWxsPSIjRkZGIi8+Cjwvc3ZnPgo=)

## Background

In Germany's cancer registries, substances are reported in a free text field, e.g., "Interferon alpha-2a weekly i.v.".
The reported text might include additional information such as dosage or labels or typos.
For analysis, it is helpful to have substances reported in a harmonized way. The aim of this Python function is extracting the substance name from the free text field. The script uses the FuzzyMatcher from spaczz and spacy to scan each free text for potential matches.

## usage

Call  `create_substance_service_var()` to get a pandas dataframe including the recoded column / variable.

The data should be provided as a column of a pandasDataFrame (i.e., a PandaSeries) with free text records in each row.
The substances that we want to find in the free text field should provided as a PandaSeries as well.

code example:

```python
import re
import pandas as pd
import spacy
from spaczz.matcher import FuzzyMatcher
from utils import preprocess_data, get_matches

#get reference table from web
URL_V2 = "https://gitlab.opencode.de/robert-koch-institut/zentrum-fuer-krebsregisterdaten/cancerdata-references/-/raw/main/data/v2/Klassifikationen/substanz.csv?ref_type=heads"
reference_list = pd.read_csv(URL_V2, sep=";")

#get the column with reference substances
col_with_ref_substances_ZfKD = reference_list["substanz"]

#create a pandaSeries with some test data
col_with_made_up_data = pd.Series(["Interferon alpha 2a", "Paclitaxel (nab)", "Filgrastim", "Leuprorelin; Tamoxifen"])

#get only one match per free text field
results_atomic = create_substance_service_var(
    col_with_substances=col_with_made_up_data,
    col_with_ref_substances=col_with_ref_substances_ZfKD,
    only_first_match=True,
    threshold=0.85
)

#allow more hits
results_multiple_hits = create_substance_service_var(
    col_with_substances=col_with_made_up_data,
    col_with_ref_substances=col_with_ref_substances_ZfKD,
    only_first_match=False,
    threshold=0.85,
    max_per_match_id=2
)
```

## Options and parameters

The function features some parameters: The threshold parameter defines the accuracy. The lower the more matches but higher
values lead to more accurate matches. The graph below plots the number of substances extracted from free text fields against the corresponding threshold parameter. There is a tradeoff between the number of matches and their accuracy. A threshold value of 0.85 is set as default as it usually ensures sufficient accuracy.

![show_num](https://github.com/msauerberg/spacy_matching/blob/master/images/plot_match_count_vs_threshold.png?raw=true)

The option "only_first_match = True" should be used if the user wants to allow only one match per free text field.
Even if there are several substances in the free text field such as "Leuprorelin; Tamoxifen", the function will return only the first match.
If the option is set to "False", the function can return multiple hits. For the input "Interferon alpha-2a weekly i.v.", the function might return two substances "Interferon alpha-2a" and "Interferon alpha-2b" (assuming they are both on the reference list). Results based on both options is shown below.

![show_num](https://github.com/msauerberg/spacy_matching/blob/master/images/atomic_vs_multiple.png?raw=true)

## Credits

Thanks to Stefan Meisegeier for helpful feedback on the code and for making the function available as a python package.

