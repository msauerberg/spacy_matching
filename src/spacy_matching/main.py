"""
Uses the test dataset provided by Stefan Meisegeier with fake clin data
Functions are imported from utils and data is read in with pandas
"""

import duckdb
import pandas as pd
from utils import create_substance_service_var, create_protocol_service_var, sort_row

# get data from here:
# https://gitlab.opencode.de/robert-koch-institut/zentrum-fuer-krebsregisterdaten/cancerdata-generator/-/tree/main/assets?ref_type=heads
db_con = duckdb.connect(".local/fake_clin_data.db", read_only=True)
substance_data = db_con.sql("SELECT distinct Bezeichnung FROM Substanz").to_df()
protocol_data = db_con.sql("SELECT distinct Bezeichnung FROM Protokoll").to_df()

db_con.close()

col_with_substances_ZfKD_fake_data = substance_data["Bezeichnung"]


URL_LINK_substance = "https://gitlab.opencode.de/robert-koch-institut/zentrum-fuer-krebsregisterdaten/cancerdata-references/-/raw/main/data/v2/Klassifikationen/substanz.csv?ref_type=heads"
reference_list_substance = pd.read_csv(URL_LINK_substance, sep=";")
col_with_ref_substances_ZfKD = reference_list_substance["substanz"]

URL_LINK_protocol = "https://gitlab.opencode.de/robert-koch-institut/zentrum-fuer-krebsregisterdaten/cancerdata-references/-/raw/main/data/v2/Klassifikationen/protokoll.csv"
reference_list_protocol = pd.read_csv(URL_LINK_protocol, sep=";")

# get the columns with substance names in it, here
# they are called substanz_1 to  substanz_7
required_columns = [f'substanz_{i}' for i in range(1, 8)]

#combine all substance names from these seven columns into one
# vector with unique substance names which can be used as
# a reference list for extracting substance names from free text fields
substanz_cols = [col for col in reference_list_protocol.columns if col.startswith('substanz_')]
all_subs = pd.Series(pd.unique(reference_list_protocol[substanz_cols].values.ravel()))

# protocol codes from the reference list 
codes = reference_list_protocol["code"]

# get the service variable for protocols
protocol_service = create_protocol_service_var(protocol_data["Bezeichnung"],
                                               codes,
                                               all_subs,
                                               required_columns,
                                               reference_list_protocol)
# save it
protocol_service.to_csv(".local/results_protocol.csv", index=False)


#get the service variable for substances in two versions
results_atomic = create_substance_service_var(
    col_with_substances=col_with_substances_ZfKD_fake_data,
    col_with_ref_substances=col_with_ref_substances_ZfKD,
    only_first_match=True,
)

results_multiple_hits = create_substance_service_var(
    col_with_substances=col_with_substances_ZfKD_fake_data,
    col_with_ref_substances=col_with_ref_substances_ZfKD,
    only_first_match=False,
)

# save both versions
results_atomic.to_csv(".local/results_atomic.csv", index=False)
results_multiple_hits.to_csv(".local/results_multiple_hits.csv", index=False)
