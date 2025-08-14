import pandas as pd
from .utils import preprocess_data, get_matches, get_codes, merge_frame

def add_substance(
    col_with_substances: pd.Series,
    col_with_ref_substances: pd.Series,
    threshold: float = 0.85,
    max_per_match_id: int = 2,
    only_first_match: bool = False,
) -> pd.DataFrame:
    """
    This is the pipeline for creating the service variable
    for substances using ZfKD data.
    The functions are described in detail in utils.py.
    In short, the functions takes a pandasDataFrame column
    as an input and preprocesses its entries first.
    This results in a pandasDataFrame with the original
    input in one column and the preprocessed text in another one.
    The fuzzy matching relies on FuzzyMatcher from spaczz.
    It uses the preprocessed input and a reference list that
    the uses needs to provide. The reference list must be 
    a pandasDataFrame column (pd.Series) with substance names.
    The output is a pandasDataFrame with the original input,
    the preprocessed text and all possible matches with similary score.
    Use parameters to control output and sensitivity of the matcher. 
    
    arguments:
        col_with_substances: column with substances to be recoded
        col_with_ref_substances: column with reference substances
        threshold: similarity threshold, default 0.85
        max_per_match_id: maximum number of matches per ID, default 2
        only_first_match: return only the first match per ID
    """
    preprocessed_out = preprocess_data(col_with_substances)

    final_output = get_matches(
        preprocessed_out,
        col_with_ref_substances,
        threshold=threshold,
        max_per_match_id=max_per_match_id,
        only_first_match=only_first_match,
    )

    return final_output

def add_protocol(col_with_protocols: pd.Series,
                                col_with_ref_codes: pd.Series,
                                col_with_substances_for_protocols: pd.Series,
                                required_columns: list,
                                reference_list_protocol: pd.DataFrame,
                                threshold: int = 0.9):
    """
    Applies the protocol-relevant functions to make it
    more user-friendly.
    """    
    df_with_protocols = get_codes(col_with_protocols,
                                col_with_ref_codes,
                                col_with_substances_for_protocols,
                                required_columns,
                                threshold=threshold)
    
    out = merge_frame(df_with_protocols,
                    reference_list_protocol,
                    required_columns)
    
    return out