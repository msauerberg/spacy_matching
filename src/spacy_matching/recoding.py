import pandas as pd
from .utils import preprocess_data, get_matches, fuzzy_match, add_spaces, get_zfkd_ref_substance

def add_substance(
    col_with_substances: pd.Series,
    col_with_ref_substances: pd.Series | None = None,
    threshold: float = 0.85,
    max_per_match_id: int = 2,
    only_first_match: bool = True,
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
        col_with_substances: column with substances to be recoded. if empty, zfkd reference will be auto-added
        col_with_ref_substances: column with reference substances
        threshold: similarity threshold, default 0.85
        max_per_match_id: maximum number of matches per ID, default 2
        only_first_match: return only the first match per ID
    """
    
    # * check if col_with_substances is None
    if col_with_ref_substances is None:
        col_with_ref_substances = get_zfkd_ref_substance()
    
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
            col_with_ref: pd.Series,
            threshold: float = 0.9):
    """
    Returns DataFrame with extracted code and similarity (normalized 0..1).
    """
    protocol_df = col_with_protocols.to_frame(name="Original")
    protocol_df[["Extracted_Code", "Similarity"]] = protocol_df["Original"].apply(
        lambda x: pd.Series(fuzzy_match(x, col_with_ref, threshold=threshold))
    )

    protocol_df["Preprocessed"] = add_spaces(protocol_df["Original"])
    protocol_df["Similarity"] = protocol_df["Similarity"] / 100.0
    protocol_df = protocol_df[["Original", "Extracted_Code", "Similarity"]]
    protocol_df.rename(columns={
        "Extracted_Code": "Extracted_Protocol_Code",
        "Similarity": "SimilarityCode"
        }, inplace=True)
        
    return protocol_df

# def add_protocol(col_with_protocols: pd.Series,
#                                 col_with_ref_codes: pd.Series,
#                                 col_with_substances_for_protocols: pd.Series,
#                                 required_columns: list,
#                                 reference_list_protocol: pd.DataFrame,
#                                 threshold: int = 0.9):
    
#     Applies the protocol-relevant functions to make it
#     more user-friendly.
      
#     df_with_protocols = get_codes(col_with_protocols,
#                                 col_with_ref_codes,
#                                 col_with_substances_for_protocols,
#                                 required_columns,
#                                 threshold=threshold)
    
#     out = merge_frame(df_with_protocols,
#                     reference_list_protocol,
#                     required_columns)
    
#     return out
