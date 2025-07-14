"""
Substance matching relies on
data wrangling using pandas and
fuzzy string machting with spacy and spaczz
"""
import re
import pandas as pd
import spacy
from spaczz.matcher import FuzzyMatcher
import numpy as np
from rapidfuzz import process

# helper functions


def prepare_free_text(input_col: pd.Series) -> pd.DataFrame:
    """Prepares data by renaming, stripping, and cleaning null or empty entries."""
    input_data = pd.DataFrame(
        {
            "ID": range(1, len(input_col) + 1),
            "Original": input_col.fillna("NA").replace("", "NA"),
        }
    )
    input_data["Original"] = input_data["Original"].astype(str).str.strip()
    return input_data


def remove_short_words(text: str) -> str:
    """removes words that are shorter than 3 characters
    """    
    return " ".join([word for word in text.split() if len(word) >= 3])


def find_5FU(text: str) -> str:
    """5FU and all the variants are abbrevations for
    Flourouracil. The function translate it to the actual substance name
    and catches common misspellings. In principle, fuzzy matching should be able
    to deal with misspellings but since this is a very common substance it might make sense
    to explicitly take care of them
    """    
    pattern = (
        r"5 fu|5fu|5-fu|5_fu|Fluoruracil|flourouracil|5-fluoruuracil|"
        r"5-fluoro-uracil|5-fluoruuracil|5-fluoruracil|floururacil|"
        r"5-fluorounacil|flourouraci|5-fluourouracil"
    )
    return re.sub(pattern, "fluorouracil", text, flags=re.IGNORECASE)


def calciumfolinat_to_folin(text: str) -> str:
    """This is again a common substance and depending on the threshold parameter
    for fuzzy matching it might be overlooked by fuzzy matching. This is why 
    this function translates it.
    """    
    return re.sub(r"\b(Calciumfolinat)\b", "folinsäure", text, flags=re.IGNORECASE)


def find_gemcitabin(text: str) -> str:
    """Another common substance that should be found, independend of the
    threshold parameter of the fuzzy matcher.
    """    
    return re.sub(
        r"Gemcibatin(?:e)?(?: Mono)?", "gemcitabin", text, flags=re.IGNORECASE
    )


def find_Paclitaxel_nab(text: str) -> str:
    """Sometimes the algorithm does not find nab-Paclitaxel since
    it puts "nab" in the fron. Here, we translate it to the correct
    substance name "Paclitaxel nab"
    """    
    return re.sub(
        r"\bnab[\s\-]?Paclitaxel\b", "Paclitaxel nab", text, flags=re.IGNORECASE
    )


# preprocessing using helper functions


def preprocess_data(col_with_free_text: pd.Series) -> pd.DataFrame:
    """Applies functions from above sequentially to input data
    """    
    df = prepare_free_text(col_with_free_text)
    processed = (
        df["Original"]
        .apply(find_5FU)
        .apply(find_gemcitabin)
        .apply(find_Paclitaxel_nab)
        .apply(calciumfolinat_to_folin)
        .apply(remove_short_words)
        .str.strip()
    )
    df["Preprocessed_text"] = processed
    return df


# find matches with FuzzyMatcher from spaczz


def get_matches(
    preprocessed_data: pd.DataFrame,
    ref_substance: pd.Series,
    threshold: float = 0.85,
    max_per_match_id: int = 2,
    only_first_match: bool = False,
) -> pd.DataFrame:
    """
    Extracts substances from the input text.
    Each row is taken as one input text. In principle,
    there should be only one substance per row. However, 
    often there are more than one (e.g., "sub1; sub2 and sub3").
    The FuzzyMatcher is capable of finding more than one match per row.
    It stores matches in additional columns. Use "only_first_match" to 
    remove extra columns and return only the first matched substance per row.
    The max_per_match_id parameter defines how many matches are returned per word.
    For instance, a row might include "Interferon alpha" and the algorithm might
    find "Interferon alpha 2a", "Interferon alpha 2b" and "Peginterferon alpha"
    as potential matches. If the parameter is set to 2, it would only return 2 matches-
    the two best matches measured by the similarity score. Please note that the
    parameter controlls the number of matches per word. For instance, if
    the input is "sub1; sub2 and sub3" It can return two (if set to two) per ID,
    meaning the output can include up to 6 potential matches for this input.
    It is recommanded to use a rather high threshold parameter because substance
    names are often very similar to each other.
    """
    nlp = spacy.blank("en")
    matcher = FuzzyMatcher(nlp.vocab)

    for sub in ref_substance.dropna().astype(str):
        matcher.add(sub, [nlp(sub)])

    results = []

    for _, row in preprocessed_data.iterrows():
        text = row["Preprocessed_text"] #uses preprocessed text for FuzzyMatcher
        original = row["Original"]
        doc = nlp(text)
        matches = matcher(doc)

        matches_filtered = [m for m in matches if m[3] >= threshold * 100]
        matches_sorted = sorted(matches_filtered, key=lambda x: x[3], reverse=True)

        result_row = {"Original": original}
        result_row["Preprocessed"] = text
        match_id_counts = {}
        match_idx = 1

        for match_id, start, end, ratio, _ in matches_sorted:
            count = match_id_counts.get(match_id, 0)
            if count >= max_per_match_id:
                continue

            result_row[f"Hit{match_idx}"] = doc[start:end].text
            result_row[f"Mapped_to{match_idx}"] = match_id
            result_row[f"Similarity{match_idx}"] = ratio

            match_id_counts[match_id] = count + 1
            match_idx += 1

        results.append(result_row)

    out = pd.DataFrame(results)

    if only_first_match:
        cols_to_keep = ["Original", "Preprocessed", "Hit1", "Mapped_to1", "Similarity1"]
        available_columns = [col for col in cols_to_keep if col in out.columns]
        dta_col_selected = out[available_columns]
        dta_col_selected.columns = [
            re.sub(r"\d+$", "", col) for col in dta_col_selected.columns
        ]
        return dta_col_selected

    return out


def create_substance_service_var(
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


def get_matches_protocol(
    preprocessed_data: pd.DataFrame,
    ref_substance: pd.Series,
    threshold: float = 0.90
) -> pd.DataFrame:
    """
    The function is based on the previous function 'get_matches'.
    The aim is finding substances from the input column which
    may describe a protocol. Hence, it uses all the substances that are
    included in the protocol reference list. There should be only one
    match per substance. For this reason, the function does not need
    paramters 'max_per_match_id' or 'only_first_match'.
    """
    nlp = spacy.blank("en")
    matcher = FuzzyMatcher(nlp.vocab)

    for sub in ref_substance.dropna().astype(str):
        matcher.add(sub, [nlp(sub)])

    results = []

    for _, row in preprocessed_data.iterrows():
        text = row["Preprocessed_text"] 
        original = row["Original"]
        extracted_codes = row["Extracted_Codes"]
        similarity_score = row["Similarity_Score"]
        doc = nlp(text)
        matches = matcher(doc)

        matches_filtered = [m for m in matches if m[3] >= threshold]
        matches_sorted = sorted(matches_filtered, key=lambda x: x[3], reverse=True)

        result_row = {"Original": original}
        result_row["Preprocessed"] = text
        result_row["Extracted_Codes"] = extracted_codes
        result_row["Similarity_Score"] = similarity_score
        match_id_counts = {}
        match_idx = 1

        for match_id, start, end, ratio, _ in matches_sorted:
            count = match_id_counts.get(match_id, 0)
            if count >= 1:
                continue

            result_row[f"substanz_{match_idx}"] = match_id
            
            match_id_counts[match_id] = count + 1
            match_idx += 1

        results.append(result_row)

    out = pd.DataFrame(results)


    return out


def sort_row(row, required_columns):
    """
    The function will order columns alphabetically.
    This is important because we are looking for specific protocols,
    which can be comprised of several substances. For example, the combination
    "Cisplatin" and "Gemcitabin" is called "Gem-Cis". Ordering both, the extraced
    substances as well as the reference list alphabetically makes detecting the
    combinations much easier. 
    """    
    values = row[required_columns].dropna().astype(str).tolist()
    values.sort()
    values += [np.nan] * (len(required_columns) - len(values))
    return pd.Series(values, index=required_columns)

def fuzzy_match(text, ref_codes):
    """
    Uses process from rapidfuzz to find protocol codes.

    text: The input text from the free text field 'Protocol'
    ref_codes: The codes for protocols provided by the reference list
    threshold: parameter for accuaracy of matches
    """
    match = process.extractOne(text, ref_codes, score_cutoff=90)
    if match:
        return match[0], match[1]
    else:
        return np.nan, np.nan
    
def get_codes(col_with_protocols: pd.Series,
              col_with_ref: pd.Series,
              all_subs: pd.Series,
              required_columns: list,
              threshold: int = 0.9):
    """
    Function extracts first, the protocol codes from the free text field
    'col_with_protocol' using the protocol reference list 'col_with_ref'.
    Then, it extracts the subtance names using the
    substances from the protocol reference list 'all_subs'.
    The input 'required_column' are the substance columns. In this case,
    'substance_1', substance_2, substance_3 ... substance_7
    """    
    protocol_df = col_with_protocols.to_frame(name = "Original")
    protocol_df[["Extracted_Codes", "Similarity_Score"]] = protocol_df["Original"].apply(
    lambda x: pd.Series(fuzzy_match(x, col_with_ref))
    )  
    protocol_df["Preprocessed_text"] = preprocess_data(protocol_df["Original"])["Preprocessed_text"]
    out = get_matches_protocol(protocol_df, all_subs, threshold * 100)

    # Add missing columns with NaN
    for col in required_columns:
        if col not in out.columns:
            out[col] = np.nan
    
    # Apply only to substanz columns
    out[required_columns] = out.apply(lambda row: sort_row(row, required_columns),
                                      axis=1)
    
    return out

def merge_frame(df_data: pd.DataFrame,
                df_references:pd.DataFrame,
                required_columns: list):
    """
    After substances are extracted and ordered alphabetically,
    we left join the dataframe 'df_data' with the protocol reference list
    'df_references'. This gives us the corresponding protocol codes.
    For example, if we could extract "Gemcitabin" and "Cisplatin",
    the function orders it to "Cisplatin" and "Gemcitabin" and
    the left join with the protocol reference list adds the
    column code with the corresponding protocol code "Gem-Cis".
    """    
    merge_columns = required_columns
    df_references[required_columns] = df_references.apply(
    lambda row: sort_row(row, required_columns),
                                      axis=1)

    try:
        merged_df = pd.merge(
            df_data,
            df_references.drop("therapieart", axis=1, errors='ignore'),
            on=merge_columns,
            how='left'
        )
    except Exception as e:
        print(f"Defined NaN as a string to avoid error: {e}")
        for df in [df_data, df_references]:
            for col in merge_columns:
                df[col] = df[col].astype(str).replace('nan', 'NaN').fillna('NaN')
        
        merged_df = pd.merge(
            df_data,
            df_references.drop("therapieart", axis=1, errors='ignore'),
            on=merge_columns,
            how='left'
        )
    
    mask_all_nan_or_empty = merged_df[merge_columns].map(lambda x: pd.isna(x) or x == 'NaN').all(axis=1)
    merged_df.loc[mask_all_nan_or_empty, "code"] = np.nan

    return merged_df


def create_protocol_service_var(col_with_protocols: pd.Series,
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

def test():
    print("this is just a test")
    return