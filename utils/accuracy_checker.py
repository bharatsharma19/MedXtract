import pandas as pd
import io

def compare_csvs(actual_file, extracted_file):
    actual_df = pd.read_csv(io.StringIO(actual_file.read().decode()))
    extracted_df = pd.read_csv(io.StringIO(extracted_file.read().decode()))

    actual_row = actual_df.iloc[0]
    extracted_row = extracted_df.iloc[0]

    total = len(actual_row)
    correct = sum(
        1 for col in actual_row.index 
        if str(actual_row[col]).strip().lower() == str(extracted_row[col]).strip().lower()
    )
    accuracy = correct / total * 100

    mismatches = {
        col: {"actual": actual_row[col], "extracted": extracted_row[col]}
        for col in actual_row.index
        if str(actual_row[col]).strip().lower() != str(extracted_row[col]).strip().lower()
    }

    return {
        "accuracy": f"{accuracy:.2f}%",
        "total_fields": total,
        "correct_fields": correct,
        "mismatched_fields": mismatches
    }
