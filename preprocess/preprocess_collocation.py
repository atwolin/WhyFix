import pandas as pd

from process.utils.helper_functions import (
    SupportMaterial,
)

SM = SupportMaterial('_', '_')


def preprocess_collocation_data(input_txt_filepath):
    """
    Preprocesses the raw collocation data into various formats.

    Args:
        input_txt_filepath (str): Path to the raw collocation TXT file.
    """
    data = []
    with open(input_txt_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|||')
            # Clean up leading/trailing spaces for each part
            cleaned_parts = [part.strip() for part in parts]
            data.append(cleaned_parts)

    # Define column names
    # column_names = [
    #     "error_word", "correct_word", "error_collocation", "correct_collocation",
    #     "error_correct_collocation_errors", "error_correct_total_errors",
    #     "collocation_word", "collocation_category", "category_accumulated_errors",
    #     "category_unique_error_count", "total_error_count"
    # ]
    column_names = ['error_component', 'error_component_pos', 'error_collocation',
                    'correct_collocation', 'collocation_correction_freq', 'component_change_freq_details',
                    'collocation_pivot_and_category_details', 'component_change_pivot_category_accum_freq_details',
                    'component_change_pivot_category_uniq_freq_details', 'error_component_total_freq_details',
                    'error_component_total_freq_in_pivot_category_details']

    # Create a DataFrame for full data
    df_full = pd.DataFrame(data, columns=column_names[:len(data[0])])  # Adjust columns based on actual data length

    # Save full data to TXT (comma-separated)
    df_full.to_csv(SM.filePath_collocation_full_txt, index=False, header=False, sep='|')
    print(f"Full preprocessed data saved to: {SM.filePath_collocation_full_txt}")

    # Save full data to CSV
    df_full.to_csv(SM.filePath_collocation_full_csv, index=False)
    print(f"Full preprocessed data saved to: {SM.filePath_collocation_full_csv}")

    # Create a DataFrame for simplified data (first 4 columns)
    df_simplified = df_full.iloc[:, :4]
    df_simplified = df_simplified.drop_duplicates()

    # Save simplified data to TXT (comma-separated)
    df_simplified.to_csv(SM.filePath_collocation_simplified_txt, index=False, header=False, sep='|')
    print(f"Simplified preprocessed data saved to: {SM.filePath_collocation_simplified_txt}")

    # Save simplified data to CSV
    df_simplified.to_csv(SM.filePath_collocation_simplified_csv, index=False)
    print(f"Simplified preprocessed data saved to: {SM.filePath_collocation_simplified_csv}")


if __name__ == "__main__":
    preprocess_collocation_data(SM.filePath_collocation_txt_raw)
