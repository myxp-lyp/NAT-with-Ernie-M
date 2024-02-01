def combine_files(file1_path, file2_path, output_path):
    with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # Ensure both files have the same number of lines
    if len(lines1) != len(lines2):
        raise ValueError("Input files must have the same number of lines.")

    combined_lines = [f"{line1.strip()} ||| {line2.strip()}\n" for line1, line2 in zip(lines1, lines2)]

    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.writelines(combined_lines)

# Example usage
combine_files('train.ro', 'train.en', 'combined_output.txt')
