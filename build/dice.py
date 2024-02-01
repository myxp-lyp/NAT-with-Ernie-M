import json

def read_alignment_file(file_path):
    alignments = []
    with open(file_path, 'r') as f:
        for line in f:
            alignment_pairs = line.strip().split()
            alignments.append([(int(pair.split('-')[0]), int(pair.split('-')[1])) for pair in alignment_pairs])
    return alignments

def create_word_mapping(alignments, source_tokens, target_tokens):
    word_mapping = {}
    for alignment, source_token, target_token in zip(alignments, source_tokens, target_tokens):
        for src_index, tgt_index in alignment:
            if src_index < len(source_token) and tgt_index < len(target_token):
                word_mapping[source_token[src_index]] = target_token[tgt_index]
    return word_mapping

def save_mapping_to_file(word_mapping, output_file):
    with open(output_file, 'w') as f:
        json.dump(word_mapping, f, indent=2)

def main():
    alignment_file_path = 'alignment_output.txt'
    source_file_path = 'train.ro'
    target_file_path = 'train.en'
    output_mapping_file = 'word_mapping.json'

    # Read the alignment file
    alignments = read_alignment_file(alignment_file_path)

    # Read source and target tokenized files
    with open(source_file_path, 'r') as f:
        source_tokens = [line.strip().split() for line in f]

    with open(target_file_path, 'r') as f:
        target_tokens = [line.strip().split() for line in f]

    # Create word mapping dictionary
    word_mapping = create_word_mapping(alignments, source_tokens, target_tokens)

    # Save the word mapping dictionary to a file
    save_mapping_to_file(word_mapping, output_mapping_file)

if __name__ == "__main__":
    main()
