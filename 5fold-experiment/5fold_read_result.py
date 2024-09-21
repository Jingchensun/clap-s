import os
import re
import json

# Folder path
folder_path = '/home/onsi/jsun/clap/log'

# Regular expressions to parse the content of log files
pattern_fold = re.compile(r'Fold (\d+)')
pattern_best_acc = re.compile(r'Best model saved with accuracy: (\d\.\d+)')

# Get all files in the folder
files = os.listdir(folder_path)

# Iterate over each file
for file_name in files:
    if file_name.endswith('_5fold.txt'):
        # Define a dictionary to store the maximum best_acc for each fold
        results = {}
        
        # File path
        log_file_path = os.path.join(folder_path, file_name)
        
        # Current fold number
        current_fold = None
        
        # Read the log file
        with open(log_file_path, 'r') as file:
            for line in file:
                fold_match = pattern_fold.search(line)
                best_acc_match = pattern_best_acc.search(line)
                
                if fold_match:
                    current_fold = int(fold_match.group(1))
                
                if best_acc_match and current_fold is not None:
                    best_acc = float(best_acc_match.group(1))
                    
                    if current_fold not in results or results[current_fold] < best_acc:
                        results[current_fold] = best_acc
        
        # Calculate the average of the maximum best_acc for each fold
        average_results = {
            'max_best_acc_per_fold': results,
            'average_best_acc': sum(results.values()) / len(results) if results else 0
        }

        # Generate the output file name
        output_file_name = file_name.replace('_5fold.txt', '.json')
        output_file_path = os.path.join(folder_path, output_file_name)
        
        # Save the results to a JSON file
        with open(output_file_path, 'w') as f:
            json.dump(average_results, f, indent=4)
        
        print(f"Results for {file_name} saved to {output_file_name}")
