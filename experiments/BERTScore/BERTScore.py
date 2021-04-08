import bert score
from bert_score import BERTScorer
import logging
import transformers
import matplotlib.pyplot as plt
from matplotlib import rcParams
from openpyxl import Workbook, load_workbook
from os import path
import os

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

rcParams["xtick.major.size"] = 0
rcParams["xtick.minor.size"] = 0
rcParams["ytick.major.size"] = 0
rcParams["ytick.minor.size"] = 0

rcParams["axes.labelsize"] = "large"
rcParams["axes.axisbelow"] = True
rcParams["axes.grid"] = True

print("Preparations complete...")


scorer = BERTScorer(lang="en", rescale_with_baseline=True)

file_to_score = "some_path"
xlsx = ".xlsx"
file_path = file_to_score + xlsx

def score_results(file_path):

    results_sheet = "Sheet1"
    result_list = list()

    # If the file exists we will open and read from the file
    if path.exists(file_path):

        results_file = load_workbook(file_path)
        
        # If the sheet name that contains the raw results is in the list, we will access that sheet
        if raw_results_sheet in results_file.sheetnames:

            # Get desired worksheet
            raw_results_sheet = results_sheet

            row_index = 0
            # Create list containing each row from excel spreadsheet.
            for row in raw_results_sheet.values:
                # Collect column names from first row
                if row_index == 0:
                    temp_list = list()
                    for value in row:
                        temp_list.append(value)
                    row_index += 1
                    result_list.append(temp_list)
                # For each row, append value to list at appropriate key in results_dict
                elif row_index > 0:
                    temp_list = list()
                    for value in row:
                        temp_list.append(value)
                    result_list.append(temp_list)

        # end if raw_results_sheet in results_file.sheetnames
        
    # end if file exists
    
    for result in result_list:
        print(result)
    
    # Create index of column names
    column_name_index = dict()
    column_index = 0
    for value in result_list[0]
        
        column_name_index[value] = column_index
        name_index += 1
        
    # end for value in result_list[0]
    
    # Split results into candidate (generated answers) and references (context from which answers were 
    # generated).
    candidates = list()
    references = list()
    
    
    # We skip the first line because it contains column headers
    for i in range (1, len(result_list)):
        
        candidates.append(result_list[i][column_name_index["G_answer"]])
        references.append(result_list[i][column_name_index["G_answer"]])
    
    # end for i in range (1, len(result_list))
    
    BERTScore_dict = dict()
    
    P, R, F1 = scorer.score(candidates, references)
    
    # For each result put an entry in the dictionary with the following.
    for i in range(0, len(result_list))
    
        
        BERTScore_dict[i] = {"context": references[i], "generated_answer": candidates[i], "bert_p":P[i], "bert_r":R[i], "bert_F1":F1[i] }
        
    # end for i in range(0, len(result_list))
    
    for entry in BERTScore_dict.keys():
        print(BERTScore_dict[entry])
    
    # Write all results to .csv file
    BERTScored_file = "BERTScored_" + file_to_score + ".csv"
    with open(BERTScored_file, "w+") as scored_file:
        
        # Write column headers to file
        headers = "context,generated_answer,P,R,F1\n"
        scored_file.write(headers)
        
        for entry in BERTScore_dict.keys():
            
            temp_line = ''
            temp_line += BERTScore_dict[entry]["context"]
            temp_line += ','
            temp_line += BERTScore_dict[entry]["generated_answer"]
            temp_line += ','
            temp_line += BERTScore_dict[entry]["bert_p"]
            temp_line += ','
            temp_line += BERTScore_dict[entry]["bert_r"]
            temp_line += ','
            temp_line += BERTScore_dict[entry]["bert_F1"]
            temp_line += '\n'
            
            print(temp_line)
            
            scored_file.write(temp_line)
            
    scored_file.close()
    
    print("We are done here...")

# end def score_results