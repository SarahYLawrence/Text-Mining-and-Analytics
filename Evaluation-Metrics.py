import numpy as np

import math

Human_answers2022 = [
    [6, 5, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 1, 3, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2, 1],
    [1, 6, 0, 0, 2, 1, 3, 1, 0, 0, 0, 2, 2, 1, 1, 2, 3, 0, 2, 3, 4, 2],
    [2, 1, 4, 6, 1, 6, 4, 5, 4, 6, 6, 6, 4, 6, 2, 2, 6, 1, 3, 6, 5, 4],
    [6, 0, 3, 6, 5, 4, 3, 1, 3, 0, 1, 0, 5, 2, 1, 0, 1, 0, 4, 5, 2, 3],
    [6, 3, 0, 5, 2, 2, 2, 2, 1, 4, 5, 5, 2, 2, 1, 5, 4, 6, 6, 5, 5, 4],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 1, 0, 0, 1, 1, 1],
    [5, 2, 3, 6, 1, 1, 6, 0, 6, 6, 0, 3, 0, 0, 0, 5, 5, 0, 0, 2, 4, 0],
    [6, 6, 0, 6, 6, 6, 6, 0, 6, 6, 6, 6, 2, 2, 2, 6, 6, 4, 6, 6, 6, 6],
    [4, 3, 3, 6, 4, 6, 6, 6, 6, 6, 3, 4, 4, 5, 5, 6, 6, 6, 6, 5, 4, 5],
    [2, 1, 1, 1, 1, 2, 0, 2, 1, 1, 6, 2, 0, 1, 0, 3, 3, 1, 5, 4, 4, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [5, 2, 1, 2, 5, 6, 2, 1, 4, 6, 3, 3, 4, 6, 1, 5, 5, 1, 2, 3, 4, 5],
    [6, 2, 6, 6, 6, 6, 2, 2, 4, 6, 4, 6, 1, 3, 2, 6, 6, 5, 6, 6, 4, 6],
    [6, 1, 6, 3, 6, 6, 6, 5, 6, 6, 6, 6, 3, 4, 2, 5, 5, 6, 6, 5, 6, 5],
    [6, 5, 5, 6, 5, 6, 6, 6, 6, 6, 6, 6, 0, 6, 4, 6, 6, 5, 6, 6, 6, 6],
    [5, 0, 4, 6, 4, 6, 6, 6, 6, 6, 6, 6, 1, 6, 2, 6, 6, 3, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 2, 3, 2, 0],
    [1, 1, 0, 0, 3, 1, 0, 1, 0, 1, 2, 2, 1, 0, 0, 3, 3, 0, 4, 4, 1, 1],
    [6, 3, 6, 6, 6, 4, 6, 6, 6, 6, 6, 3, 1, 0, 0, 5, 2, 1, 2, 1, 4, 6],
    [6, 5, 4, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 1, 5, 5, 6, 6],
    [3, 1, 6, 2, 0, 3, 4, 4, 6, 6, 6, 6, 1, 2, 3, 5, 6, 4, 6, 6, 6, 2],
    [6, 0, 6, 6, 6, 6, 3, 0, 4, 1, 5, 6, 4, 3, 1, 3, 5, 2, 0, 6, 6, 5],
    [3, 3, 3, 0, 2, 6, 3, 3, 5, 5, 5, 4, 1, 1, 1, 5, 5, 6, 5, 5, 5, 6],
    [6, 5, 4, 6, 6, 6, 3, 3, 5, 6, 6, 6, 0, 6, 6, 6, 6, 0, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 1, 6, 6, 6, 2, 6, 5, 6, 6],
    [1, 0, 0, 0, 0, 3, 0, 0, 0, 1, 3, 1, 0, 0, 1, 3, 4, 0, 4, 4, 4, 4],
    [2, 0, 1, 1, 3, 5, 3, 1, 2, 6, 6, 3, 3, 5, 0, 4, 2, 6, 4, 1, 2, 3]
]

Human_answers2023 = [
    [2, 0, 0, 0, 0, 3, 1, 0, 6, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 1, 1, 1],
    [5, 1, 5, 3, 4, 6, 5, 5, 4, 6, 6, 5, 3, 5, 6, 6, 6, 6, 6, 6, 6, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0],
    [0, 6, 6, 0, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 1, 0, 2],
    [1, 0, 1, 0, 0, 2, 0, 0, 0, 0, 4, 2, 0, 1, 0, 1, 2, 4, 2, 2, 2, 2],
    [6, 5, 6, 6, 0, 0, 0, 0, 6, 6, 6, 6, 2, 1, 3, 1, 1, 0, 5, 5, 5, 4],
    [6, 5, 5, 5, 6, 6, 4, 4, 6, 6, 4, 4, 1, 1, 1, 6, 3, 5, 5, 3, 4, 4],
    [5, 4, 6, 4, 5, 6, 1, 1, 1, 4, 3, 6, 0, 3, 0, 3, 3, 3, 4, 3, 2, 3],
    [3, 2, 3, 0, 2, 6, 0, 0, 0, 0, 1, 3, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0],
    [2, 0, 2, 1, 3, 6, 3, 3, 1, 2, 2, 2, 0, 1, 0, 2, 2, 6, 2, 1, 2, 1],
    [5, 0, 4, 6, 4, 6, 4, 3, 2, 4, 4, 6, 1, 3, 2, 5, 5, 6, 5, 5, 5, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 2, 3, 0, 1],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 6, 0, 0, 1, 6],
    [5, 3, 6, 5, 3, 6, 4, 2, 5, 6, 5, 6, 1, 3, 2, 4, 4, 1, 6, 5, 4, 5],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [5, 1, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 3, 3, 6, 5, 5, 4, 5, 5, 6, 5],
    [6, 1, 4, 6, 2, 1, 2, 1, 1, 1, 1, 3, 0, 1, 1, 2, 2, 6, 3, 2, 1, 0],
    [6, 0, 4, 0, 4, 6, 0, 0, 0, 3, 4, 5, 0, 1, 0, 3, 3, 0, 4, 4, 4, 2],
    [4, 1, 1, 0, 4, 1, 0, 0, 2, 2, 4, 3, 0, 1, 0, 0, 6, 1, 4, 6, 5, 6],
    [6, 0, 1, 5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 3, 0, 2, 6, 4, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 2],
    [6, 1, 3, 2, 3, 6, 2, 1, 0, 6, 4, 6, 3, 2, 4, 3, 4, 2, 3, 4, 2, 2],
    [0, 0, 6, 0, 0, 6, 0, 0, 0, 6, 3, 3, 0, 1, 0, 2, 2, 0, 4, 3, 2, 5],
    [6, 5, 6, 6, 6, 6, 5, 5, 6, 6, 6, 6, 5, 3, 3, 5, 5, 0, 6, 4, 3, 6],
    [3, 1, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 4, 0, 1, 1, 0, 0],
    [6, 6, 6, 6, 1, 6, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 2, 1],
    [4, 3, 2, 4, 6, 4, 5, 6, 6, 4, 0, 3, 1, 3, 3, 3, 5, 6, 4, 3, 4, 5],
    [5, 5, 5, 1, 6, 6, 6, 6, 2, 1, 6, 6, 1, 6, 3, 6, 6, 4, 6, 6, 5, 6]]
""",
    [1, 0, 4, 1, 0, 6, 1, 1, 3, 1, 1, 1, 0, 1, 0, 4, 4, 1, 2, 4, 4, 4],
    [6, 1, 0, 5, 0, 6, 1, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 3, 6, 6, 2, 6],
    [6, 1, 5, 6, 1, 6, 1, 0, 1, 6, 5, 5, 5, 4, 2, 4, 5, 0, 3, 2, 3, 5],
    [4, 1, 4, 4, 3, 6, 3, 6, 0, 6, 6, 6, 1, 4, 5, 6, 6, 6, 6, 6, 6, 6],
    [6, 1, 2, 6, 2, 6, 6, 6, 6, 6, 6, 6, 2, 6, 0, 6, 6, 0, 6, 6, 6, 6],
    [6, 0, 6, 6, 6, 6, 4, 6, 6, 6, 6, 5, 3, 4, 3, 6, 6, 0, 6, 6, 5, 6],
    [1, 1, 4, 0, 6, 6, 3, 2, 6, 6, 5, 6, 1, 4, 2, 2, 3, 4, 1, 3, 4, 6],
    [0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 3, 2, 4],
    [0, 0, 0, 0, 0, 0, 2, 0, 4, 1, 1, 0, 0, 0, 0, 1, 1, 2, 0, 2, 3, 4],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 6, 2, 0, 1, 0, 6, 6, 0, 6, 6, 6, 0],
    [4, 3, 4, 4, 4, 4, 2, 2, 3, 6, 6, 6, 1, 6, 2, 6, 6, 6, 6, 6, 6, 4],
    [5, 1, 6, 5, 5, 6, 5, 4, 6, 6, 5, 6, 0, 1, 0, 6, 5, 5, 4, 5, 5, 6],
    [6, 4, 0, 0, 2, 6, 0, 0, 3, 1, 2, 2, 1, 2, 0, 4, 4, 6, 4, 4, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2],
    [6, 0, 6, 6, 0, 4, 0, 1, 6, 3, 6, 6, 5, 0, 3, 6, 6, 0, 6, 6, 3, 6],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 3, 0, 0, 0],
    [3, 1, 2, 3, 3, 1, 3, 2, 4, 6, 6, 6, 3, 2, 3, 6, 5, 0, 6, 5, 5, 3],
    [5, 1, 5, 6, 6, 1, 3, 3, 0, 1, 0, 6, 0, 1, 0, 1, 5, 0, 3, 6, 3, 4]
]"""
rows = len(Human_answers2022)
cols = len(Human_answers2022[0])
zeros_list = [[0 for _ in range(cols)] for _ in range(rows)]
Machine_answer2022=zeros_list
rows = len(Human_answers2023)
cols = len(Human_answers2023[0])
Machine_answer2023=[[0 for _ in range(cols)] for _ in range(rows)]


for k in range(0,22):
    
        file_path=f"C:\\Users\\reihaneh.maarefdoust\\Desktop\\ERisk\\test-llama3-f\\clear-2022-test-answer-Question{k}.txt"
        
        with open(file_path,'r') as file:
            text=file.read()
        for i in range(0,28):
           if text[i].isdigit():
                Machine_answer2022[i][k]=int(text[i])
           else:
               Machine_answer2022[i][k]=10


print(Machine_answer2022)


for k in range(0,22):
    
        file_path=f"C:\\Users\\reihaneh.maarefdoust\\Desktop\\ERisk\\test-llama3-f\\clear-2023-test-answer-Question{k}.txt"
        
        with open(file_path,'r') as file:
            text=file.read()
        for i in range(0,28):#46):
           if text[i].isdigit():
                Machine_answer2023[i][k]=int(text[i])
           else:
               Machine_answer2023[i][k]=10


print(Machine_answer2023)


def calculate_mzoe(human_answers, machine_answers):
    num_questions = len(human_answers[0])
    total_errors = 0

    for human, machine in zip(human_answers, machine_answers):
        errors = sum(h != m for h, m in zip(human, machine))
        total_errors += errors

    mzoe = total_errors / (len(human_answers) * num_questions)
    return mzoe

# Calculate MZOE
MZOE = calculate_mzoe(Human_answers2022, Machine_answer2022)
print("2022 Mean Zero-One Error (MZOE):", MZOE)

MZOE = calculate_mzoe(Human_answers2023, Machine_answer2023)
print("2023  Mean Zero-One Error (MZOE):", MZOE)

# Function to calculate MAE
def calculate_mae(human_answers, machine_answers):
    num_questions = len(human_answers[0])
    total_absolute_error = 0

    for human, machine in zip(human_answers, machine_answers):
        absolute_errors = sum(abs(h - m) for h, m in zip(human, machine))
        total_absolute_error += absolute_errors

    mae = total_absolute_error / (len(human_answers) * num_questions)
    return mae

# Calculate MAE
MAE = calculate_mae(Human_answers2022, Machine_answer2022)
print("2022 Mean Absolute Error (MAE):", MAE)

MAE = calculate_mae(Human_answers2023, Machine_answer2023)
print("2023 Mean Absolute Error (MAE):", MAE)

# Function to calculate MAE macro
def calculate_maemacro(human_answers, machine_answers):
    num_answers = 7  # Number of possible answers (0 to 6)
    total_absolute_error = 0
    total_questions = 0

    for j in range(num_answers):
        num_questions_j = sum(ans.count(j) for ans in human_answers)
        total_questions += num_questions_j

        absolute_error_j = sum(abs(h - m) for human, machine in zip(human_answers, machine_answers) for h, m in zip(human, machine) if h == j)
        total_absolute_error += absolute_error_j / num_questions_j if num_questions_j != 0 else 0

    maemacro = total_absolute_error / num_answers
    return maemacro

# Calculate MAE macro
MAEmacro = calculate_maemacro(Human_answers2022, Machine_answer2022)
print("2022 Macroaveraged Mean Absolute Error (MAEmacro):", MAEmacro)
MAEmacro = calculate_maemacro(Human_answers2023, Machine_answer2023)
print("2023 Macroaveraged Mean Absolute Error (MAEmacro):", MAEmacro)

import math

# Function to calculate RMSE for Restraint Subscale
def calculate_rmse_rs(real_scores, estimated_scores):
    total_squared_error = sum((rrs - frs) ** 2 for rrs, frs in zip(real_scores, estimated_scores))
    rmse = math.sqrt(total_squared_error / len(real_scores))
    return rmse


# Function to calculate RMSE for Restraint Subscale
def calculate_rmse_rs(real_scores, estimated_scores):
    total_squared_error = sum((rrs - frs) ** 2 for rrs, frs in zip(real_scores, estimated_scores))
    rmse = math.sqrt(total_squared_error / len(real_scores))
    return rmse


# Extract restraint scores from Human_answers2022 and Machine_answer2022
human_restraint_scores_2022 = [sum(user_answers[:5]) / 5 for user_answers in Human_answers2022]
machine_restraint_scores_2022 = [sum(user_answers[:5]) / 5 for user_answers in Machine_answer2022]

# Calculate RMSE for Restraint Subscale
RMSE_rs_2022 = calculate_rmse_rs(human_restraint_scores_2022, machine_restraint_scores_2022)
print("2022 Root Mean Square Error (RMSE) for Restraint Subscale (RS):", RMSE_rs_2022)

# Assuming Human_answers2023 and Machine_answer2023 are lists containing the real and estimated restraint scores for each user respectively

# Extract restraint scores from Human_answers2023 and Machine_answer2023
human_restraint_scores_2023 = [sum(user_answers[:5]) / 5 for user_answers in Human_answers2023]
machine_restraint_scores_2023 = [sum(user_answers[:5]) / 5 for user_answers in Machine_answer2023]

# Calculate RMSE for Restraint Subscale
RMSE_rs_2023 = calculate_rmse_rs(human_restraint_scores_2023, machine_restraint_scores_2023)
print("2023 Root Mean Square Error (RMSE) for Restraint Subscale (RS):", RMSE_rs_2023)



# Function to calculate RMSE for Eating Concern Subscale
def calculate_rmse_ecs(real_scores, estimated_scores):
    total_squared_error = sum((recs - fecs) ** 2 for recs, fecs in zip(real_scores, estimated_scores))
    rmse = math.sqrt(total_squared_error / len(real_scores))
    return rmse

# Extract eating concern scores from Human_answers2022 and Machine_answer2022
human_ecs_scores_2022 = [(user_answers[6] + user_answers[8] + user_answers[12] + user_answers[13] + user_answers[14]) / 5 for user_answers in Human_answers2022]
machine_ecs_scores_2022 = [(user_answers[6] + user_answers[8] + user_answers[12] + user_answers[13] + user_answers[14]) / 5 for user_answers in Machine_answer2022]

# Calculate RMSE for Eating Concern Subscale
RMSE_ecs_2022 = calculate_rmse_ecs(human_ecs_scores_2022, machine_ecs_scores_2022)
print("2022 Root Mean Square Error (RMSE) for Eating Concern Subscale (ECS):", RMSE_ecs_2022)

# Extract eating concern scores from Human_answers2023 and Machine_answer2023
human_ecs_scores_2023 = [(user_answers[6] + user_answers[8] + user_answers[12] + user_answers[13] + user_answers[14]) / 5 for user_answers in Human_answers2023]
machine_ecs_scores_2023 = [(user_answers[6] + user_answers[8] + user_answers[12] + user_answers[13] + user_answers[14]) / 5 for user_answers in Machine_answer2023]

# Calculate RMSE for Eating Concern Subscale
RMSE_ecs_2023 = calculate_rmse_ecs(human_ecs_scores_2023, machine_ecs_scores_2023)
print("2023 Root Mean Square Error (RMSE) for Eating Concern Subscale (ECS):", RMSE_ecs_2023)

import math

# Function to calculate RMSE for Shape Concern Subscale
def calculate_rmse_scs(real_scores, estimated_scores):
    total_squared_error = sum((rscs - fscs) ** 2 for rscs, fscs in zip(real_scores, estimated_scores))
    rmse = math.sqrt(total_squared_error / len(real_scores))
    return rmse


# Extract shape concern scores from Human_answers2022 and Machine_answer2022
human_scs_scores_2022 = [(user_answers[5] + user_answers[7] + user_answers[16] + user_answers[9] + user_answers[19] + user_answers[20] + user_answers[21] + user_answers[10]) / 8 for user_answers in Human_answers2022]
machine_scs_scores_2022 = [(user_answers[5] + user_answers[7] + user_answers[16] + user_answers[9] + user_answers[19] + user_answers[20] + user_answers[21] + user_answers[10]) / 8 for user_answers in Machine_answer2022]

# Calculate RMSE for Shape Concern Subscale
RMSE_scs_2022 = calculate_rmse_scs(human_scs_scores_2022, machine_scs_scores_2022)
print("2022 Root Mean Square Error (RMSE) for Shape Concern Subscale (SCS):", RMSE_scs_2022)


# Extract shape concern scores from Human_answers2023 and Machine_answer2023
human_scs_scores_2023 = [(user_answers[5] + user_answers[7] + user_answers[16] + user_answers[9] + user_answers[19] + user_answers[20] + user_answers[21] + user_answers[10]) / 8 for user_answers in Human_answers2023]
machine_scs_scores_2023 = [(user_answers[5] + user_answers[7] + user_answers[16] + user_answers[9] + user_answers[19] + user_answers[20] + user_answers[21] + user_answers[10]) / 8 for user_answers in Machine_answer2023]

# Calculate RMSE for Shape Concern Subscale
RMSE_scs_2023 = calculate_rmse_scs(human_scs_scores_2023, machine_scs_scores_2023)
print("2023 Root Mean Square Error (RMSE) for Shape Concern Subscale (SCS):", RMSE_scs_2023)



# Function to calculate RMSE for Weight Concern Subscale
def calculate_rmse_wcs(real_scores, estimated_scores):
    total_squared_error = sum((rwcs - fwcs) ** 2 for rwcs, fwcs in zip(real_scores, estimated_scores))
    rmse = math.sqrt(total_squared_error / len(real_scores))
    return rmse


# Extract weight concern scores from Human_answers2022 and Machine_answer2022
human_wcs_scores_2022 = [(user_answers[15] + user_answers[17] + user_answers[7] + user_answers[18] + user_answers[11]) / 5 for user_answers in Human_answers2022]
machine_wcs_scores_2022 = [(user_answers[15] + user_answers[17] + user_answers[7] + user_answers[18] + user_answers[11]) / 5 for user_answers in Machine_answer2022]

# Calculate RMSE for Weight Concern Subscale
RMSE_wcs_2022 = calculate_rmse_wcs(human_wcs_scores_2022, machine_wcs_scores_2022)
print("2022 Root Mean Square Error (RMSE) for Weight Concern Subscale (WCS):", RMSE_wcs_2022)


# Extract weight concern scores from Human_answers2023 and Machine_answer2023
human_wcs_scores_2023 = [(user_answers[15] + user_answers[17] + user_answers[7] + user_answers[18] + user_answers[11]) / 5 for user_answers in Human_answers2023]
machine_wcs_scores_2023 = [(user_answers[15] + user_answers[17] + user_answers[7] + user_answers[18] + user_answers[11]) / 5 for user_answers in Machine_answer2023]

# Calculate RMSE for Weight Concern Subscale
RMSE_wcs_2023 = calculate_rmse_wcs(human_wcs_scores_2023, machine_wcs_scores_2023)
print("2023 Root Mean Square Error (RMSE) for Weight Concern Subscale (WCS):", RMSE_wcs_2023)


real_global_scores = [sum(scores) / 4 for scores in zip(*[human_wcs_scores_2023, human_ecs_scores_2023, human_scs_scores_2023, human_wcs_scores_2023])]
estimated_global_scores = [sum(scores) / 4 for scores in zip(*[machine_restraint_scores_2023, machine_ecs_scores_2023, machine_scs_scores_2023, machine_wcs_scores_2023])]

real_global_array = np.array(real_global_scores)
estimated_global_array = np.array(estimated_global_scores)

# Calculate RMSE for global ED scores
RMSE_global = np.sqrt(np.mean((real_global_array - estimated_global_array) ** 2))
print("Root Mean Square Error (RMSE) for Global Eating Disorder (ED) scores:", RMSE_global)


