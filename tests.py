import torch
import numpy as np

def evaluate_top_x_accuracy(model_output, ground_truth_numbers, x):
    top_x_correct = 0
    total_correct_predictions = 0
    correct_prediction_indexes = []

    for i in range(len(model_output)):
        # Extract top x predictions
        top_x_predictions = torch.argsort(model_output[i], descending=True)[:x].cpu().numpy()
        
        # Check how many of the top x predictions are correct
        correct_predictions = sum(pred in range(ground_truth_numbers[i]*10, ground_truth_numbers[i]*10+10) for pred in top_x_predictions)
        # Append the index of the correct prediction
        if correct_predictions > 0:
            correct_prediction_indexes.append(i)

        # If at least one correct, count this instance as correct
        if correct_predictions > 0:
            top_x_correct += 1

        # Sum the number of correct predictions
        total_correct_predictions += correct_predictions

    # summed_class_correct = []

    # for i in range(0, len(model_output), 10):
    #     # Sum the classification results for each set of 10
    #     summed_predictions = sum(torch.argmax(model_output[j]) for j in range(i, i + 10))

    #     # Compare with ground truth
    #     ground_truth_value = ground_truth_numbers[i // 10]

    #     summed_class_correct.append((summed_predictions, ground_truth_value))

    accuracy = top_x_correct / len(model_output) * 100

    return top_x_correct, total_correct_predictions, accuracy, correct_prediction_indexes