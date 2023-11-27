import torch

def evaluate_top_x_accuracy(model_output, ground_truth_numbers, x):
    top_x_correct = 0
    total_correct_predictions = 0

    for i in range(len(model_output)):
        # Extract top x predictions
        top_x_predictions = torch.argsort(model_output[i], descending=True)[:x].numpy()
        
        # Check how many of the top x predictions are correct
        correct_predictions = sum(pred in range(ground_truth_numbers[i]*10, ground_truth_numbers[i]*10+10) for pred in top_x_predictions)
        
        # If at least one correct, count this instance as correct
        if correct_predictions > 0:
            top_x_correct += 1

        # Sum the number of correct predictions
        total_correct_predictions += correct_predictions

    accuracy = top_x_correct / len(model_output) * 100
    return top_x_correct, total_correct_predictions, accuracy
