import torch

def evaluate_top_x_accuracy(model_output, ground_truth_numbers, x):
    top_x_correct = 0
    for i in range(len(model_output)):
        # Extract top x predictions
        top_x_predictions = torch.argsort(model_output[i])[-x:].numpy()
        
        # Check if any of the top x predictions are correct
        if any(pred in range(ground_truth_numbers[i]*10, ground_truth_numbers[i]*10+10) for pred in top_x_predictions):
            top_x_correct += 1

    accuracy = top_x_correct / len(model_output) * 100
    return top_x_correct, accuracy
