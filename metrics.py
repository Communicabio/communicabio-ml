import torch
import logging

def prediction_accuracy(inputs, predicted, ground_truth, delta=0.1):
    return (torch.abs(predicted - ground_truth)  < delta).double().mean().item()

def classification_accuracy(inputs, predicted, ground_truth):
    return (predicted.argmax() == ground_truth).double().mean().item()

def binary_accuracy(inputs, predicted, ground_truth):
    return (torch.abs(predicted - ground_truth) < .5).double().mean().item()
