import matplotlib.pyplot as plt

def plot_training_progress(loss_history):
    losses = [log["loss"] for log in loss_history if "loss" in log]
    
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss", marker='o')
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    return plt
