import matplotlib.pyplot as plt

def plot_training_progress(loss_history):
    """
    Create a matplotlib figure showing training loss over steps.

    Args:
        loss_history (list): List of dictionaries with keys 'step' and 'loss'.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    steps = [entry["step"] for entry in loss_history]
    losses = [entry["loss"] for entry in loss_history]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, marker='o', linestyle='-')
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Progress")
    plt.tight_layout()
    return fig
