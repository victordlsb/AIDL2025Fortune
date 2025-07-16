import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def create_training_plot(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None, title_prefix="Training and Validation"):
    """
    Create training and validation metrics plots
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        save_path: If provided, saves the plot to this path
        title_prefix: Prefix for plot titles
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_title(f'{title_prefix} Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add min/max annotations
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    min_train_epoch = train_losses.index(min_train_loss) + 1
    min_val_epoch = val_losses.index(min_val_loss) + 1
    
    ax1.annotate(f'Min Train: {min_train_loss:.4f}', 
                xy=(min_train_epoch, min_train_loss), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                fontsize=9)
    
    ax1.annotate(f'Min Val: {min_val_loss:.4f}', 
                xy=(min_val_epoch, min_val_loss), 
                xytext=(10, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                fontsize=9)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    ax2.set_title(f'{title_prefix} Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add max accuracy annotations
    max_train_acc = max(train_accuracies)
    max_val_acc = max(val_accuracies)
    max_train_epoch = train_accuracies.index(max_train_acc) + 1
    max_val_epoch = val_accuracies.index(max_val_acc) + 1
    
    ax2.annotate(f'Max Train: {max_train_acc:.4f}', 
                xy=(max_train_epoch, max_train_acc), 
                xytext=(10, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                fontsize=9)
    
    ax2.annotate(f'Max Val: {max_val_acc:.4f}', 
                xy=(max_val_epoch, max_val_acc), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)
    
    return min_train_loss, min_val_loss, min_train_epoch, min_val_epoch, max_train_acc, max_val_acc, max_train_epoch, max_val_epoch

def load_and_plot_metrics(metrics_file="training_metrics.pt", save_path="training_metrics.png"):
    """
    Load and plot training metrics from a saved file
    
    Args:
        metrics_file: Path to the saved metrics file
        save_path: Path to save the plot
    """
    if not os.path.exists(metrics_file):
        print(f"❌ Metrics file {metrics_file} not found!")
        return
    
    # Load metrics
    metrics = torch.load(metrics_file)
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    train_accuracies = metrics['train_accuracies']
    val_accuracies = metrics['val_accuracies']
    
    # Create plot and get statistics
    min_train_loss, min_val_loss, min_train_epoch, min_val_epoch, max_train_acc, max_val_acc, max_train_epoch, max_val_epoch = create_training_plot(
        train_losses, val_losses, train_accuracies, val_accuracies, save_path=save_path
    )
    
    # Print summary statistics
    epochs = range(1, len(train_losses) + 1)
    print(f"\n📊 Training Summary:")
    print(f"  Total epochs: {len(epochs)}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}")
    print(f"  Final train accuracy: {train_accuracies[-1]:.4f}")
    print(f"  Final val accuracy: {val_accuracies[-1]:.4f}")
    print(f"  Best train loss: {min_train_loss:.4f} (epoch {min_train_epoch})")
    print(f"  Best val loss: {min_val_loss:.4f} (epoch {min_val_epoch})")
    print(f"  Best train accuracy: {max_train_acc:.4f} (epoch {max_train_epoch})")
    print(f"  Best val accuracy: {max_val_acc:.4f} (epoch {max_val_epoch})")
    
    # Print test results if available
    if 'final_test_loss' in metrics:
        print(f"\n🧪 Test Results:")
        print(f"  Final model test loss: {metrics['final_test_loss']:.4f}")
        print(f"  Final model test accuracy: {metrics['final_test_acc']:.4f}")
    
    print(f"\n📊 Plot saved as: {save_path}")

def create_loss_components_plot(train_classification_losses, val_classification_losses, train_regression_losses, val_regression_losses, save_path=None, title_prefix="Loss Components", horizons=None):
    """
    Create plots for classification and regression loss components with meaningful labels
    
    Args:
        train_classification_losses: List of lists - each inner list contains classification losses for that epoch
        val_classification_losses: List of lists - each inner list contains classification losses for that epoch
        train_regression_losses: List of lists - each inner list contains regression losses for that epoch
        val_regression_losses: List of lists - each inner list contains regression losses for that epoch
        save_path: If provided, saves the plot to this path
        title_prefix: Prefix for plot titles
        horizons: Dictionary of horizon names (e.g., {'1h': ..., '1d': ...}) for meaningful labels
    """
    epochs = range(1, len(train_classification_losses) + 1)
    
    # Define label names for meaningful plotting (updated for 3-label structure)
    classification_label_names = ['ranking', 'risk']
    regression_label_names = ['future_return']
    
    # If horizons not provided, use default names
    if horizons is None:
        horizon_names = ['1h', '1d']  # Default assumption based on parameters.py
    else:
        horizon_names = list(horizons.keys())
    
    # Create meaningful label names for each component
    def get_meaningful_labels(label_names, horizon_names):
        meaningful_labels = []
        for horizon in horizon_names:
            for label in label_names:
                meaningful_labels.append(f"{label}_{horizon}")
        return meaningful_labels
    
    classification_meaningful_labels = get_meaningful_labels(classification_label_names, horizon_names)
    regression_meaningful_labels = get_meaningful_labels(regression_label_names, horizon_names)
    
    # Create the plot with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Plot classification losses
    ax1.set_title(f'{title_prefix} - Classification Losses', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot each classification loss component
    max_classification_components = max(len(losses) for losses in train_classification_losses + val_classification_losses if losses)
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, max_classification_components))
    
    for component_idx in range(max_classification_components):
        # Extract this component across all epochs
        train_component = []
        val_component = []
        
        for epoch_losses in train_classification_losses:
            if component_idx < len(epoch_losses):
                train_component.append(epoch_losses[component_idx])
            else:
                train_component.append(np.nan)
        
        for epoch_losses in val_classification_losses:
            if component_idx < len(epoch_losses):
                val_component.append(epoch_losses[component_idx])
            else:
                val_component.append(np.nan)
        
        # Get meaningful label name
        label_name = classification_meaningful_labels[component_idx] if component_idx < len(classification_meaningful_labels) else f"classification_{component_idx+1}"
        
        # Plot this component
        ax1.plot(epochs, train_component, color=colors[component_idx], linestyle='-', 
                label=f'Train {label_name}', linewidth=2, marker='o', markersize=3)
        ax1.plot(epochs, val_component, color=colors[component_idx], linestyle='--', 
                label=f'Val {label_name}', linewidth=2, marker='s', markersize=3)
    
    ax1.legend(fontsize=9, loc='upper right')
    
    # Plot regression losses
    ax2.set_title(f'{title_prefix} - Regression Losses', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Squared Error Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot each regression loss component
    max_regression_components = max(len(losses) for losses in train_regression_losses + val_regression_losses if losses)
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, max_regression_components))
    
    for component_idx in range(max_regression_components):
        # Extract this component across all epochs
        train_component = []
        val_component = []
        
        for epoch_losses in train_regression_losses:
            if component_idx < len(epoch_losses):
                train_component.append(epoch_losses[component_idx])
            else:
                train_component.append(np.nan)
        
        for epoch_losses in val_regression_losses:
            if component_idx < len(epoch_losses):
                val_component.append(epoch_losses[component_idx])
            else:
                val_component.append(np.nan)
        
        # Get meaningful label name
        label_name = regression_meaningful_labels[component_idx] if component_idx < len(regression_meaningful_labels) else f"Regression_{component_idx+1}"
        
        # Plot this component
        ax2.plot(epochs, train_component, color=colors[component_idx], linestyle='-', 
                label=f'Train {label_name}', linewidth=2, marker='o', markersize=3)
        ax2.plot(epochs, val_component, color=colors[component_idx], linestyle='--', 
                label=f'Val {label_name}', linewidth=2, marker='s', markersize=3)
    
    ax2.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"📊 Loss components plot saved to {save_path}")
    
    plt.show()
    plt.close(fig)

def create_metrics_plot(train_f1_scores, val_f1_scores, train_smapes, val_smapes, save_path=None, title_prefix="Additional Metrics"):
    """
    Create plots for F1 scores and SMAPE metrics
    
    Args:
        train_f1_scores: List of training F1 scores
        val_f1_scores: List of validation F1 scores
        train_smapes: List of training SMAPE values
        val_smapes: List of validation SMAPE values
        save_path: If provided, saves the plot to this path
        title_prefix: Prefix for plot titles
    """
    epochs = range(1, len(train_f1_scores) + 1)
    
    # Create the plot with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot F1 scores
    ax1.plot(epochs, train_f1_scores, 'b-', label='Training F1', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_f1_scores, 'r-', label='Validation F1', linewidth=2, marker='s', markersize=4)
    ax1.set_title(f'{title_prefix} F1 Score', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add min/max annotations for F1
    max_train_f1 = max(train_f1_scores)
    max_val_f1 = max(val_f1_scores)
    max_train_epoch = train_f1_scores.index(max_train_f1) + 1
    max_val_epoch = val_f1_scores.index(max_val_f1) + 1
    
    ax1.annotate(f'Max Train: {max_train_f1:.4f}', 
                xy=(max_train_epoch, max_train_f1), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                fontsize=9)
    
    ax1.annotate(f'Max Val: {max_val_f1:.4f}', 
                xy=(max_val_epoch, max_val_f1), 
                xytext=(10, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                fontsize=9)
    
    # Plot SMAPE
    ax2.plot(epochs, train_smapes, 'g-', label='Training SMAPE', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_smapes, 'orange', label='Validation SMAPE', linewidth=2, marker='s', markersize=4)
    ax2.set_title(f'{title_prefix} SMAPE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('SMAPE (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add min/max annotations for SMAPE
    min_train_smape = min(train_smapes)
    min_val_smape = min(val_smapes)
    min_train_epoch = train_smapes.index(min_train_smape) + 1
    min_val_epoch = val_smapes.index(min_val_smape) + 1
    
    ax2.annotate(f'Min Train: {min_train_smape:.2f}%', 
                xy=(min_train_epoch, min_train_smape), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                fontsize=9)
    
    ax2.annotate(f'Min Val: {min_val_smape:.2f}%', 
                xy=(min_val_epoch, min_val_smape), 
                xytext=(10, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.3),
                fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"📊 Metrics plot saved to {save_path}")
    
    plt.show()
    plt.close(fig)

def create_per_label_accuracy_plot(train_per_label_accuracies, val_per_label_accuracies, save_path=None, title_prefix="Per-Label Accuracy"):
    """
    Create per-label accuracy plots for each label type (updated for 3-label structure)
    
    Args:
        train_per_label_accuracies: Dictionary with label names as keys and lists of training accuracies as values
        val_per_label_accuracies: Dictionary with label names as keys and lists of validation accuracies as values
        save_path: If provided, saves the plot to this path
        title_prefix: Prefix for plot titles
    """
    # Extract label names and data (updated for new structure)
    label_names = ['ranking', 'risk']
    
    # Check if we have any data
    if not train_per_label_accuracies or not any(train_per_label_accuracies.values()):
        print("⚠️  No per-label accuracy data to plot")
        return
    
    # Get number of epochs
    epochs = range(1, len(train_per_label_accuracies[label_names[0]]) + 1)
    
    # Create subplots - 1x2 grid for the 2 labels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{title_prefix} Metrics', fontsize=16, fontweight='bold')
    
    # Plot each label
    for i, label_name in enumerate(label_names):
        ax = axes[i]
        
        # Get data for this label
        train_data = train_per_label_accuracies.get(label_name, [])
        val_data = val_per_label_accuracies.get(label_name, [])
        
        if train_data and val_data:
            # Plot training and validation accuracy for this label
            ax.plot(epochs, train_data, 'b-', label=f'Training {label_name.replace("_", " ").title()}', linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, val_data, 'r-', label=f'Validation {label_name.replace("_", " ").title()}', linewidth=2, marker='s', markersize=4)
            
            # Add best validation marker
            best_val_epoch = np.argmax(val_data) + 1
            best_val_acc = max(val_data)
            ax.axvline(x=float(best_val_epoch), color='green', linestyle='--', alpha=0.7, label=f'Best Val (Epoch {best_val_epoch})')
            ax.plot(best_val_epoch, best_val_acc, 'go', markersize=8, label=f'Best: {best_val_acc:.3f}')
            
            # Customize the subplot
            ax.set_title(f'{label_name.replace("_", " ").title()} Accuracy', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Balanced Accuracy')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)  # Accuracy is between 0 and 1
            
            # Add final values as text
            final_train = train_data[-1] if train_data else 0
            final_val = val_data[-1] if val_data else 0
            ax.text(0.02, 0.98, f'Final Train: {final_train:.3f}\nFinal Val: {final_val:.3f}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # No data for this label
            ax.text(0.5, 0.5, f'No data for {label_name.replace("_", " ").title()}', transform=ax.transAxes, 
                   fontsize=12, ha='center', va='center')
            ax.set_title(f'{label_name.replace("_", " ").title()} Accuracy', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"📊 Per-label accuracy plot saved to {save_path}")
    
    plt.show()
    plt.close(fig)


def create_sign_accuracy_plot(train_sign_accuracies, val_sign_accuracies, save_path=None, title_prefix="Returns Sign Accuracy"):
    """
    Create sign_accuracy plots 
    
    """
    # Extract label names and data (updated for new 2-label structure)
    label_names = ['returns sign']
    
    # Check if we have any data
    if not train_sign_accuracies or not val_sign_accuracies:
        print("⚠️  No signs accuracy data to plot")
        return

    epochs = range(1, len(train_sign_accuracies) + 1)
    
    # Create subplots - 1x2 grid for the 2 labels
    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle(f'{title_prefix} Metrics', fontsize=16, fontweight='bold')
    
    # Plot each label
    for i, label_name in enumerate(label_names):
        ax = axes
        
        # Get data for this label
        train_data = train_sign_accuracies
        val_data = val_sign_accuracies
        
        if train_data and val_data:
            # Ensure data are on CPU and converted to numpy arrays if they are torch tensors
            if isinstance(train_data, torch.Tensor):
                train_data = train_data.cpu().numpy()
            if isinstance(val_data, torch.Tensor):
                val_data = val_data.cpu().numpy()
            # Also handle list of tensors
            if len(train_data) > 0 and isinstance(train_data[0], torch.Tensor):
                train_data = np.array([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in train_data])
            if len(val_data) > 0 and isinstance(val_data[0], torch.Tensor):
                val_data = np.array([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in val_data])
            # Plot training and validation accuracy for this label
            ax.plot(epochs, train_data, 'b-', label=f'Training {label_name.replace("_", " ").title()}', linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, val_data, 'r-', label=f'Validation {label_name.replace("_", " ").title()}', linewidth=2, marker='s', markersize=4)
            # Add best validation marker
            best_val_epoch = np.argmax(val_data) + 1
            best_val_acc = max(val_data)
            ax.axvline(x=float(best_val_epoch), color='green', linestyle='--', alpha=0.7, label=f'Best Val (Epoch {best_val_epoch})')
            ax.plot(best_val_epoch, best_val_acc, 'go', markersize=8, label=f'Best: {best_val_acc:.3f}')
            # Customize the subplot
            ax.set_title(f'{label_name.replace("_", " ").title()} Accuracy', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Balanced Accuracy')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)  # Accuracy is between 0 and 1
            # Add final values as text
            final_train = train_data[-1] if len(train_data) > 0 else 0
            final_val = val_data[-1] if len(val_data) > 0 else 0
            ax.text(0.02, 0.98, f'Final Train: {final_train:.3f}\nFinal Val: {final_val:.3f}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # No data for this label
            ax.text(0.5, 0.5, f'No data for {label_name.replace("_", " ").title()}', transform=ax.transAxes, 
                   fontsize=12, ha='center', va='center')
            ax.set_title(f'{label_name.replace("_", " ").title()} Accuracy', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"📊 Returns sign accuracy plot saved to {save_path}")
    
    plt.show()
    plt.close(fig)


def create_per_label_f1_plot(train_per_label_f1s, val_per_label_f1s, save_path=None, title_prefix="Per-Label F1"):
    """
    Create per-label F1 plots for each label type (ranking, risk)
    Args:
        train_per_label_f1s: Dictionary with label names as keys and lists of training F1s as values
        val_per_label_f1s: Dictionary with label names as keys and lists of validation F1s as values
        save_path: If provided, saves the plot to this path
        title_prefix: Prefix for plot titles
    """
    label_names = ['ranking', 'risk']
    if not train_per_label_f1s or not any(train_per_label_f1s.values()):
        print("⚠️  No per-label F1 data to plot")
        return
    epochs = range(1, len(train_per_label_f1s[label_names[0]]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{title_prefix} Metrics', fontsize=16, fontweight='bold')
    for i, label_name in enumerate(label_names):
        ax = axes[i]
        train_data = train_per_label_f1s.get(label_name, [])
        val_data = val_per_label_f1s.get(label_name, [])
        if train_data and val_data:
            ax.plot(epochs, train_data, 'b-', label=f'Training {label_name.replace("_", " ").title()}', linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, val_data, 'r-', label=f'Validation {label_name.replace("_", " ").title()}', linewidth=2, marker='s', markersize=4)
            best_val_epoch = np.argmax(val_data) + 1
            best_val_f1 = max(val_data)
            ax.axvline(x=float(best_val_epoch), color='green', linestyle='--', alpha=0.7, label=f'Best Val (Epoch {best_val_epoch})')
            ax.plot(best_val_epoch, best_val_f1, 'go', markersize=8, label=f'Best: {best_val_f1:.3f}')
            ax.set_title(f'{label_name.replace("_", " ").title()} F1', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            final_train = train_data[-1] if train_data else 0
            final_val = val_data[-1] if val_data else 0
            ax.text(0.02, 0.98, f'Final Train: {final_train:.3f}\nFinal Val: {final_val:.3f}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'No data for {label_name.replace("_", " ").title()}', transform=ax.transAxes, 
                   fontsize=12, ha='center', va='center')
            ax.set_title(f'{label_name.replace("_", " ").title()} F1', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"📊 Per-label F1 plot saved to {save_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Load and plot metrics
    stocks = 10
    load_and_plot_metrics(metrics_file=f"training_metrics_{stocks}_stocks.pt", save_path=f"training_metrics_{stocks}_stocks.png")
