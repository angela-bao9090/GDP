import seaborn as sns
import matplotlib.pyplot as plt



class ConfusionMatrix:
    '''
    Class to plot a confusion matrix
    
    parameters:
        cms - Array of confusion matrices
        titles - Array of corresponding titles for each confusion matrix in cms
    '''
    
    def __init__(self, cms, titles): # Note: len(cms) should be the same as len(title)
        # Plot the confusion matrices
        
        self.num_matrices = len(cms)
        
        # Create grid layout
        self.rows = self.num_matrices // 2 + (self.num_matrices % 2)  
        self.cols = 3
        
        _, self.axes = plt.subplots(self.rows, self.cols, figsize=(9, 3 * self.rows))  # Adjust the size based on number of rows
        self.axes = self.axes.flatten()

        
        for i, cm in enumerate(cms):
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=self.axes[i], cbar=False, 
                        xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
            self.axes[i].set_title(f'{titles[i]} {i + 1}')
            self.axes[i].set_xlabel('Predicted')
            self.axes[i].set_ylabel('Actual')

        # Hide any unused subplots
        for j in range(i + 1, len(self.axes)):
            self.axes[j].axis('off')

        plt.tight_layout()
        plt.show()