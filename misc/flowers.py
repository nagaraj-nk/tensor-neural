import matplotlib.pyplot as plt
import numpy as np

# Function to plot simple petals for different flowers
def plot_flower(petals, color, title):
    angles = np.linspace(0, 2 * np.pi, petals + 1)
    radii = np.ones(petals + 1)
    fig, ax = plt.subplots(subplot_kw={'polar': True})
    ax.fill(angles, radii, color=color, alpha=0.6)
    ax.set_title(title, va='bottom')
    plt.show()

# Generate images of simple flowers
def generate_simple_flower_images():
    # Simple representation of a rose with 6 petals
    plot_flower(petals=6, color='red', title="Rose")
    
    # Simple representation of a lotus with 8 petals
    plot_flower(petals=8, color='pink', title="Lotus")
    
    # Simple representation of a jasmine with 5 petals
    plot_flower(petals=5, color='white', title="Jasmine")
    
    # Simple representation of a daisy with 10 petals
    plot_flower(petals=10, color='yellow', title="Daisy")

generate_simple_flower_images()
