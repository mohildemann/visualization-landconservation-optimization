# Interactive visualization of land conservation optimization Pareto fronts

## Description
This App shall enable users to explore the resulting Pareto fronts of a land conservation optimization.
It is not possible to describe and discuss all optimal solutions of the three Ethiopoian study areas Gumobila, Ennerata and Mender 51, so feel free to explore!

## Requirements for running the app locally:
1. Install Python 3.8 or 3.9
2. Download or clone the repository (git clone https://github.com/mohildemann/visualization-landconservation-optimization.git)
3. Create virtual environment (here we call it venv)
4. Activate the virtual environment venv
5. Install the requirements into the virtual environment venv by typing "pip install -r requirements.txt" within your Python terminal
6. Run interactive_visualization.py
7. In your Python console the following text should appear: Dash is running on http://127.0.0.1:8050/
8. Click on the link

## Instructions for using the App
1. The default study area is Gumobila. If you want to change it to Ennerata or Mender 51, change it with the first drop down.
2. Click on one solution in the Pareto front if you want to see the corresponding map/spatial configuration
3. If you want to explore where terraces are proposed for allocation, click on the dropdown titled Plot bench terraces and change it from "No" to "Yes"
4. If you want to all objective values that belong to one solution, change the type of visualization from "Boxplot" to "Scatter plot"
