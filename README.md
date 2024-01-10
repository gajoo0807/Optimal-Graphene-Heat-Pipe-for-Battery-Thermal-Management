# Optimal Graphene Heat Pipe Parameters for Enhanced Battery Thermal Management through Global Optimization
[Graphene Heat Pipe Optimization for Improved Battery Thermal Management](https://ieeexplore.ieee.org/abstract/document/10231239) - User Interface Code

In this work, we utilizes machine learning techniques to address the time and cost challenges in the field of nanotechnology. Through the Surrogate Model Algorithm, it becomes possible to explore potential optimization parameters with limited experimental data, thereby reducing the time and manpower costs associated with finding the optimal parameters. 

We received a dataset of 3780 graphene heat pipe records from the Industrial Technology Research Institute (ITRI). Using techniques such as regression, we aimed to identify correlations within the data. Finally, an interactive interface was implemented to allow users to interact with the program and explore the optimal parameters.

## Requirements
Ensure that you are using Windows 10 or a later version to run this code.
### Getting Started
1. Clone the repository to your local machine:
```
git clone https://github.com/gajoo0807/Optimal-Graphene-Heat-Pipe-for-Battery-Thermal-Management
```
2. Navigate to the project directory:
```
cd Optimal-Graphene-Heat-Pipe-for-Battery-Thermal-Management
```
3. Install the required dependencies using pip:
```
pip install -r requirements.txt
```
### Running the Code
To run the application, follow these steps:
1. Execute the following command:
2. Click on "New Model."
3. Choose a CSV file:
<br/>Select your current experimental data. If you want to view an example, you can choose "example.csv."
4. Choose the desired functionality:
* If you select "Regression," the model will perform regression analysis on the data. It will visualize the data, calculate Mean Squared Error (MSE), and provide the R-square score.
* If you choose "Parameter Optimization," you can pick a specific algorithm and fix certain parameters. The model will then find the optimal values for the remaining parameters. Users can conduct experiments based on these parameters, input the corresponding experimental results to facilitate the model in iterative learning.
