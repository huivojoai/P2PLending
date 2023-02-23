# LendingclubProject


DESCRIPTION - Describe the package in a few paragraphs

      This package can be used to train a credit risk model that predicts the probability of default given some credit-related and demographic features of the borrower typically found in P2P lending platform API datasets. By importing the specified dataset this package utilizes a pipeline to efficiently model the probability of default of P2P loans which eventually is used to aid the investment decisions of our optimized portfolio assistant showcased in the PowerBI dashboard. 
      
      The PowerBI interactive data visualization dashboard can be accessed online for a limited time via the link in the instructions below. The PowerBI file, optimization code, and the generated datasets are also available for assessment within this package. Finally, the respective jupyter notebook and python code files used for experiments are also included.
      
      A detailed explanation of the motivation and significance behind the data modeling and dashboard visualization created in this package can be found in the DOC subfolder which contains our official research paper and academic poster. 


EXECUTION - How to run a demo on your code

      1. P2P Lending Club Dashboard

         The interactive dashboard with a portfolio building assistant can be primarily accessed online through the following link:

         https://app.powerbi.com/Redirect?action=OpenApp&appId=dfc0c7ee-e4dc-4fbb-9402-41be5c36e231&ctid=e47a7e55-368c-4992-9292-8838a053cad2

         Please login to the following Microsoft account:

         user: test123@teamforteam.onmicrosoft.com  
         password: team4team@
	 
	 ** If a user is having trouble viewing the dashboard using the link above, the dashboard can also be opened locally using PowerBI Desktop **
	 
	 The Power BI file for this interactive dashboard can be downloaded from "./CODE/P2P Lending Dashboard.pbix". This file can be opened with the latest version of Power BI desktop downloaded from https://powerbi.microsoft.com/en-us/downloads/. Along with needing PowerBI desktop, it is also necessary to have installed a Python 3 version along with the pandas and matplotlib packages installed using pip, as Power BI does not work with Conda packages.
	 
	 ** Please Note **
	 	- When opening the dashboard on PowerBI desktop the user must click "Enable" when prompted to Enable script visuals upon opening the Power BI file.
         	- It is important to check if PowerBI is pointing to the correct Python environment or some visualizations may fail to execute. 
	 		- To select a python environment within PowerBI desktop a user can go to File -> Options and settings -> Options -> Python scripting and set a Python home directory which contains the necessary packages to run the visuals.


         The dashboard comprises four sections:

         1.1 Optimized Portfolios: Here, the lender can choose from a list of optimized portfolios according to their choice on a combo of loan grades and a target number of loans to diversify across:

         These optimized portfolios were calculated by solving an optimization problem for each combination of user selections, where:
         	- The objective function is to maximize the sum of the expected return for a portfolio.
         	- The optimal values are the binary variables representing the optimal loans for a portfolio.
        	- The constraints are the loan grade and the maximum number of loans.

         This problem was solved by a Python script using the PuLP library in the following Jupyter notebook: "./CODE/Portfolio Optimization/Expected Return optimization.ipynb".
	 
	 During the user experience, it is important to highlight that the Sharpe Ratio and Risk Diversification metrics take some time to be updated, due to complex calculations executed by Python scripts. Consequently, the lender needs to wait for some seconds until the related metrics are updated after each interaction.

         1.2 Portfolio Builder (Choose your own loans): Here, the lender browses through the test loans where they can choose any of them, building their own portfolio while the performance metrics are simultaneously updated.
		 
         1.3 Insights: Here, the lender has an overview of the P2P Market and visualize high-level summary statistics of loan and borrower characteristics.

	 1.4 Features: Here, feature importance visuals and trend charts are available for lenders.


INSTALLATION - How to install and setup code that builds model, generates data for PowerBI dashboard, and runs experiments

      1. Go into the "./CODE/" directory.
      
      2. Create a conda environment using the package requirements.txt file with the code below. Run the following lines in order.

         conda create --name team14_product_env python==3.8.13
         conda activate team14_product_env
         pip install -r requirements.txt
         conda install -c conda-forge scikit-learn
         conda install jupyterlab
         jupyter lab
	 
      3. Go to the link below and download the archive.zip file that containts the .gzip and .xlsx files.

         https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1
      
         Please open and place the 1.7GB data file named "Loan_status_2007-2020Q3.gzip" in the "./CODE/data/" path of this directory.

         (This step is needed for the "train.ipynb" which requires loading the full dataset.)

      4. In the "./CODE/Model Development" directory, we have two Jupyter notebooks
         
	 4.1 "train.ipynb" is a notebook that instantiates the creditrisk pipeline class to call the algorithm class with configuration and save the pipeline. (requires scikit-learn 1.0.2)
	 
	 Note: if warning messages show up with ERROR, we can safely ignore
         
	 In the "./CODE/config" directory, we have a .yaml file you can use to configurize different path for data/pipeline saving + update model hyperparameters
	 
	 
	 4.2 "automl.ipynb" is a notebook you can use to experiment models using PyCaret autoML library (requires scikit-learn 0.23.2)
	 	
		Note:
		- Currently the "clean & automl.ipynb" is experiencing an issue for users when running on the "team14_product_env" python environment installed above. The last few cells may return errors if the pycaret package fails to import.
		- This notebook was used strictly to test different ML models, and is NOT necessary to training the selected model used throughout our analysis
	 
      5. Experiments
	 
	     The following jupyter notebook files are available for assessment:
	  
		 5.1 Credit Risk Model Experiment which can be run at "./CODE/experiments/Credit Risk Model Experiment.ipynb". Reads in the list of optimized portfolios and calculates the actual realized return compared to similar benchmark portfolios of randomly selected loans.
		 
		 5.2 Optimized Portfolio Experiment which can be run at "./CODE/experiments/Portfolio Experiment.ipynb". Obtains the saved model pipeline and the test data to come up with helpful visualizations and model outputs.
		 
      
      (You can run any Jupyter notebook as-is, including the first cell which has pip install of specific scikit-learn versions)
