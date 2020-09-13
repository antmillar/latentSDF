
# TRAINING TEMPLATE

The Model_Trainer.ipynb file in the root directory provides a template for training the deepsdf model

It's best to run this in google colab with GPU activated 


# APPLICATION

# Instructions for Local Installation



Python and Conda must be installed in order to run the application. The easiest way to get these is to install https://docs.conda.io/en/latest/miniconda.html


##INSTALLATION##

From the anaconda prompt run:

gets the code:

	git clone https://github.com/antmillar/latentSDF.git

	cd latentSDF

creates a conda environment and installs all the required libraries:

	conda env create -f latentsdf.yml 

	conda activate latentsdf

##RUNNING##

	python app.py

Once the app is running you should see the following:


	 * Serving Flask app "app" (lazy loading)
	 * Environment: production
	   WARNING: This is a development server. Do not use it in a production deployment.
	   Use a production WSGI server instead.
	 * Debug mode: off
	 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

Open this address in a browser to load the application. The application is optimised to firefox, but should work in other browsers though the layout might not be optimal.




Notes:

conda may need to be manually added to your PATH if a conda not a recognized command error is returned.

Pre-prepared model zips can be found in /static/models/torch. The upload model button should target this directory by default.




 
