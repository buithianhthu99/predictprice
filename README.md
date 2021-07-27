# Guide to run app (in windows)
## Step 1: Install python 3.9.4
- Get file download from https://www.python.org/downloads/windows/ 
- Install python 3.9.4
	* *Remember to check "Add python 3.9.4 to path"*
## Step 2: Install required modules
Open Windows PowerShell and run command:
-	pip install -r .\requirement.txt
	* *Make sure all installation success before go to next step*
## Step 3: Build model
Open Windows PowerShell and run command:
-	.\build_models.ps1
	* *Make sure build success before go to next step*
## Step 4: Run app
Open Windows PowerShell and run command:
- python .\stock_app.py

*See dashboard at: http://localhost:8050/*
