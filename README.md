# COMP-4730-final-project
Make a research data analysis project using multiple learning approaches. You should use your own initiative to define one or several research questions and use the resources available to apply different approaches, using any Python libraries, to this data to predict accurately the class or the value of the unlabeled data and to answer your research questions.
This Project used the CIFAR-100 dataset. 
the page below has all the information about the dataset. 
`https://www.cs.toronto.edu/~kriz/cifar.html`

# Project Structure
- `train.py` is the main script which has the setup for training the models
- `Models` folder has all the architectures defined inside of them.
- `main_pca.py` code to run PCA which is for future work
- `main_printingdata.py` was used to visualize our data 
	
# Setup Instructions
```bash
# 1. Clone or Download the Project Repository
git clone <repository-url>

# 2. Navigate to the Project Directory
cd project-directory

# 3. Create a Virtual Environment (replace <env-name> with your preferred name)
python -m venv <env-name>

# 4. Activate the Virtual Environment (Windows)
.\<env-name>\Scripts\activate

# 4. Activate the Virtual Environment (macOS and Linux)
source <env-name>/bin/activate

# 5. Install Project Dependencies
pip install -r requirements.txt

# To run the model we use option parser to select the configuration 
# -m takes in the model name
# -s is the session
# Run CNN
# for example, To run the cnn
# you would type the following 
train.py -m cnn -s 1
```
# SSL Error 
if you encounter an error with SSL verification and cant download the dataset, simply add this piece of code at the top of `train.py`. 
```py
import ssl
ssl._create_default_https_context = ssl_create_unverified_context
```

# Contributors
Akshat Sharma <br>
Justin Neal <br>
George Kaceli 