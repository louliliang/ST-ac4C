# ST-ac4C
ST-ac4C, which identifies ac4C using Stacking model, using hybrid features composed of physico-chemical patterns and a distributed representation of nucleic acids.
datasets
A webserver is available at: http://www.rnanut.net/paces/ or at:http://lin-group.cn/server/iRNA-ac4C/.

The source code and datasets(both training and testing datasets) can be freely download from the github and the webserver page.

Brife tutorial
1. Environment requirements
Before running, please make sure the following packages are installed in Python environment:

umbalanced-learn=0.9.1
joblib=1.2.0
scikit-learn=1.1.2
pandas==1.2.2
python==3.9
numpy==1.23.2
For convenience, we strongly recommended users to install the Anaconda Python 3.9 (or above) in your local computer.

2. Running
Changing working dir to ST-ac4C, and then running the following command:
Select testpredict.py file, right-click to run.

3. Output
The output file (in ".csv" format) can be found in results folder, which including sequence name, predicted probability and redicted result.
Sequence with predicted probability > 0.5 was regared as ac4C site.

