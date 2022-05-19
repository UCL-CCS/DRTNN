[![GitHub issues](https://img.shields.io/github/issues/UCL-CCS/DRTNN)](https://github.com/UCL-CCS/DRTNN/issues)
[![GitHub forks](https://img.shields.io/github/forks/UCL-CCS/DRTNN)](https://github.com/UCL-CCS/DRTNN/network)
[![GitHub stars](https://img.shields.io/github/stars/UCL-CCS/DRTNN)](https://github.com/UCL-CCS/DRTNN/stargazers)
[![GitHub license](https://img.shields.io/github/license/UCL-CCS/DRTNN)](https://github.com/UCL-CCS/DRTNN/blob/master/LICENSE)


# [Deep Residual Transformer Neural Network (DRTNN)]


<p float="center">
  <img src="images/σ01_ε01.gif" width="271" />
  <img src="images/σ02_ε02.gif" width="271" /> 
  <img src="images/σ11_ε11.gif" width="271" />
</p>

<br>
 <img height="400" src="images/st.png"/>
</br>

<p float="center">
  <img src="images/σ12_ε12.gif" width="271" />
  <img src="images/σ22_ε22.gif" width="271" /> 
  <img src="images/σ33_ε33.gif" width="271" />
</p>


<br>
 <img height="550" src="images/architecture.png"/>
</br>

#  Neural transformer network for predicting stress!

<br>
 <img height="550" src="images/scema_data.png"/>
</br>


<br>
 <img height="550" src="images/model.png"/>
</br>


# Running the Program! 

User needs first to install Anaconda https://www.anaconda.com/

Then


```sh
  - conda env create -f environment.yml
  or
  - conda create --name traintest --file Building_identical_conda_environment-file.txt
``` 
and 

```sh
  - conda activate traintest
``` 
finally

```sh
  - python  main.py
``` 

