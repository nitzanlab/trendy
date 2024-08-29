# TRENDy (Temporal Regression of Effective Nonlinear Dynamics)
Reduced-order modeling for bifurcations in PDEs

## Install

Create a new conda virtual environment:
```
conda create -n trendy python=3.8
```

Activate the newly created environment:
```
conda activate trendy
```

Install this repo

For *production* usage:
```
pip install git+https://github.com/nitzanlab/trendy.git     # if you like to use git over https
```
OR for *development* usage:
```
git clone https://github.com/nitzanlab/trendy.git     # if you like to use git over https

pip install -e trendy
```

## Basic usage

Solving a PDE: 

```
pde = PDE(<pde_name>) # e.g. <pde_name> = 'GrayScott'
solution = pde.run()
```

PDE names are found in `./trendy/data/pde_configurations.json`

