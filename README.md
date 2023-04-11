# LFP Toolbox

## Installation
Using anaconda:  
```  
conda env create --file=environment.yml  
```  

Using pip:  
```  
pip install -r requirements.txt  
```  

## Running  
```  
cd src && python -m webapp  
```  

## Updating
```
git fetch && git pull
```


# TODOs
- Support CMOS
- Extend analysis to spontaneous acitvity too
- Use [Fastplotlib](https://github.com/kushalkolar/fastplotlib) for plotting
- Use mne with MEA declared as ECoG for CSD, functional connectivities and the like.
- Detect bad channels using pyprep
