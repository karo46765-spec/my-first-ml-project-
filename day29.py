from sklearn.utils._repr_html import estimator
import yaml

config_path = "config.yaml"

with open(config_path,"r") as stream:
    config = yaml.safe_load(stream)
    print(config)

model_name =  config['model']['name']
estimators =  config['model']['n_estimators']
output     =  config['paths']['model_output']

print(model_name,estimators,output)