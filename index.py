import pandas as pd
import keras

a = [1, 7, 2]

# myvar = pandas.DataFrame(mydataset["bikes"]) 
myvar = pd.Series(a)
df = pd.read_json("package-lock.json")



print(myvar[2]) 

print(df) 

hnorm = (df - df.min)/(df.max-df.min)

target = df.pop("medain-house-value")

y=target.values

x=hnorm.values
