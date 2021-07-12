#### This package is meant to offer a variety of convenient inference functions, particularly for fine-tuned transformers.

Limited to [Jochen Hartmann's](https://github.com/j-hartmann) fine-tuned [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) for now.


##### Installs:
```
pip install numpy
pip install torch
pip install transformers
```

##### How to use:

```
from ezpredict import predict

preds = predict.predict_input(model_name="j-hartmann/emotion-english-distilroberta-base",
                             input=["What a beautiful day!"],
                              return_values=True,
                              print_values=True)
``` 
                        
               
                
                
##### Variables: 
 1. `model_name` -> name of model to perform inference on (limited to [Jochen Hartmann's](https://github.com/j-hartmann) fine-tuned [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) for now)
 2. `input` -> list of strings to perform predictions on
 3. `return_values` -> True/False, True: returns predictions as list of tuples 
 4. `print_values` -> True/False,  True: returns verbose outputs
 
 
##### Sample output:
`[('What a beautiful day!', {'anger': 0.0013013791, 'disgust': 0.00047031444, 'fear': 0.001256481, 'joy': 0.95772415, 'neutral': 0.005870249, 'sadness': 0.004233605, 'surprise': 0.029143812})]`



