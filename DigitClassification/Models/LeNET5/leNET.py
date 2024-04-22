# 1. Load the Dataset
# 2. Check its Datatype and make ready that fro pytorch
# 3. Visualise your data
# 4. Start making Your Model
# 5. Select a Loss Function & optimiser
# 6. Write Code for your training the Model
# 7. Write Code fro testing your Model
# 8. Write Code for evaluating your Model
#
#
#
#
#
#
#
#

import time
import customDataset
import printImages as prIm

startTime = time.time() # Start Time for the Project

loader = customDataset.CustomDataset()
trainDataloader, testDataloader = loader.load()

trainBatch, testBatch = next(iter(trainDataloader))



from helperFunction import printTime

printTime(startTime, "Model Completed Successfully")
