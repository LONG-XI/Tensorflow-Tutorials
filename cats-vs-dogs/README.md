1. showing how to run. 
2. go to the train_and_val.py file and click run.
3. call run_training().

In [1]: runfile('C:/xilong/cat-dog/01 cats vs dogs/new_version/train_and_val.py', wdir='C:/xilong/cat-dog/01 cats vs dogs/new_version')

In [2]: run_training()
There are 12500 cats
There are 12500 dogs
WARNING:tensorflow:From C:\xilong\cat-dog\01 cats vs dogs\new_version\input_train_val_split.py:107: slice_input_producer (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(tuple(tensor_list)).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.
WARNING:tensorflow:From C:\Users\lxi\AppData\Local\Continuum\anaconda3\envs\python37\lib\site-packages\tensorflow\python\training\input.py:374: range_input_producer (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.range(limit).shuffle(limit).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.
WARNING:tensorflow:From C:\Users\lxi\AppData\Local\Continuum\anaconda3\envs\python37\lib\site-packages\tensorflow\python\training\input.py:320: input_producer (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(input_tensor).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.
WARNING:tensorflow:From C:\Users\lxi\AppData\Local\Continuum\anaconda3\envs\python37\lib\site-packages\tensorflow\python\training\input.py:190: limit_epochs (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensors(tensor).repeat(num_epochs)`.
WARNING:tensorflow:From C:\Users\lxi\AppData\Local\Continuum\anaconda3\envs\python37\lib\site-packages\tensorflow\python\training\input.py:199: QueueRunner.__init__ (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
WARNING:tensorflow:From C:\Users\lxi\AppData\Local\Continuum\anaconda3\envs\python37\lib\site-packages\tensorflow\python\training\input.py:199: add_queue_runner (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
WARNING:tensorflow:From C:\Users\lxi\AppData\Local\Continuum\anaconda3\envs\python37\lib\site-packages\tensorflow\python\training\input.py:202: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From C:\Users\lxi\AppData\Local\Continuum\anaconda3\envs\python37\lib\site-packages\tensorflow\python\ops\control_flow_ops.py:3632: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From C:\Users\lxi\AppData\Local\Continuum\anaconda3\envs\python37\lib\site-packages\tensorflow\python\ops\image_ops_impl.py:1241: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.
WARNING:tensorflow:From C:\xilong\cat-dog\01 cats vs dogs\new_version\input_train_val_split.py:127: batch (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.batch(batch_size)` (or `padded_batch(...)` if `dynamic_pad=True`).
WARNING:tensorflow:From C:/xilong/cat-dog/01 cats vs dogs/new_version/train_and_val.py:88: start_queue_runners (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
Step 0, train loss = 0.69, train accuracy = 50.00%
**  Step 0, val loss = 0.69, val accuracy = 51.56%  **
Step 50, train loss = 0.69, train accuracy = 53.12%
Step 100, train loss = 0.65, train accuracy = 64.06%
Step 150, train loss = 0.63, train accuracy = 68.75%
Step 200, train loss = 0.65, train accuracy = 60.94%
**  Step 200, val loss = 0.63, val accuracy = 62.50%  **
Step 250, train loss = 0.57, train accuracy = 79.69%
Step 300, train loss = 0.62, train accuracy = 65.62%
Step 350, train loss = 0.60, train accuracy = 68.75%
Step 400, train loss = 0.57, train accuracy = 65.62%
**  Step 400, val loss = 0.56, val accuracy = 71.88%  **
Step 450, train loss = 0.56, train accuracy = 75.00%
Step 500, train loss = 0.55, train accuracy = 76.56%
Step 550, train loss = 0.59, train accuracy = 70.31%
Step 600, train loss = 0.61, train accuracy = 70.31%
**  Step 600, val loss = 0.64, val accuracy = 62.50%  **
Step 650, train loss = 0.52, train accuracy = 68.75%
Step 700, train loss = 0.55, train accuracy = 70.31%
Step 750, train loss = 0.67, train accuracy = 57.81%
Step 800, train loss = 0.55, train accuracy = 73.44%
**  Step 800, val loss = 0.68, val accuracy = 50.00%  **
Step 850, train loss = 0.59, train accuracy = 65.62%
Step 900, train loss = 0.47, train accuracy = 79.69%
Step 950, train loss = 0.58, train accuracy = 71.88%
Step 1000, train loss = 0.61, train accuracy = 68.75%
**  Step 1000, val loss = 0.56, val accuracy = 68.75%  **
Step 1050, train loss = 0.58, train accuracy = 68.75%
Step 1100, train loss = 0.47, train accuracy = 78.12%
Step 1150, train loss = 0.57, train accuracy = 78.12%
Step 1200, train loss = 0.43, train accuracy = 82.81%
**  Step 1200, val loss = 0.59, val accuracy = 68.75%  **
Step 1250, train loss = 0.41, train accuracy = 82.81%
Step 1300, train loss = 0.51, train accuracy = 75.00%
Step 1350, train loss = 0.56, train accuracy = 68.75%
Step 1400, train loss = 0.38, train accuracy = 85.94%
**  Step 1400, val loss = 0.52, val accuracy = 71.88%  **
Step 1450, train loss = 0.43, train accuracy = 79.69%
Step 1500, train loss = 0.48, train accuracy = 75.00%
Step 1550, train loss = 0.55, train accuracy = 73.44%
Step 1600, train loss = 0.29, train accuracy = 89.06%
**  Step 1600, val loss = 0.56, val accuracy = 75.00%  **
Step 1650, train loss = 0.41, train accuracy = 78.12%
Step 1700, train loss = 0.41, train accuracy = 81.25%
Step 1750, train loss = 0.42, train accuracy = 76.56%
Step 1800, train loss = 0.44, train accuracy = 81.25%
**  Step 1800, val loss = 0.54, val accuracy = 75.00%  **
Step 1850, train loss = 0.43, train accuracy = 81.25%
Step 1900, train loss = 0.38, train accuracy = 82.81%
Step 1950, train loss = 0.41, train accuracy = 84.38%
Step 2000, train loss = 0.38, train accuracy = 84.38%
**  Step 2000, val loss = 0.46, val accuracy = 79.69%  **
Step 2050, train loss = 0.48, train accuracy = 75.00%
Step 2100, train loss = 0.34, train accuracy = 84.38%
Step 2150, train loss = 0.40, train accuracy = 85.94%
Step 2200, train loss = 0.37, train accuracy = 82.81%
**  Step 2200, val loss = 0.71, val accuracy = 62.50%  **
Step 2250, train loss = 0.33, train accuracy = 85.94%
Step 2300, train loss = 0.31, train accuracy = 89.06%
Step 2350, train loss = 0.30, train accuracy = 87.50%
Step 2400, train loss = 0.31, train accuracy = 84.38%
**  Step 2400, val loss = 0.47, val accuracy = 76.56%  **
Step 2450, train loss = 0.41, train accuracy = 76.56%
Step 2500, train loss = 0.33, train accuracy = 87.50%
Step 2550, train loss = 0.31, train accuracy = 85.94%
Step 2600, train loss = 0.40, train accuracy = 84.38%
**  Step 2600, val loss = 0.80, val accuracy = 62.50%  **
Step 2650, train loss = 0.36, train accuracy = 82.81%
Step 2700, train loss = 0.30, train accuracy = 87.50%
Step 2750, train loss = 0.27, train accuracy = 85.94%
Step 2800, train loss = 0.37, train accuracy = 87.50%
**  Step 2800, val loss = 0.65, val accuracy = 68.75%  **
Step 2850, train loss = 0.19, train accuracy = 92.19%
Step 2900, train loss = 0.30, train accuracy = 89.06%
Step 2950, train loss = 0.31, train accuracy = 85.94%
Step 3000, train loss = 0.37, train accuracy = 85.94%
**  Step 3000, val loss = 0.91, val accuracy = 65.62%  **
Step 3050, train loss = 0.35, train accuracy = 84.38%
Step 3100, train loss = 0.32, train accuracy = 87.50%
Step 3150, train loss = 0.17, train accuracy = 93.75%
Step 3200, train loss = 0.32, train accuracy = 87.50%
**  Step 3200, val loss = 0.67, val accuracy = 73.44%  **
Step 3250, train loss = 0.17, train accuracy = 96.88%
Step 3300, train loss = 0.25, train accuracy = 90.62%
Step 3350, train loss = 0.36, train accuracy = 79.69%
Step 3400, train loss = 0.21, train accuracy = 93.75%
**  Step 3400, val loss = 0.64, val accuracy = 81.25%  **
Step 3450, train loss = 0.13, train accuracy = 95.31%
Step 3500, train loss = 0.21, train accuracy = 87.50%
Step 3550, train loss = 0.11, train accuracy = 98.44%
Step 3600, train loss = 0.20, train accuracy = 93.75%
**  Step 3600, val loss = 0.86, val accuracy = 67.19%  **
Step 3650, train loss = 0.15, train accuracy = 93.75%
Step 3700, train loss = 0.15, train accuracy = 92.19%
Step 3750, train loss = 0.13, train accuracy = 95.31%
Step 3800, train loss = 0.13, train accuracy = 93.75%
**  Step 3800, val loss = 0.82, val accuracy = 70.31%  **
Step 3850, train loss = 0.06, train accuracy = 98.44%
Step 3900, train loss = 0.12, train accuracy = 95.31%
Step 3950, train loss = 0.13, train accuracy = 95.31%
Step 4000, train loss = 0.19, train accuracy = 93.75%
**  Step 4000, val loss = 1.05, val accuracy = 67.19%  **
Step 4050, train loss = 0.11, train accuracy = 95.31%
Step 4100, train loss = 0.08, train accuracy = 98.44%
Step 4150, train loss = 0.08, train accuracy = 98.44%
Step 4200, train loss = 0.06, train accuracy = 100.00%
**  Step 4200, val loss = 0.99, val accuracy = 70.31%  **
Step 4250, train loss = 0.13, train accuracy = 93.75%
Step 4300, train loss = 0.18, train accuracy = 95.31%
Step 4350, train loss = 0.09, train accuracy = 96.88%
Step 4400, train loss = 0.07, train accuracy = 98.44%
**  Step 4400, val loss = 0.98, val accuracy = 71.88%  **
Step 4450, train loss = 0.07, train accuracy = 98.44%
Step 4500, train loss = 0.05, train accuracy = 96.88%
Step 4550, train loss = 0.03, train accuracy = 100.00%
Step 4600, train loss = 0.07, train accuracy = 100.00%
**  Step 4600, val loss = 1.16, val accuracy = 68.75%  **
Step 4650, train loss = 0.05, train accuracy = 100.00%
Step 4700, train loss = 0.07, train accuracy = 98.44%
Step 4750, train loss = 0.03, train accuracy = 100.00%
Step 4800, train loss = 0.03, train accuracy = 100.00%
**  Step 4800, val loss = 0.86, val accuracy = 78.12%  **
Step 4850, train loss = 0.02, train accuracy = 100.00%
Step 4900, train loss = 0.05, train accuracy = 98.44%
Step 4950, train loss = 0.06, train accuracy = 95.31%
Step 5000, train loss = 0.03, train accuracy = 100.00%
**  Step 5000, val loss = 0.92, val accuracy = 76.56%  **
Step 5050, train loss = 0.02, train accuracy = 100.00%
Step 5100, train loss = 0.01, train accuracy = 100.00%
Step 5150, train loss = 0.04, train accuracy = 100.00%
Step 5200, train loss = 0.01, train accuracy = 100.00%
**  Step 5200, val loss = 0.81, val accuracy = 75.00%  **
Step 5250, train loss = 0.03, train accuracy = 100.00%
Step 5300, train loss = 0.03, train accuracy = 98.44%
Step 5350, train loss = 0.07, train accuracy = 98.44%
Step 5400, train loss = 0.04, train accuracy = 100.00%
**  Step 5400, val loss = 2.00, val accuracy = 57.81%  **
Step 5450, train loss = 0.04, train accuracy = 96.88%
Step 5500, train loss = 0.01, train accuracy = 100.00%
Step 5550, train loss = 0.02, train accuracy = 100.00%
Step 5600, train loss = 0.02, train accuracy = 100.00%
**  Step 5600, val loss = 0.64, val accuracy = 84.38%  **
Step 5650, train loss = 0.01, train accuracy = 100.00%
Step 5700, train loss = 0.02, train accuracy = 100.00%
Step 5750, train loss = 0.21, train accuracy = 90.62%
Step 5800, train loss = 0.10, train accuracy = 95.31%
**  Step 5800, val loss = 1.25, val accuracy = 68.75%  **
Step 5850, train loss = 0.02, train accuracy = 100.00%
Step 5900, train loss = 0.03, train accuracy = 98.44%
Step 5950, train loss = 0.02, train accuracy = 98.44%
**  Step 5999, val loss = 0.67, val accuracy = 84.38%  **

In [3]: 
