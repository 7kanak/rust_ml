# rust_ml


### bert-ft
tried to do inference on onnx loaded transformer model using burn crate. was not able to load the model (raised issue with burn) 


### bert-onnx-inference
##### loaded albert trasnformer model using ort, and was able to do inference on it
##### performed a simple test of rust vs python with 672 examples to do prediction in one batch rust seems around 50% faster than python, with 26 examples in test.txt file, minimum time running multiple times in rust was around 590sec, and in python it was 2.03 sec