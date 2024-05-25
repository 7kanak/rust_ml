mod model;

#[warn(unused_imports)]
use burn::tensor;
use burn_ndarray::{NdArray, NdArrayDevice};
use model::mnist::Model;
#[warn(unused_imports)]
use tokenizers::tokenizer::{Tokenizer, PaddingParams, pad_encodings};


#[warn(unused_variables)]
fn main() {
    // let tkn = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    // let tokens = tkn.encode("the movie was awesome", true).unwrap();
    // let encodings = vec![tokens];
    // let mut tokens_list: Vec<Vec<usize>> = Vec::with_capacity(encodings.len());

    // for item in encodings{
    //     tokens_list.push(self.tokenizer.encode(&item.text));
    // }

    // let token_ids: Vec<usize> = tokens.get_ids().iter().map(|t| *t as usize).collect();

    // let custom_padding_param = PaddingParams {
    //     pad_token : String::from("<PAD>"),
    //     ..PaddingParams::default()
    // };

    // let padded_enc = pad_encodings(&mut encodings, &custom_padding_param);

    // let mask = generate_padding_mask(
    //     self.tokenizer.pad_token(),
    //     tokens_list,
    //     Some(self.max_seq_length),
    //     &B::Device::default(),
    // );
    // println!("{:?}", token_ids);

    // Initialize a new model instance
    let device = NdArrayDevice::default();
    let model: Model<NdArray<f32>> = Model::new(&device);

    // Create a sample input tensor (zeros for demonstration)
    // let input = tensor::Tensor::<NdArray<f32>, 4>::zeros([1, 1, 28, 28], &device);

    // // Perform inference
    // let output = model.forward(input);

    // Print the output
    println!("{:?}", model);
}