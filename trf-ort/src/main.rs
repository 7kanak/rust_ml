use tokenizers::tokenizer::{Tokenizer, pad_encodings, PaddingParams};
use ort::{inputs, self, CUDAExecutionProvider, Session, GraphOptimizationLevel};
use ndarray::{ArrayBase, Dim, OwnedRepr, ArrayViewD, Array2, Array};
use std::env::args;
use std::time::Instant;
use std::fs::File;
use std::io::{self, BufRead, BufReader};


fn main() -> ort::Result<()>{
    let start = Instant::now();
    tracing_subscriber::fmt::init();

    ort::init()
    .with_name("albert")
    .with_execution_providers([CUDAExecutionProvider::default().build()])
    .commit()?;

    let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level1)?
    .with_intra_threads(1)?
    .commit_from_file("src/model/albert2.onnx")?;

    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();

    let args: Vec<String> = args().skip(1).collect();
    let mut test_lines = load_test_from_file().unwrap();
    test_lines.extend(args);
    let example_count = test_lines.len();
    
    predict(test_lines, &tokenizer, &session).unwrap();

    let duration = start.elapsed();
    println!("Time elapsed for running {:?} examples : {:?}", example_count,duration);
    Ok(())
}


fn load_test_from_file() -> io::Result<Vec<String>>{
    let file = File::open("/home/kanak/Documents/dev/rust_ml/trf-ort/data/test.txt")?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
    Ok(lines)
}

fn create_simple_batch(texts: Vec<String>, tokenizer:&Tokenizer) -> ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>{
    let mut tokens = tokenizer.encode_batch(texts, true).unwrap();


    let pad_param = PaddingParams{
        pad_token: String::from("[PAD]"),
        .. PaddingParams::default()
    };

    pad_encodings(&mut tokens[..], &pad_param).unwrap();

    let row_count = tokens.len();
    let col_count = tokens[0].get_ids().len();

    let flat_ids: Vec<i64> = tokens.iter()
        .flat_map(|encoding| encoding.get_ids().iter().map(|&id| id as i64))
        .collect();
    
    let input_id_array: Array2<i64> = Array2::from_shape_vec((row_count, col_count), flat_ids).unwrap();

    input_id_array

}


fn predict(texts: Vec<String>, tokenizer:&Tokenizer, session: &Session) -> ort::Result<()>{
    let input_ids: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>> = create_simple_batch(texts, &tokenizer);
    let attention_mask: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>> = Array::ones(input_ids.raw_dim());
    let input_3: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>> = Array::zeros(input_ids.raw_dim());

    let outputs = session.run(inputs![input_ids,attention_mask,input_3]?)?;
    let ans: ArrayViewD<f32> = outputs["logits"].try_extract_tensor()?;
    println!("{:?}", ans);
    Ok(())
}