use tokenizers::tokenizer::Tokenizer;
use ort::{inputs, self, CUDAExecutionProvider, Session, GraphOptimizationLevel};
use ndarray::{Array1, ArrayBase, Axis, Dim, OwnedRepr, ArrayViewD};
use std::env::args;
use std::time::Instant;
use std::fs::File;
use std::io::{self, BufRead, BufReader};


fn main() -> ort::Result<()>{
    let start = Instant::now();
    ort::init()
    .with_name("albert")
    .with_execution_providers([CUDAExecutionProvider::default().build()])
    .commit()?;

    let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level1)?
    .with_intra_threads(1)?
    .commit_from_file("src/model/albert2.onnx")?;

    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();


    tracing_subscriber::fmt::init();
    let args: Vec<String> = args().skip(1).collect();
    let mut test_lines = load_test_from_file().unwrap();
    test_lines.extend(args);
    let example_count = test_lines.len();

    for text in test_lines{
        infer(&session,&tokenizer, &text).unwrap();
    }
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



fn infer(session: &Session,tokenizer:&Tokenizer, text: &str)  -> ort::Result<()> {
    let tokens = tokenizer.encode(text, false).unwrap();
    let tokens = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();
    let tokens = Array1::from_iter(tokens.iter().cloned());
    let array = tokens.view().insert_axis(Axis(0));
    // println!("{:?}", array);

    let attention_mask: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>  = Array1::ones(tokens.len()).insert_axis(Axis(0));
    let input_3: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>  = Array1::zeros(tokens.len()).insert_axis(Axis(0));
    
    let outputs = session.run(inputs![array,attention_mask,input_3]?)?;
    let ans: ArrayViewD<f32> = outputs["logits"].try_extract_tensor()?;


    println!("{:?}", ans);
    Ok(())  
}
