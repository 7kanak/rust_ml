use tokenizers::tokenizer::Tokenizer;
use ort::{inputs, self, CUDAExecutionProvider, Session, GraphOptimizationLevel};
use ndarray::{Array1, ArrayBase, Axis, Dim, OwnedRepr, ArrayViewD};

fn main()  -> ort::Result<()> {
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
    let tokens = tokenizer.encode("good", false).unwrap();

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
