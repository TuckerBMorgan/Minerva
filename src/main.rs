
extern crate stopwatch;
extern crate num_cpus;
extern crate rayon;
use rayon::prelude::*;
use stopwatch::{Stopwatch};


#[derive(Copy, Clone)]
enum Operator {
//    Mul,
    Add
}

struct MatrixConfig {
    x_size: usize,
    y_size: usize
}

impl MatrixConfig {
    pub fn new(x_size: usize, y_size: usize) -> MatrixConfig {
        MatrixConfig {
            x_size,
            y_size
        }
    }
}

#[derive(Copy, Clone)]
struct MatrixToken {
    start: usize,
    size: usize,
    x_size: usize,
    y_size: usize
}

impl MatrixToken {
    pub fn new(start: usize,
        size: usize,
        x_size: usize,
        y_size: usize
    ) -> MatrixToken {
        MatrixToken {
            start,
            size,
            x_size,
            y_size
        }
    }
}

#[derive(Copy, Clone)]
struct Operation {
    lhs: MatrixToken,
    rhs: MatrixToken,
    operator: Operator
}

impl Operation {
    pub fn new(lhs: &MatrixToken, rhs: &MatrixToken, operator: Operator) -> Operation {
        Operation {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            operator
        }
    }
}

struct AddJob {
    lhs_location: usize,
    rhs_location: usize,
    destination_location: usize
}

impl AddJob {
    pub fn new(lhs_location: usize, rhs_location: usize, destination_location: usize) -> AddJob {
        AddJob {
            lhs_location,
            rhs_location,
            destination_location
        }
    }
}

struct Computation {
    memory: Vec<f32>,
    operations: Vec<Operation>
}

impl Computation {
    pub fn new() -> Computation {
        Computation {
            memory: vec![],
            operations: vec![]
        }
    }
    
    pub fn add_variable(&mut self, matrix: MatrixConfig) -> MatrixToken {
        let start = self.memory.len();
        let size = matrix.x_size * matrix.y_size;
        for i in 0..size {
            self.memory.push(0.0f32);
        }
        return MatrixToken::new(start, size, matrix.x_size, matrix.y_size);
    }

    pub fn add_operation(&mut self, lhs: &MatrixToken, rhs: &MatrixToken, operator: Operator) {
        let operation = Operation::new(&lhs, &rhs, operator);
        self.operations.push(operation);
    }

    pub fn fill_variable(&mut self, matrix: &MatrixToken, values: &Vec<f32>) -> Result<(), &'static str> {
        if matrix.size != values.len() {
            return Err("Matrix and fill values are not the same size");
        }
        let matrix_start = matrix.start;
        for (i, v) in values.iter().enumerate() {
            self.memory[matrix_start + i] = *v;
        }
        Ok(())
    }

    pub fn get_matrix(&self, matrix: &MatrixToken) -> Vec<f32> {
        return self.memory[matrix.start..(matrix.start + matrix.size)].to_vec();
    }

    pub fn evaluate(&mut self, variables: Vec<(&MatrixToken, Vec<f32>)>) -> Result<MatrixToken, &'static str>  {
        let mut output_matrix = variables[0].0.clone();
        
        //Fill the input variables with data
        for (mt, vals) in variables {
            self.fill_variable(mt, &vals);
        }

        let owned_operations = self.operations.clone();
        let mut add_jobs = vec![];
        for operation in owned_operations {
            match operation.operator {
                Operator::Add => {
                    //Check to make sure that the operation is valid
                    if operation.lhs.x_size != operation.lhs.x_size || operation.rhs.y_size != operation.rhs.y_size {
                        return Err("Unmatched matrix sizes");
                    }

                    //Allocate the destination memory
                    let result_operation = MatrixConfig::new(operation.lhs.x_size, operation.lhs.y_size);
                    let result_token = self.add_variable(result_operation);

                    //Build all of the job configs
                    for x in 0..operation.lhs.x_size {
                        for y in 0..operation.lhs.y_size {
                            let offset = y + x * operation.lhs.x_size;
                            let add_job = AddJob::new(operation.lhs.start + offset, operation.rhs.start + offset, result_token.start + offset);
                            add_jobs.push(add_job);
                        }
                    }

                    output_matrix = result_token.clone();

                }
            }
        }
        //Split the Work in CPUS number of jobs, easy,
        //giving out those jobs.... harde

        for aj in add_jobs {
            self.memory[aj.destination_location] = self.memory[aj.lhs_location] + self.memory[aj.rhs_location];
        }

        return Ok(output_matrix);
    }

}

fn speed_test(number_of_runs: usize) {
    let matrix_size = 10000;
    let mut total_time_taken = 0;
    for x in 0..number_of_runs {
        let matrix_a = MatrixConfig::new(matrix_size, matrix_size);
        let matrix_b = MatrixConfig::new(matrix_size, matrix_size);
        let mut computation = Computation::new();
        let matrix_token_a = computation.add_variable(matrix_a);
        let matrix_token_b = computation.add_variable(matrix_b);
        let mut data_a = vec![];
        let mut data_b = vec![];
        for x in 0..matrix_size {
            for y in 0..matrix_size {
                data_a.push((x + y) as f32);
                data_b.push((x + y) as f32);
            }
        }

        computation.add_operation(&matrix_token_a, &matrix_token_b, Operator::Add);
        let sw = Stopwatch::start_new();
        let result = computation.evaluate(vec![(&matrix_token_a, data_a), (&matrix_token_b, data_b)]).unwrap();
        total_time_taken += sw.elapsed_ms();
        println!("Finished with {} out of {} runs",x + 1, number_of_runs);
    }
    println!("Average time taken: {}ms", total_time_taken as f32 / number_of_runs as f32);
}

fn main() {
    speed_test(25);
    /*
    let matrix_size = 10000;
    let matrix_a = MatrixConfig::new(matrix_size, matrix_size);
    let matrix_b = MatrixConfig::new(matrix_size, matrix_size);
    let mut computation = Computation::new();
    let matrix_token_a = computation.add_variable(matrix_a);
    let matrix_token_b = computation.add_variable(matrix_b);
    let mut data_a = vec![];
    let mut data_b = vec![];
    for x in 0..matrix_size {
        for y in 0..matrix_size {
            data_a.push((x + y) as f32);
            data_b.push((x + y) as f32);
        }
    }

    computation.add_operation(&matrix_token_a, &matrix_token_b, Operator::Add);
    let sw = Stopwatch::start_new();
    let result = computation.evaluate(vec![(&matrix_token_a, data_a), (&matrix_token_b, data_b)]).unwrap();
    println!("Thing took {}ms", sw.elapsed_ms());
    //println!("{:?}", computation.get_matrix(&result));
    */
}