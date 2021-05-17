extern crate stopwatch;


use stopwatch::{Stopwatch};

use std::{hash::Hash, sync::Arc};
use std::thread;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Copy, Clone)]
enum Operator {
    Add,
    Mul
}

struct AddJob {
    lhs_start: usize,
    rhs_start: usize,
    destination_start: usize,
    length: usize
}

impl AddJob {
    pub fn new(lhs_start: usize, rhs_start: usize, destination_start: usize, length: usize) -> AddJob {
        AddJob {
            lhs_start,
            rhs_start,
            destination_start,
            length
        }
    }
}

struct MulJob {
    destination_start: usize,
    matrix_size: usize,
    number_of_operations: usize,
    lhs_start: usize,
    rhs_start: usize

}

impl MulJob {
    pub fn new(destination_start: usize, matrix_size: usize, number_of_operations: usize, lhs_start: usize, rhs_start: usize) -> MulJob {
        MulJob {
            destination_start,
            matrix_size,
            number_of_operations,
            lhs_start,
            rhs_start
        }
    }
}

fn imperative_mul(matrix_size: usize) -> i64 {
    let mut vectors = vec![];
    for _ in 0..3 {
        let mut new_one = vec![];
        for _ in 0..matrix_size {
            for x in 0..matrix_size {
                new_one.push(x);
            }
        }
        vectors.push(new_one);
    }

    let sw = Stopwatch::start_new();

    for y in 0..matrix_size {
        for x in 0..matrix_size {
            let mut running_total = 0;
            for i in 0..matrix_size {
                let left_side_offset = i + x * matrix_size;
                let right_hand_offset = y + i * matrix_size;
                running_total += vectors[1][left_side_offset] * vectors[2][right_hand_offset];
            }
            let offset = x + y * matrix_size;
            vectors[0][offset] = running_total;
        }
    }

    return sw.elapsed_ms();
}

fn bench_imperative_mul(matrix_size: usize) -> f32 {
    let mut total_results = 0;

    for i in 0..25 {
        let result = imperative_mul(matrix_size);
        total_results += result;
    }
    return total_results as f32 / 25.;
}

fn jank_mul(matrix_size: usize) -> i64 {
    let total_matrix_size = matrix_size * matrix_size;
    let mut memory_buf = vec![];
    for _ in 0..3 {
        for _ in 0..matrix_size {
            for x in 0..matrix_size {
                memory_buf.push(x as f32);
            }
        }
    }
    let sw = Stopwatch::start_new();
    let arcss = Arc::new(memory_buf);
    let mut jobs = vec![];
    let cpus = num_cpus::get() - 1;//Give our computer some room to breath

    //allocate this work for the number of cpus
    let number_of_rows = total_matrix_size / cpus;
    let number_of_rows_remainder = (total_matrix_size) % cpus;
    for cpu in 0..cpus {
        let left_hand_start = total_matrix_size;
        let right_hand_start = total_matrix_size * 2;
        let operations;
        if cpu == cpus - 1 {
            operations = number_of_rows + number_of_rows_remainder;
        }
        else {
            operations = number_of_rows;
        }
        let index_start = number_of_rows * cpu;
        let job = MulJob::new(index_start, matrix_size, operations, left_hand_start, right_hand_start);
        jobs.push(job);
    }
    let sw = Stopwatch::start_new();
    let mut handles = vec![];
    for job in jobs {
        let memory = arcss.clone();
        let handle = thread::spawn(move ||{
            let memory = memory.as_ptr() as *mut f32;

            //For each slot in the outpuit matrix
            for location in 0..job.number_of_operations {
                let output_location = location + job.destination_start;
                //println!("location {}", output_location);
                let mut total_result = 0.;
                let output_x = output_location % job.matrix_size;
                let output_y = output_location / job.matrix_size;

                for i in 0..job.matrix_size {
                    let left_side_offset = (i + (output_y * job.matrix_size)) + job.lhs_start;//We want to walk along the coloums of the the
                    //Row defined by ouput_y * job.matrix_size

                    let right_hand_offset = ((i * job.matrix_size) + output_x) + job.rhs_start;//We want to walk along the row refined by
                    //i * job.matrix_size, down the coloum defined by output_x
                    //Spooky pointer math
                    unsafe{
                        total_result += *memory.offset(left_side_offset as isize) * (*memory.offset(right_hand_offset as isize));
                    }
                }
                //Spooky pointer math
                unsafe {
                    *memory.offset(output_location as isize) = total_result;
                }
            }
        });
        handles.push(handle);
    }
    for handle in handles {
        let result = handle.join();
        match result {
            Err(e) => {
                panic!("Error in Jank Mul {:?}", e);
            },
            Ok(_) => {

            }
        }
    }
    return sw.elapsed_ms();
}

fn bench_jank_mul(matrix_size: usize) -> f32 {
    let mut total_results = 0;

    for i in 0..25 {
        let result = jank_mul(matrix_size);
        total_results += result;
    }
    return total_results as f32 / 25.;
}


fn imperative_add(matrix_size: usize) -> i64 {
    let mut vectors = vec![];
    for i in 0..3 {
        let mut new_one = vec![];
        for y in 0..matrix_size {
            for x in 0..matrix_size {
                new_one.push(x);
            }
        }
        vectors.push(new_one);
    }

    let sw = Stopwatch::start_new();
    for y in 0..matrix_size {
        for x in 0..matrix_size {
            let offset = x + y * matrix_size;
            vectors[0][offset] = vectors[1][offset] + vectors[2][offset];
        }
    }
    return sw.elapsed_ms();
}

pub fn bench_imperative_add(matrix_size: usize) -> f32 {
    let mut total_results = 0;

    for i in 0..25 {
        let result = imperative_add(matrix_size);
        total_results += result;
    }
    return total_results as f32 / 25.;
}

fn jank_add(matrix_size: usize) -> i64 {
    let total_matrix_size = matrix_size * matrix_size;
    let mut memory_buf = vec![];
    for _ in 0..3 {
        for y in 0..matrix_size {
            for x in 0..matrix_size {
                memory_buf.push(0.0f32);
            }
        }
    }
    let sw = Stopwatch::start_new();
    let arcss = Arc::new(memory_buf);
    let mut jobs = vec![];
    let cpus = num_cpus::get() - 1;//Give our computer some room to breath

    //allocate this work for the number of cpus
    let number_of_operations = (matrix_size * matrix_size) / cpus;
    let number_of_operations_1 = (matrix_size * matrix_size) % cpus;
    for cpu in 0..cpus {
        let left_hand_start = total_matrix_size + cpu * number_of_operations;
        let right_hand_start = (total_matrix_size + total_matrix_size) + cpu * number_of_operations;
        let operations;
        if cpu == cpus - 1 {
            operations = number_of_operations + number_of_operations_1;
        }
        else {
            operations = number_of_operations;
        }
        let job = AddJob::new(left_hand_start, right_hand_start, number_of_operations * cpu, operations);
        jobs.push(job);
    }



    let mut handles = vec![];
    for tj in jobs {
        let memory = arcss.clone();
        let handle = thread::spawn(move||
            unsafe {
                let memory = memory.as_ptr() as *mut f32;
                for i in 0..tj.length {
                    let memory_offset = tj.destination_start + i;
                    let left_offset = tj.lhs_start + i;
                    let right_offset = tj.rhs_start + i;
                    //println!("{} {} {}", memory_offset, left_offset, right_offset);
                    *memory.offset(memory_offset as isize) = *memory.offset(left_offset as isize) + *memory.offset(right_offset as isize);
                }
            }
        );
        handles.push(handle);
    }

    for handle in handles {
        let result = handle.join();
        match result {
            Err(e) => {
                panic!("Error in jank add {:?}", e);
            },
            Ok(_) => {

            }
        }
    }

    return sw.elapsed_ms();
}

fn bench_jank_add(matrix_size: usize) -> f32 {
    println!("Benching Jank Add");
    let total_matrix_size = matrix_size * matrix_size;
    let mut total_results = 0;

    for i in 0..25 {
        let result = jank_add(matrix_size);
        total_results += result;
    }
    return total_results as f32 / 25.;
}

fn bench_add(matrix_size: usize) {
    let matrix_size = matrix_size;
    //println!("Starting Add Benchmark");
    let imperative_result = bench_imperative_add(matrix_size);
    println!("{},", imperative_result);
    //let jank_result = bench_jank_add(matrix_size);
    //println!("Time taken for imperative Add {}, jank Add {} for matrix of size {}", imperative_result, jank_result, matrix_size);
}

fn bench_mul(matrix_size: usize) {
    //Imperative has some trouble with larger sizes, so we reduce it by a factor of 10
    let mut matrix_size = matrix_size / 10;
    if matrix_size == 0  {
        matrix_size = 1;
    }
    let imperative_result = bench_imperative_mul(matrix_size);
    let jank_result = bench_jank_mul(matrix_size);
    println!("Time taken for imperative mul {}, jank mul {} for matrix of size {}", imperative_result, jank_result, matrix_size);
}

#[derive(Clone)]
struct Variable {
    x: usize,
    y: usize,
    name: u64,
    inputs: Vec<u64>,
    dependant_opertions: Vec<u64>
}

impl Variable {
    pub fn new(x: usize, y: usize, name: u64) -> Variable {
        Variable {
            x,
            y,
            name,
            inputs: vec![],
            dependant_opertions:  vec![]
        }
    }

    pub fn add_dependant_operation(&mut self, operation_name: u64) {
        self.dependant_opertions.push(operation_name);
    }
}

#[derive(Debug, Clone)]
struct Operation {
    operator: Operator,
    satisfied_input: Vec<u64>,
    inputs: HashSet<u64>,
    output_variable: u64,
}

impl Operation {
    pub fn new(operator: Operator, inputs: Vec<u64>, output_variable: u64) -> Operation {
        let mut changed_inputs = HashSet::new();
        inputs.iter().for_each(
                |x|{
                    changed_inputs.insert(*x);
        }
    );

        Operation {
            operator,
            satisfied_input: Vec::new(),
            inputs: changed_inputs,
            output_variable
        }
    }
}


struct Equation {
    variables: HashMap<u64, Variable>,
    operations: HashMap<u64, Operation>,
    variable_count: usize
}

impl Equation {
    pub fn new() -> Equation {
        Equation {
            variables: HashMap::new(),
            operations: HashMap::new(),
            variable_count: 0
        }
    }

    pub fn new_variable(&mut self, x_size: usize, y_size: usize) -> u64 {
        let name = self.variable_count as u64;
        let variable = Variable::new(x_size, y_size, name);
        self.variables.insert(name, variable);
        self.variable_count += 1;
        return name;
    }

    pub fn new_operation_in_graph(&mut self, lhs: u64, rhs: u64, operator: Operator) -> u64 {
        let mut size= (0, 0);
        match operator {
            Operator::Add => {
                let lhs = self.variables.get(&lhs).unwrap();
                size.0 = lhs.x;
                size.1 = lhs.y;
            },
            Operator::Mul => {
                let lhs = self.variables.get(&lhs).unwrap();
                let rhs = self.variables.get(&rhs).unwrap();
                size.0 = lhs.y;
                size.1 = rhs.x;
            }
        }
        let output_variable = self.new_variable(size.0, size.1);

        let inputs = vec![lhs, rhs];
        let operation = Operation::new(operator, inputs, output_variable);
        self.operations.insert(output_variable, operation);
        {
            self.variables.get_mut(&lhs).unwrap().add_dependant_operation(output_variable);
        }
        {
            self.variables.get_mut(&rhs).unwrap().add_dependant_operation(output_variable);
        }
        return output_variable;
    }

    pub fn preform_add_operation(&mut self, inputs: &Vec<&Vec<f32>>, output_variable: &mut Vec<f32>) {
        println!("Starting add operation");
        for i in 0..output_variable.len() {
            output_variable[i] = inputs[0][i] + inputs[1][i];
        }
    }

    pub fn preform_mul_operation(&mut self, inputs: &Vec<&Vec<f32>>, output_variable: &mut Vec<f32>, output_x: usize, output_y: usize, shared_z: usize) {
        //TODO: Get the inverse of the the second input, this would allow for reading it in a form that takes advantage of cache conference
        println!("Starting mul operation");
        for y in 0..output_y {
            for x in 0..output_x {

                let mut running_total = 0.0;
                for z in 0..shared_z {
                    let a_index = z + (shared_z * x);
                    let b_index = z + (shared_z * y);
                    running_total += inputs[0][a_index] * inputs[1][b_index];
                }
                let index = x + y * output_y;//All of these are flat buffers, so we need to calculate what the the final index
                output_variable[index] = running_total;
            }
        }
    }

    pub fn topilogical_sort(&mut self) -> Vec<Operation> {
        println!("Starting Top sort");
        let mut l = vec![];
        let mut s = vec![];

        let mut operations = self.operations.clone();
        let variables = self.variables.clone();


        let mut keys_to_remove = vec![];
        for (k, v) in &operations {
            if v.inputs.len() == 0 {
                keys_to_remove.push(k.clone());
            }
        }

        for key in keys_to_remove {
            s.push(operations.remove(&key).unwrap());
        }
        s.reverse();
        while s.len() > 0 {
            let first:Option<Operation> = s.pop();
            match first {
                Some(operation) => {
                    let operation = operation;
                    let variable: &Variable = &variables[&operation.output_variable];
                    let mut modified_operations: Vec<u64> = vec![];
                    for dependant_opertions in &variable.dependant_opertions {

                        let dep_op = operations.get_mut(&dependant_opertions).unwrap();
                        dep_op.inputs.remove(&operation.output_variable);
                        dep_op.satisfied_input.push(operation.output_variable);
                        modified_operations.push(dep_op.output_variable);
                    }

                    l.push(operation);
                    for id in modified_operations {
                        if operations[&id].inputs.len() == 0 {
                            s.push(operations.remove(&id).unwrap());
                        }
                    }
                },
                None => {
                    break;
                }
            }
        }

        return l;
    }

    pub fn evaluate(&mut self, inputs: &mut HashMap<u64, Vec<f32>>) -> Vec<Vec<f32>> {

        println!("Starting evaluation of function");
        let mut final_output = vec![];
        for (k, v) in inputs.iter() {
            let variable = &self.variables[k];
            for dp in &variable.dependant_opertions {
                self.operations.get_mut(dp).unwrap().inputs.remove(k);
                self.operations.get_mut(dp).unwrap().satisfied_input.push(*k);
            }
        }

        let mut sorted_opertions : Vec<Operation>= self.topilogical_sort();
        sorted_opertions.reverse();
        //Ok we should now know the size of everything that we need
        //Allocate memory of that size
        //Generate variable tokens
        //and use those as the way to start our tokens

        while sorted_opertions.len() > 0 {
            let op = sorted_opertions.pop().unwrap();
            let mut operation_inputs = vec![];
            let mut variable_sizes = vec![];
            for variable in &op.satisfied_input {
                let data_copy = &inputs[variable];
                variable_sizes.push((self.variables[&variable].x, self.variables[&variable].y));
                operation_inputs.push(data_copy);
            }

            match op.operator {
                Operator::Add => {
                    let mut output_memory: Vec<f32> = vec![0.0f32; variable_sizes[0].0 * variable_sizes[0].1];
                    self.preform_add_operation(&operation_inputs, &mut output_memory);
                    inputs.insert(op.output_variable, output_memory.clone());
                    final_output = output_memory;
                },
                Operator::Mul => {
                    let mut output_memory: Vec<f32> = vec![0.0f32; variable_sizes[0].0 * variable_sizes[1].1];
                    self.preform_mul_operation(&operation_inputs, &mut output_memory, variable_sizes[0].0,  variable_sizes[1].1, variable_sizes[0].1);
                    inputs.insert(op.output_variable, output_memory.clone());
                    final_output = output_memory;
                }
            }
        }
        return vec![final_output];
    }



}

fn main() {
    // y = (((a + b) + d) * f) * i);
    let mut equation = Equation::new();

    let a = equation.new_variable(2, 2);
    let b = equation.new_variable(2, 2);
    let c = equation.new_operation_in_graph(a, b, Operator::Add);

    let d = equation.new_variable(2, 2);
    let e  = equation.new_operation_in_graph(d, c, Operator::Add);

    let f = equation.new_variable(2, 2);
    let g = equation.new_operation_in_graph(e, f, Operator::Mul);

    let i = equation.new_variable(2, 2);
    let _y = equation.new_operation_in_graph(g, i, Operator::Mul);

    let mut inputs = HashMap::new();
    inputs.insert(a, vec![1.0, 1.0, 1.0, 1.0]);
    inputs.insert(b, vec![1.0, 2.0, 1.0, 1.0]);
    inputs.insert(d, vec![10.0, 2.0, 10.0, 2.0]);
    inputs.insert(f, vec![10.0, 10.0, 10.0, 10.0]);
    inputs.insert(i, vec![2.0, 2.0, 2.0, 2.0]);

    let result = equation.evaluate(&mut inputs);
    print!("{:?}", result)
}