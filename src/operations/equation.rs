use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use crate::operations::*;
use std::thread;
use std::sync::Arc;
use std::sync::Mutex;


#[derive(Debug, Copy, Clone, Hash, Eq)]
pub struct VariableToken {
    name: u64,
    pub x_size: usize,
    pub y_size: usize
}

impl VariableToken {
    pub fn new(name: u64, x_size: usize, y_size: usize) -> VariableToken {
        VariableToken {
            name,
            x_size,
            y_size
        }
    }
}

#[derive(Copy, Clone)]
struct JobPtr {
    memory_ptr: * mut f32
}
unsafe impl Send for JobPtr {}
unsafe impl Sync for JobPtr {}

impl Ord for VariableToken {
    fn cmp(&self, other: &Self) -> Ordering {
        self.name.cmp(&other.name)
    }
}

impl PartialOrd for VariableToken {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for VariableToken {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Operator {
    Add,
    MatrixMul,
    ElementWiseMul,
    Map,
    Dif,
    Scalar,
    Conv
}

#[derive(Debug, Copy, Clone)]
pub struct MemoryToken {
    x_dim: usize,
    y_dim: usize,
    start: usize,
    size: usize
}

impl MemoryToken {
    pub fn new(x_dim: usize, y_dim: usize, start: usize) -> MemoryToken {
        MemoryToken {
            x_dim,
            y_dim,
            start,
            size: x_dim * y_dim
        }
    }
}

#[derive(Clone, Debug)]
pub struct Variable {
    x: usize,
    y: usize,
    name: u64,
    inputs: Vec<u64>,
    dependant_opertions: Vec<VariableToken>
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

    pub fn add_dependant_operation(&mut self, operation_name: VariableToken) {
        self.dependant_opertions.push(operation_name);
    }
}

#[derive(Debug, Clone)]
pub struct Operation {
    operator: Operator,
    satisfied_input: Vec<VariableToken>,
    inputs: HashSet<VariableToken>,
    inputs_in_order: Vec<VariableToken>,
    output_variable: VariableToken
}

impl Operation {
    pub fn new(operator: Operator, inputs: Vec<VariableToken>, output_variable: VariableToken) -> Operation {
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
            inputs_in_order: inputs.clone(),
            output_variable
        }
    }
}


pub struct Equation {
    variables: HashMap<VariableToken, Variable>,
    operations: HashMap<VariableToken, Operation>,
    memory_token: HashMap<VariableToken, MemoryToken>,
    mapping_functions: HashMap<VariableToken, Box<dyn Fn(f32) -> f32>>,
    variable_count: usize,
    memory: Vec<f32>,
}

impl Equation {
    pub fn new() -> Equation {
        Equation {
            variables: HashMap::new(),
            operations: HashMap::new(),
            memory_token: HashMap::new(),
            mapping_functions: HashMap::new(),
            variable_count: 0,
            memory: vec![],
        }
    }

    //Minerva is Col major lib
    pub fn new_variable(&mut self, y_size: usize, x_size: usize) -> VariableToken {
        //A name is a UID for any amount of data that gets either feed into, or is computed as part of the equation
        let name = self.get_new_name();
        let variable_token = VariableToken::new(name, x_size, y_size);
        //A variable is a node in the graph
        let variable = Variable::new(x_size, y_size, name);
        self.variables.insert(variable_token, variable);

        //This represents the actually values that make up the variable
        let memory_token = MemoryToken::new(x_size, y_size, self.memory.len());
        self.memory.append(&mut vec![0.0;x_size * y_size]);
        self.memory_token.insert(variable_token, memory_token);

        return variable_token;
    }

    pub fn transpose(&mut self, variable_token: VariableToken) -> VariableToken {
        let new_variable_token = self.new_variable(variable_token.x_size, variable_token.y_size);
        self.copy_and_transpose_variable(self.memory_token[&variable_token], self.memory_token[&new_variable_token]);
        return new_variable_token;
    }

    fn get_new_name(&mut self) -> u64 {
        let value_to_return = self.variable_count as u64;
        self.variable_count += 1;
        return value_to_return;
    }

    pub fn new_operation_in_graph(&mut self, operands: Vec<VariableToken>, operator: Operator) -> Result<VariableToken, &'static str> {
        let mut size= (0, 0);
        match operator {
            Operator::Add => {
                if operands.len() != 2 {
                    return Err("Incorrect number of operands for add operations, want 2");
                }

                let lhs = self.variables.get(&operands[0]).unwrap();
                let rhs = self.variables.get(&operands[1]).unwrap();
                if lhs.x != rhs.x || lhs.y != rhs.y {
                    return Err("You may only add matrices of the same size");
                }
                size.0 = lhs.x;
                size.1 = lhs.y;
            },
            Operator::ElementWiseMul => {
                if operands.len() != 2 {
                    return Err("Incorrect number of operands for Element Wise Mul opertions, want 2");
                }
                let lhs = self.variables.get(&operands[0]).unwrap();
                let rhs = self.variables.get(&operands[1]).unwrap();
                if lhs.x != rhs.x || lhs.y != rhs.y {
                    return Err("You may only Element Wise Mul matrices of the same size");
                }
                size.0 = lhs.x;
                size.1 = lhs.y;

            },
            Operator::Dif => {
                if operands.len() != 2 {
                    return Err("Incorrect number of operands for a diff operations, want 2");
                }

                let lhs = self.variables.get(&operands[0]).unwrap();
                let rhs = self.variables.get(&operands[1]).unwrap();
                if lhs.x != rhs.x || lhs.y != rhs.y {
                    return Err("You may only diff matrices of the same size");
                }
                size.0 = lhs.x;
                size.1 = lhs.y;
            },
            Operator::MatrixMul => {
                if operands.len() != 2 {
                    return Err("Incorrect number of operands for mul operations, want 2");
                }
                let lhs = self.variables.get(&operands[0]).unwrap();
                let rhs = self.variables.get(&operands[1]).unwrap();
                if lhs.x != rhs.y {
                    return Err("Incomptabile matrices");
                }
                size.0 = lhs.y;
                size.1 = rhs.x;
            },
            Operator::Map => {
                panic!("Dont use new_operation for a mapping opertion, use new_mapping_operation");
            },
            Operator::Scalar => {
                if operands.len() != 2 {
                    return Err("Incorrect number of operands for Element Wise Mul opertions, want 2");
                }

                let lhs = self.variables.get(&operands[0]).unwrap();
                let rhs = self.variables.get(&operands[1]).unwrap();
                if rhs.x != 1 || rhs.y != 1 {
                    return Err("The right hand side of a Scalar op must be a matrix of 1x1 size");
                }
                size.0 = lhs.x;
                size.1 = lhs.y;
            }
            Operator::Conv => {
                panic!("Dont use new_operation for a conv opertion, use new_conv_operations");
            }
        }
        let output_variable = self.new_variable(size.0, size.1);
        for operand in &operands {
            self.variables.get_mut(&operand).unwrap().add_dependant_operation(output_variable);
        }
        let operation = Operation::new(operator, operands, output_variable);
        self.operations.insert(output_variable, operation);
        return Ok(output_variable);
    }

    #[inline(always)]
    pub fn new_mapping_operation(&mut self, operand: VariableToken, function: Box<dyn Fn(f32) -> f32>) -> Result<VariableToken, &'static str> {
        //Copy the mapping function in


        let original_value = &self.variables[&operand].clone();
        //Create the memory for the output of the mapping
        let output_variable = self.new_variable(original_value.x, original_value.y);

        //Add it to the dependency graph
        self.variables.get_mut(&operand).unwrap().add_dependant_operation(output_variable);

        //lets create a mapping for the operation back to the mapping function
        self.mapping_functions.insert(output_variable, function);
        let operation = Operation::new(Operator::Map, vec![operand], output_variable);
        self.operations.insert(output_variable, operation);

        return Ok(output_variable);
    }

    #[inline(always)]
    pub fn new_conv_operation(&mut self, target_matrix: VariableToken, kernel: VariableToken, stride: u32, padding: bool) -> Result<VariableToken, &'static str> {
        let mut padding_amount = 0;
        if padding {
            padding_amount = 0;
        }
        //using this formula https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
        let x_output_size = ((target_matrix.x_size - kernel.x_size) + (2 * padding_amount)) + 1;
        let y_output_size = ((target_matrix.y_size - kernel.y_size) + (2 * padding_amount)) + 1;
        let output_variable = self.new_variable(y_output_size, x_output_size);
        let operation = Operation::new(Operator::Conv, vec![target_matrix, kernel], output_variable);
        self.variables.get_mut(&target_matrix).unwrap().add_dependant_operation(output_variable);
        self.variables.get_mut(&kernel).unwrap().add_dependant_operation(output_variable);
        self.operations.insert(output_variable, operation);
        return Ok(output_variable);
    }

    #[inline(always)]
    fn preform_add_operation(&mut self, inputs: Vec<MemoryToken>, output_variable: MemoryToken) {

        //TODO: Turn this value into a const, that has found by testing various sizes of matrices
        if inputs[0].x_dim * inputs[0].y_dim > 100000 {
            println!("Starting large add operation");   
            let cpus = num_cpus::get() - 1;//Give our computer some room to breath
            let number_of_operations = (output_variable.size * output_variable.size) / cpus;
            let number_of_operations_1 = (output_variable.size * output_variable.size) % cpus;
            let mut jobs = vec![];
            for cpu in 0..cpus {
                let chunk_start = cpu * number_of_operations;
                let left_hand_start = inputs[0].start + chunk_start;
                let right_hand_start = inputs[1].start + chunk_start;
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

            let vector_mut_ptr = JobPtr{memory_ptr: self.memory.as_mut_ptr()};
            let mut handles = vec![];
            for tj in jobs {
                //let memory = arcss.clone();
                let handle = thread::spawn(move||
                    unsafe {
                        let memory =  vector_mut_ptr.memory_ptr;
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
        
        }
        else {
            println!("Starting simple add operation");
            for i in 0..output_variable.size {
                self.memory[i + output_variable.start] = self.memory[i + inputs[0].start] + self.memory[i + inputs[1].start];
            }
        }
    }

    #[inline(always)]
    fn preform_dif_operation(&mut self, inputs: Vec<MemoryToken>, output_variable: MemoryToken) {
        println!("Starting add operation");
        for i in 0..output_variable.size {
            self.memory[i + output_variable.start] = self.memory[i + inputs[0].start] - self.memory[i + inputs[1].start];
        }
    }

    #[inline(always)]
    fn preform_element_wise_mul_operation(&mut self, inputs: Vec<MemoryToken>, output_variable: MemoryToken) {
        println!("Starting add operation");
        for i in 0..output_variable.size {
            self.memory[i + output_variable.start] = self.memory[i + inputs[0].start] * self.memory[i + inputs[1].start];
        }
    }

    #[inline(always)]
    fn preform_mul_operation(&mut self, inputs: Vec<MemoryToken>, output_token: MemoryToken, output_x: usize, output_y: usize, shared_z: usize) {
        //TODO: Get the inverse of the the second input, this would allow for reading it in a form that takes advantage of cache conference
        println!("Starting mul operation");
        for y in 0..output_y {
            for x in 0..output_x {
                let mut running_total = 0.0f32;
                for z in 0..shared_z {
                    let first_index = z  + (y * shared_z);
                    let second_index = x + (output_y * z);
                    running_total += self.memory[inputs[0].start + first_index] * self.memory[inputs[1].start + second_index];
                }
                let index = x + y * output_x;//All of these are flat buffers, so we need to calculate what the the final index
                self.memory[output_token.start + index] = running_total;
            }
        }
    }

    #[inline(always)]
    fn preform_scalar_mul_operation(&mut self, inputs: Vec<MemoryToken>, output_token: MemoryToken) {
        println!("Starting scalar mul operation");
        let scalar = self.memory[inputs[1].start];
        for i in 0..inputs[0].size {
            self.memory[output_token.start + i] = scalar * self.memory[inputs[0].start + i];
        }
    }

    #[inline(always)]
    fn preform_map_operation(&mut self, input: MemoryToken, output: MemoryToken, mapping_function: VariableToken) {
        println!("Starting map operation");
        for i in 0..output.size {
            self.memory[output.start + i] = (*self.mapping_functions[&mapping_function])(self.memory[input.start + i]);
        }
    }

    #[inline(always)]
    fn preform_conv_operation(&mut self, inputs: Vec<MemoryToken>, output: MemoryToken) {
        println!("Starting conv operation");
        let target_transform = inputs[0];
        let kernel = inputs[1];
        let x_kernel_difference = target_transform.x_dim - kernel.x_dim;

        for i in 0..output.size {
            let mut summed = 0.0f32;
            for k in 0..kernel.size {

                let x_shift = i % output.x_dim;
                let y_shift = i % output.y_dim;
                let memory_index = k + x_shift + (y_shift * target_transform.x_dim) + ((k / kernel.x_dim) * x_kernel_difference);
                summed += self.memory[kernel.start + k] * self.memory[target_transform.start + memory_index];
            }
            self.memory[output.start + i] = summed;
        }
    }

    #[inline(always)]
    fn topilogical_sort(&mut self) -> Vec<Operation> {
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
                    let mut modified_operations: Vec<VariableToken> = vec![];
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

    pub fn fill_in_inputs(&mut self, inputs: &mut HashMap<VariableToken, Vec<f32>>) {
        println!("Filling in inputs");
        for (k, v) in inputs.iter() {
            let memory_token = self.memory_token[k];
            for i in 0..memory_token.size {
                self.memory[memory_token.start + i] = v[i];
            }
        }
    }

    /*
    I would like for this to be called between finishing up your graph and calling exectue
    pub fn compile()
    */

    pub fn evaluate(&mut self, inputs: &mut HashMap<VariableToken, Vec<f32>>) {
        println!("Starting evaluation of function");
        self.fill_in_inputs(inputs);
        let mut keys : Vec<VariableToken> = inputs.keys().map(|x|*x).collect();
        keys.sort();

        for k in keys {
            let variable = &self.variables[&k];
            for dp in &variable.dependant_opertions {
                self.operations.get_mut(dp).unwrap().inputs.remove(&k);
                self.operations.get_mut(dp).unwrap().satisfied_input.push(k);
            }
        }

        let mut sorted_opertions : Vec<Operation>= self.topilogical_sort();
        sorted_opertions.reverse();

        while sorted_opertions.len() > 0 {
            let op = sorted_opertions.pop().unwrap();
            let mut variable_tokens = vec![];
            for variable in &op.inputs_in_order {
                variable_tokens.push(self.memory_token[&variable]);
            }

            match op.operator {
                Operator::Add => {
                    self.preform_add_operation(variable_tokens, self.memory_token[&op.output_variable]);
                },
                Operator::Dif => {
                    self.preform_dif_operation(variable_tokens, self.memory_token[&op.output_variable]);
                },
                Operator::ElementWiseMul => {
                    self.preform_element_wise_mul_operation(variable_tokens, self.memory_token[&op.output_variable]);
                },
                Operator::MatrixMul => {
                    let output_token = self.memory_token[&op.output_variable];
                    let y_dim = variable_tokens[0].x_dim;
                    self.preform_mul_operation(variable_tokens, output_token, output_token.x_dim, output_token.y_dim, y_dim);
                },
                Operator::Map => {
                    let output_token = self.memory_token[&op.output_variable];
                    self.preform_map_operation(variable_tokens[0], output_token, op.output_variable);

                },
                Operator::Scalar => {
                    self.preform_add_operation(variable_tokens, self.memory_token[&op.output_variable]);
                },
                Operator::Conv => {
                    self.preform_conv_operation(variable_tokens, self.memory_token[&op.output_variable]);
                }
            }
        }
    }

    pub fn copy_and_transpose_variable(&mut self, a: MemoryToken, b: MemoryToken) {
        for y in 0..a.y_dim {
            for x in 0..a.x_dim {
                let a_index = x + (y * b.x_dim);
                let b_index = y + (x * b.x_dim);
                self.memory[b_index + b.start] = self.memory[a_index + a.start];
            }
        }
    }

    pub fn set_variable(&mut self, variable_name: VariableToken, values: &Vec<f32>) {
        let token = self.memory_token[&variable_name];
        for i in 0..token.size {
            self.memory[token.start + i] = values[i];
        }
    }

    pub fn random_init_variable(&mut self, variable_name: VariableToken) {
        let mut rng = rand::thread_rng();
        let token = self.memory_token[&variable_name];
        for i in 0..token.size {
            self.memory[token.start + i] = 0.;
        }
    }

    pub fn get_variable(&self, variable_name: VariableToken) -> Vec<f32> {
        //TODO: make this not trully N operation I know we can do better
        let token = self.memory_token[&variable_name];
        let mut return_memory = vec![];
        for i in 0..token.size {
            return_memory.push(self.memory[token.start + i]);
        }
        return return_memory;

    }
}
