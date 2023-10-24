use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use crate::operations::*;
use std::thread;
use std::sync::mpsc::channel;
use stopwatch::{Stopwatch};
use log::info;
// JobType: Enum used for both laying out the total work
// done by a graph
// and a internprocess commnication medium
#[derive(Copy, Clone)]
pub enum JobType {
    Add(usize, usize, usize, usize),
    MatrixMul(MemoryToken, MemoryToken, MemoryToken, usize, usize),
    ElementWiseMul(usize, usize, usize, usize),
    Map(usize, usize, usize, fn(f32)->f32),
    Diff(usize, usize, usize, usize),
    Scalar(usize, usize, usize, usize),
    Copy(MemoryToken, MemoryToken),
    Conv,
    Fence,
    End
}

impl JobType {
    pub fn new_add_type(lhs_start: usize, rhs_start: usize, destination_start: usize, length: usize) -> JobType {
        return JobType::Add(lhs_start, rhs_start, destination_start, length);
    }

    pub fn new_element_wise_mul_type(lhs_start: usize, rhs_start: usize, destination_start: usize, length: usize) -> JobType {
        return JobType::ElementWiseMul(lhs_start, rhs_start, destination_start, length);
    }

    pub fn new_diff_type(lhs_start: usize, rhs_start: usize, destination_start: usize, length: usize) -> JobType {
        return JobType::Diff(lhs_start, rhs_start, destination_start, length);
    }

    pub fn new_matrix_mul_type(left_hand_side: MemoryToken, right_hand_side: MemoryToken, destination: MemoryToken, output_start: usize, output_end: usize) -> JobType {
        return JobType::MatrixMul(left_hand_side, right_hand_side, destination, output_start, output_end);
    }

    pub fn new_map_type(lhs_start: usize, destination_start: usize, length: usize, mapping_function: fn(f32) -> f32) -> JobType {
        return JobType::Map(lhs_start, destination_start, length, mapping_function);
    }

    pub fn new_scaler_type(lhs_start: usize, rhs_start: usize, destination_start: usize, length: usize) -> JobType {
        return JobType::Scalar(lhs_start, rhs_start, destination_start, length);
    }

    pub fn new_copy_type(from: MemoryToken, to: MemoryToken) -> JobType {
        return JobType::Copy(from, to);
    }
}

#[derive(Debug, Copy, Clone, Hash, Eq)]
pub struct VariableToken {
    name: u64,
    pub x_size: usize,
    pub y_size: usize,
    pub transposed: bool
}

impl VariableToken {
    pub fn new(name: u64, x_size: usize, y_size: usize) -> VariableToken {
        VariableToken {
            name,
            x_size,
            y_size,
            transposed: false
        }
    }
}

#[derive(Copy, Clone)]
pub struct JobPtr {
    pub memory_ptr: * mut f32
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
    Diff,
    Scalar,
    Conv,
    Copy
}

#[derive(Debug, Copy, Clone)]
pub struct MemoryToken {
    pub x_dim: usize,
    pub y_dim: usize,
    pub start: usize,
    pub size: usize,
    pub transposed: bool
}

impl MemoryToken {
    pub fn new(x_dim: usize, y_dim: usize, start: usize) -> MemoryToken {
        MemoryToken {
            x_dim,
            y_dim,
            start,
            size: x_dim * y_dim,
            transposed: false
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
    pub memory_token: HashMap<VariableToken, MemoryToken>,
    mapping_functions: HashMap<VariableToken, fn(f32)->f32>,
    has_been_compiled: bool,
    variable_count: usize,
    pub memory: Vec<f32>,
    jobs: Vec<JobType>,
    is_compiled: bool
}

impl Equation {
    pub fn new() -> Equation {
        Equation {
            variables: HashMap::new(),
            operations: HashMap::new(),
            memory_token: HashMap::new(),
            mapping_functions: HashMap::new(),
            variable_count: 0,
            has_been_compiled: false,
            memory: vec![],
            jobs: vec![],
            is_compiled: false
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

    //Minerva is Col major lib
    fn new_variable_transposed(&mut self, y_size: usize, x_size: usize) -> VariableToken {
        //A name is a UID for any amount of data that gets either feed into, or is computed as part of the equation
        let name = self.get_new_name();
        let mut variable_token = VariableToken::new(name, x_size, y_size);
        variable_token.transposed = true;
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
        let mut new_variable_token = self.new_variable_transposed(variable_token.x_size, variable_token.y_size);
        self.memory_token.get_mut(&new_variable_token).unwrap().transposed = true;
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
                size.0 = lhs.y;
                size.1 = lhs.x;
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
                size.0 = lhs.y;
                size.1 = lhs.x;

            },
            Operator::Diff => {
                if operands.len() != 2 {
                    return Err("Incorrect number of operands for a diff operations, want 2");
                }

                let lhs = self.variables.get(&operands[0]).unwrap();
                let rhs = self.variables.get(&operands[1]).unwrap();
                if lhs.x != rhs.x || lhs.y != rhs.y {
                    return Err("You may only diff matrices of the same size");
                }
                size.0 = lhs.y;
                size.1 = lhs.x;
            },
            Operator::MatrixMul => {
                if operands.len() != 2 {
                    return Err("Incorrect number of operands for mul operations, want 2");
                }
                let lhs = self.variables.get(&operands[0]).unwrap();
                let rhs = self.variables.get(&operands[1]).unwrap();
                if lhs.y != rhs.x {
                    return Err("Incomptabile matrices");
                }
                size. 1 = lhs.x;
                size.0 = rhs.y;
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
                size.0 = lhs.y;
                size.1 = lhs.x;
            },
            Operator::Conv => {
                panic!("Dont use new_operation for a conv opertion, use new_conv_operations");
            },
            Operator::Copy => {
                if operands.len() != 2 {
                    return Err("Incorrect number of operands for Copy opertions, want 2");
                }

                let lhs = self.variables.get(&operands[0]).unwrap();
                let rhs = self.variables.get(&operands[1]).unwrap();
                if lhs.x != rhs.x || lhs.y != rhs.y {
                    return Err("You may only copy matrices of the same size");
                }
                size.0 = lhs.y;
                size.1 = lhs.x;
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

//    #[inline(always)]
    pub fn new_mapping_operation(&mut self, operand: VariableToken, function: fn(f32)->f32) -> Result<VariableToken, &'static str> {
        //Copy the mapping function in
        let original_value = &self.variables[&operand].clone();
        //Create the memory for the output of the mapping
        let output_variable = self.new_variable(original_value.y, original_value.x);

        //Add it to the dependency graph
        self.variables.get_mut(&operand).unwrap().add_dependant_operation(output_variable);

        //lets create a mapping for the operation back to the mapping function
        self.mapping_functions.insert(output_variable, function);
        let operation = Operation::new(Operator::Map, vec![operand], output_variable);
        self.operations.insert(output_variable, operation);

        return Ok(output_variable);
    }

    #[inline(always)]
    pub fn new_conv_operation(&mut self, target_matrix: VariableToken, kernel: VariableToken, stride: usize, padding: bool) -> Result<VariableToken, &'static str> {
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


    /*
    #[inline(always)]
    fn preform_add_operation(&mut self, inputs: Vec<MemoryToken>, output_variable: MemoryToken) {
        if self.jank == true {
            let cpus = num_cpus::get() - 1;//Give our computer some room to breath
            let number_of_operations = output_variable.size / cpus;
            let number_of_operations_1 = output_variable.size % cpus;
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
                let job = AddJob::new(left_hand_start, right_hand_start, output_variable.start + number_of_operations * cpu, operations);
                jobs.push(job);
            }

            let vector_mut_ptr = JobPtr{memory_ptr: self.memory.as_mut_ptr()};
            let mut handles = vec![];
            for tj in jobs {
                let handle = thread::spawn(move||
                    unsafe {
                        
                        let memory =  vector_mut_ptr.memory_ptr;
                        for i in 0..tj.length {
                            let memory_offset = tj.destination_start + i;
                            let left_offset = tj.lhs_start + i;
                            let right_offset = tj.rhs_start + i;
                            let left_hand_value = *memory.offset(left_offset as isize);
                            let right_hand_value = *memory.offset(right_offset as isize);
                            *memory.offset(memory_offset as isize) = left_hand_value + right_hand_value;
                        }
                    }
                );
                handles.push(handle);
            }

            for h in handles {
                let _ = h.join();
            }
        
        }
        else {
            for i in 0..output_variable.size {
                self.memory[i + output_variable.start] = self.memory[i + inputs[0].start] + self.memory[i + inputs[1].start];
            }
        }
    }

    #[inline(always)]
    fn preform_dif_operation(&mut self, inputs: Vec<MemoryToken>, output_variable: MemoryToken) {
        for i in 0..output_variable.size {
            self.memory[i + output_variable.start] = self.memory[i + inputs[0].start] - self.memory[i + inputs[1].start];
        }
    }

    #[inline(always)]
    fn preform_element_wise_mul_operation(&mut self, inputs: Vec<MemoryToken>, output_variable: MemoryToken) {
        for i in 0..output_variable.size {
            self.memory[i + output_variable.start] = self.memory[i + inputs[0].start] * self.memory[i + inputs[1].start];
        }
    }

    #[inline(always)]
    fn preform_mul_operation(&mut self, inputs: Vec<MemoryToken>, output_token: MemoryToken, output_x: usize, output_y: usize, shared_z: usize) {
        //TODO: Get the inverse of the the second input, this would allow for reading it in a form that takes advantage of cache conference
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
        let scalar = self.memory[inputs[1].start];
        for i in 0..inputs[0].size {
            self.memory[output_token.start + i] = scalar * self.memory[inputs[0].start + i];
        }
    }

    #[inline(always)]
    fn preform_map_operation(&mut self, input: MemoryToken, output: MemoryToken, mapping_function: VariableToken) {
        for i in 0..output.size {
            self.memory[output.start + i] = (*self.mapping_functions[&mapping_function])(self.memory[input.start + i]);
        }
    }

    #[inline(always)]
    fn preform_conv_operation(&mut self, inputs: Vec<MemoryToken>, output: MemoryToken) {
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
    */

    //https://en.wikipedia.org/wiki/Topological_sorting
    #[inline(always)]
    fn topilogical_sort(&mut self) -> Vec<Operation> {
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
        let sw = Stopwatch::start_new();

        for (k, v) in inputs.iter() {
            let memory_token = self.memory_token[k];
            for i in 0..memory_token.size {
                self.memory[memory_token.start + i] = v[i];
            }
        }

        info!("fill in input took {:?}", sw.elapsed_ms());
    } 

    pub fn compile(&mut self)  {
        let sw = Stopwatch::start_new();
        let mut keys = vec![];
        for (k, v) in &self.variables {
            if self.operations.contains_key(k) == false {
                keys.push(*k);
            }
        }

        for k in keys {
            let variable = &self.variables[&k];
            for dp in &variable.dependant_opertions {
                self.operations.get_mut(dp).unwrap().inputs.remove(&k);
                self.operations.get_mut(dp).unwrap().satisfied_input.push(k);
            }
        }

        let mut sorted_opertions : Vec<Operation> = self.topilogical_sort();
        sorted_opertions.reverse();

        while sorted_opertions.len() > 0 {
            let op = sorted_opertions.pop().unwrap();
            let mut memory_tokens = vec![];
            for variable in &op.inputs_in_order {
                memory_tokens.push(self.memory_token[&variable]);
            }

            match op.operator {
                Operator::Add => {
                    self.jobs.append(&mut compile_add_operation(memory_tokens, self.memory_token[&op.output_variable]));
                },
                Operator::Diff => {
                    self.jobs.append(&mut compile_diff_operation(memory_tokens, self.memory_token[&op.output_variable]));
                }
                Operator::MatrixMul => {
                    self.jobs.append(&mut compile_matrix_mul_operation(memory_tokens, self.memory_token[&op.output_variable]));
                }
                Operator::Map => {
                    self.jobs.append(&mut compile_map_operation(memory_tokens, self.memory_token[&op.output_variable], self.mapping_functions[&op.output_variable]));
                },
                Operator::ElementWiseMul => {
                    self.jobs.append(&mut compile_element_wise_mul_job(memory_tokens, self.memory_token[&op.output_variable]));
                }
                Operator::Scalar => {
                    self.jobs.append(&mut compile_scalar_operation(memory_tokens, self.memory_token[&op.output_variable]));
                },
                Operator::Copy => {
                    self.jobs.append(&mut compile_copy_operation(memory_tokens[0], memory_tokens[1]));
                }
                _ => {

                }

            }
        }
        /*
        use crate::operations::*;

        pub fn compile_add_operation(inputs: Vec<MemoryToken>, output_variable: MemoryToken) -> Vec<JobType> {
            let cpus = (num_cpus::get() - 1) as usize;//Give our computer some room to breath
            let number_of_operations = output_variable.size / cpus;
            let number_of_operations_1 = output_variable.size % cpus;
            if output_variable.size * 10 > cpus {
                return vec![
                    JobType::new_add_type(inputs[0].start, inputs[1].start,  output_variable.start, output_variable.size)
                ];
            }
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
                let job = JobType::new_add_type(left_hand_start, right_hand_start,  output_variable.start + chunk_start, operations);
                jobs.push(job);
            }

            return jobs;
        }

        pub fn preform_add_job(lhs_start: usize,  rhs_start: usize, destination_start: usize, length: usize, memory_pointer: &mut JobPtr) {
            unsafe {
                let memory =  memory_pointer.memory_ptr;
                for i in 0..length {
                    let memory_offset = destination_start + i;
                    let left_offset = lhs_start + i;
                    let right_offset = rhs_start + i;
                    let left_hand_value = *memory.offset(left_offset as isize);
                    let right_hand_value = *memory.offset(right_offset as isize);
                    *memory.offset(memory_offset as isize) = left_hand_value + right_hand_value;
                }
            }
        }
 */
        self.has_been_compiled = true;
        println!("Finished Compiling Graph");
        info!("Compiling took {:?}", sw.elapsed_ms());
    }

    pub fn evaluate(&mut self, inputs: &mut HashMap<VariableToken, Vec<f32>>) {
        assert!(self.has_been_compiled);
        self.fill_in_inputs(inputs);
        let sw = Stopwatch::start_new();
        let vector_mut_ptr = JobPtr{memory_ptr: self.memory.as_mut_ptr()};
        let number_of_worker_thread = num_cpus::get() - 1;//Maybe make this a config?
        let mut senders = vec![];
        let mut receivers = vec![];
        let mut handles = vec![];

        for i in 0..number_of_worker_thread {
            let (tx_to_worker, rx_from_equation) = channel();
            let (tx_to_equation, rx_from_worker) = channel();
            senders.push(tx_to_worker);
            receivers.push(rx_from_worker);
            let handle = thread::spawn(move||
                worker_thread(i, vector_mut_ptr, tx_to_equation, rx_from_equation)
            );
            handles.push(handle);
        }

        let mut current_job_index = 0;
        while current_job_index < self.jobs.len() {
            for i in 0..number_of_worker_thread {
                let wants_work = receivers[i].try_recv();
                match wants_work {
                    Ok(job_poke) => {
                        match job_poke {
                            JobPoke::Done(thread_number) => {
                                let _ = senders[thread_number].send(self.jobs[current_job_index]);
                                current_job_index += 1;
                                if current_job_index == self.jobs.len() {
                                    break;
                                }
                            }
                        }
                    },
                    Err(_) => {

                    }
                }
            }
        }

        for i in 0..number_of_worker_thread {
            let _ = senders[i].send(JobType::End);
        }

        for h in handles {
            let _ = h.join();
        }
        info!("Evaluation took {:?}", sw.elapsed_ms());
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
        //TODO: Allow the user to provide a function for how they want to init their
        //variable like, LeCun init
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
