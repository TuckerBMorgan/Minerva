use std::collections::{HashMap, HashSet};

#[derive(Debug, Copy, Clone)]
pub enum Operator {
    Add,
    Mul,
    Map
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

#[derive(Clone)]
pub struct Variable {
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
pub struct Operation {
    operator: Operator,
    satisfied_input: Vec<u64>,
    inputs: HashSet<u64>,
    inputs_in_order: Vec<u64>,
    output_variable: u64
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
            inputs_in_order: inputs.clone(),
            output_variable
        }
    }
}


pub struct Equation {
    variables: HashMap<u64, Variable>,
    operations: HashMap<u64, Operation>,
    memory_token: HashMap<u64, MemoryToken>,
    mapping_functions: HashMap<u64, Box<dyn Fn(f32) -> f32>>,
    variable_count: usize,
    memory: Vec<f32>
}

impl Equation {
    pub fn new() -> Equation {
        Equation {
            variables: HashMap::new(),
            operations: HashMap::new(),
            memory_token: HashMap::new(),
            mapping_functions: HashMap::new(),
            variable_count: 0,
            memory: vec![]
        }
    }

    //Minerva is Coloum major lib
    pub fn new_variable(&mut self, y_size: usize, x_size: usize) -> u64 {
        //A name is a UID for any amount of data that gets either feed into, or is computed as part of the equation
        let name = self.get_new_name();

        //A variable is a node in the graph
        let variable = Variable::new(x_size, y_size, name);
        self.variables.insert(name, variable);

        //This represents the actually values that make up the variable
        let memory_token = MemoryToken::new(x_size, y_size, self.memory.len());
        self.memory.append(&mut vec![0.0;x_size * y_size]);
        self.memory_token.insert(name, memory_token);
        return name;
    }

    fn get_new_name(&mut self) -> u64 {
        let value_to_return = self.variable_count as u64;
        self.variable_count += 1;
        return value_to_return;
    }

    pub fn new_operation_in_graph(&mut self, operands: Vec<u64>, operator: Operator) -> Result<u64, &'static str> {
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
            Operator::Mul => {
                if operands.len() != 2 {
                    return Err("Incorrect number of operands for mul operations, want 2");
                }
                let lhs = self.variables.get(&operands[0]).unwrap();
                let rhs = self.variables.get(&operands[1]).unwrap();
                if lhs.y != rhs.x {
                    return Err("Incomptabile matrices");
                }
                size.0 = lhs.y;
                size.1 = rhs.x;
            },
            Operator::Map => {
                if operands.len() != 2 {
                    return Err("Incorrect number of operands for map operations, want 1");
                }

                let lhs = self.variables.get(&operands[0]).unwrap();
                size.0 = lhs.x;
                size.1 = lhs.y;
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

    pub fn new_mapping_operation(&mut self, operand: u64, function: Box<dyn Fn(f32) -> f32>) -> Result<u64, &'static str> {
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

    fn preform_add_operation(&mut self, inputs: Vec<MemoryToken>, output_variable: MemoryToken) {
        println!("Starting add operation");
        for i in 0..output_variable.size {
            self.memory[i + output_variable.start] = self.memory[i + inputs[0].start] + self.memory[i + inputs[1].start];
        }
    }

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

    fn preform_map_operation(&mut self, input: MemoryToken, output: MemoryToken, mapping_function: u64) {
        for i in 0..output.size {
            self.memory[output.start + i] = (*self.mapping_functions[&mapping_function])(self.memory[input.start + i]);
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

    pub fn fill_in_inputs(&mut self, inputs: &mut HashMap<u64, Vec<f32>>) {
        for (k, v) in inputs.iter() {
            let memory_token = self.memory_token[k];
            for i in 0..memory_token.size {
                self.memory[memory_token.start + i] = v[i];
            }
        }
    }

    pub fn evaluate(&mut self, inputs: &mut HashMap<u64, Vec<f32>>) {

        println!("Starting evaluation of function");
        println!("Filling in inputs");
        self.fill_in_inputs(inputs);
        let mut keys : Vec<u64> = inputs.keys().map(|x|*x).collect();
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
                Operator::Mul => {
                    let output_token = self.memory_token[&op.output_variable];
                    let y_dim = variable_tokens[0].x_dim;
                    self.preform_mul_operation(variable_tokens, output_token, output_token.x_dim, output_token.y_dim, y_dim);
                },
                Operator::Map => {
                    let output_token = self.memory_token[&op.output_variable];
                    self.preform_map_operation(variable_tokens[0], output_token, op.output_variable);

                }
            }
        }
    }

    pub fn get_variable(&self, variable_name: u64) -> Vec<f32> {
        //TODO: make this not trully N operation I know we can do better
        let token = self.memory_token[&variable_name];
        let mut return_memory = vec![];
        for i in 0..token.size {
            return_memory.push(self.memory[token.start + i]);
        }
        return return_memory;

    }
}
