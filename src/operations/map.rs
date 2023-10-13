use crate::operations::*;

pub fn compile_map_operation(inputs: Vec<MemoryToken>, output_variable: MemoryToken, mappping_function: fn(f32)->f32) -> Vec<JobType> {
    let cpus = (num_cpus::get() - 1) as usize;//Give our computer some room to breath
    let number_of_operations = output_variable.size / cpus;
    let number_of_operations_1 = output_variable.size % cpus;
    if output_variable.size * 10 > cpus {
        return vec![
            JobType::new_map_type(inputs[0].start,output_variable.start, output_variable.size, mappping_function)
        ];
    }
    let mut jobs = vec![];
    for cpu in 0..cpus {
        let chunk_start = cpu * number_of_operations;
        let left_hand_start = inputs[0].start + chunk_start;
        let operations;
        if cpu == cpus - 1 {
            operations = number_of_operations + number_of_operations_1;
        }
        else {
            operations = number_of_operations;
        }
        let job = JobType::new_map_type(left_hand_start, output_variable.start + chunk_start, operations, mappping_function);
        jobs.push(job);
    }

    return jobs;
}

pub fn preform_map_operation(lhs_start: usize, destination_start: usize, length: usize, mapping_function: fn(f32) -> f32, memory_pointer: &mut JobPtr) {
    unsafe {
        let memory =  memory_pointer.memory_ptr;
        for i in 0..length {
            let memory_offset = destination_start + i;
            let left_offset = lhs_start + i;
            let left_hand_value = *memory.offset(left_offset as isize);
            *memory.offset(memory_offset as isize) = (mapping_function)(left_hand_value);
        }
    }
}