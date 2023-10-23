use crate::operations::*;
use log::info;

pub fn compile_matrix_mul_operation(inputs: Vec<MemoryToken>, output_variable: MemoryToken) -> Vec<JobType> {
    let cpus = (num_cpus::get() - 1) as usize;//Give our computer some room to breath
    let mut jobs = vec![];
    if output_variable.y_dim > cpus {
        let mul_job = JobType::new_matrix_mul_type(inputs[0], inputs[1], output_variable, 0, output_variable.size);
        jobs.push(mul_job);
    }
    else {
        let number_of_rows_per_cpu = output_variable.size / cpus;
        for cpu in 0..cpus {
            let output_y_start = cpu * number_of_rows_per_cpu;
            let mut end_point = output_y_start + number_of_rows_per_cpu;
            if cpu == cpus - 1 {
                end_point = output_variable.y_dim;
            }
            let mul_job = JobType::new_matrix_mul_type(inputs[0], inputs[1], output_variable, output_y_start, end_point);
            jobs.push(mul_job);
        }        

    }

    return jobs;
}

pub fn preform_matrix_multiplication_job(left_hand_side: MemoryToken, right_hand_side: MemoryToken, destination: MemoryToken, output_start: usize, ouput_end: usize,memory_pointer:  &mut JobPtr) {

    unsafe {
        let memory =  memory_pointer.memory_ptr;

        for y in output_start..ouput_end {
            for x in 0..destination.x_dim {
                let mut running_total = 0.0f32;
                for z in 0..left_hand_side.x_dim {
                    let left_side_index;
                    if left_hand_side.transposed {
                        left_side_index = z * left_hand_side.y_dim + y;
                    }
                    else {
                        left_side_index = z + (y * left_hand_side.x_dim);
                    }

                    let right_side_index;
                    if right_hand_side.transposed {
                        right_side_index = x * right_hand_side.y_dim + z;
                    }
                    else {
                        right_side_index = x + (right_hand_side.x_dim * z);
                    }
                    let left_offset = left_hand_side.start + left_side_index;
                    let right_offset = right_hand_side.start + right_side_index;

                    let left_hand_value = *memory.offset(left_offset as isize);
                    let right_hand_value = *memory.offset(right_offset as isize);

                    running_total += left_hand_value * right_hand_value;
                }

                let index = x + y * destination.x_dim;//All of these are flat buffers, so we need to calculate what the the final index
                *memory.offset((destination.start + index) as isize) = running_total;
            }
        }
    }
}

pub fn preform_matrix_mul_job( lhs_start: usize, rhs_start: usize, destination_start: usize, output_x: usize, output_y: usize, output_y_start: usize, output_y_end: usize,shared_z: usize, memory_pointer: &mut JobPtr) {

    unsafe {
        let memory =  memory_pointer.memory_ptr;

        for y in output_y_start..output_y_end {
            for x in 0..output_x {
                let mut running_total = 0.0f32;
                for z in 0..shared_z {
                    let first_index = z + (y * shared_z);

                    let second_index = x + (output_y * z);
                    
                    let left_offset = lhs_start + first_index;
                    let right_offset = rhs_start + second_index;

                    let left_hand_value = *memory.offset(left_offset as isize);
                    let right_hand_value = *memory.offset(right_offset as isize);

                    running_total += left_hand_value * right_hand_value;
                }
                
                let index = x + y * output_x;//All of these are flat buffers, so we need to calculate what the the final index
                *memory.offset((destination_start + index) as isize) = running_total;
            }
        }



    }
}
