use crate::operations::*;
use log::info;

pub fn compile_matrix_mul_operation(inputs: Vec<MemoryToken>, output_variable: MemoryToken) -> Vec<JobType> {
    let cpus = (num_cpus::get() - 1) as usize;//Give our computer some room to breath
    let mut jobs = vec![];
    if output_variable.y_dim > cpus {
        let mul_job = JobType::new_matrix_mul_type(inputs[0].start, inputs[1].start, output_variable.start, output_variable.x_dim, output_variable.y_dim,0, output_variable.y_dim, output_variable.x_dim);
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
            let mul_job = JobType::new_matrix_mul_type(inputs[0].start, inputs[1].start, output_variable.start, output_variable.x_dim, output_variable.y_dim, output_y_start, end_point, inputs[0].x_dim);
            jobs.push(mul_job);
        }        

    }

    return jobs;
}

pub fn preform_matrix_mul_job(lhs_start: usize, rhs_start: usize, destination_start: usize, output_x: usize, output_y: usize, output_y_start: usize, output_y_end: usize,shared_z: usize, memory_pointer: &mut JobPtr) {

    info!("{:?} {:?} {:?} {:?}", output_y_start, output_y_end, shared_z, output_x);
    unsafe {
        let memory =  memory_pointer.memory_ptr;

        for y in output_y_start..output_y_end {
            for x in 0..output_x {
                let mut running_total = 0.0f32;
                for z in 0..shared_z {
                    let first_index = z  + (y * shared_z);
                    let second_index = x + (output_y * z);
                    
                    info!("{:?} {:?}", first_index, second_index);

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
