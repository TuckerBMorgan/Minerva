use crate::operations::*;
use std::sync::mpsc::*;

pub enum JobPoke {
    Done(usize)//Thread number
}


//Thread number is a ID we can use in case of issue, small future programming
pub fn worker_thread(thread_number: usize, mut memory_ptr: JobPtr, to_equation_sender: Sender<JobPoke>, from_server_reciver: Receiver<JobType>) {
    let _ = to_equation_sender.send(JobPoke::Done(thread_number));
    loop {
        let result = from_server_reciver.recv();
        match result {
            Ok(job) => {
                match job {
                    JobType::Add(lhs, rhs, destination, length) =>  {
                        preform_add_job(lhs, rhs, destination, length, &mut memory_ptr);
                    },
                    JobType::Diff(lhs, rhs, destination, length) => {
                        preform_diff_job(lhs, rhs, destination, length, &mut memory_ptr);
                    },
                    JobType::MatrixMul(lhs, rhs, destination_start, output_x, output_y) => {
                        preform_matrix_multiplication_job(lhs, rhs, destination_start, output_x, output_y, &mut memory_ptr);
                    },
                    JobType::Map(lhs, destination_start,length,mapping_function) => {
                        preform_map_operation(lhs, destination_start, length, mapping_function, &mut memory_ptr);
                    },
                    JobType::ElementWiseMul(lhs, rhs, destination, length) => {
                        preform_element_wise_mul_job(lhs, rhs, destination, length, &mut memory_ptr);
                    },
                    JobType::Scalar(lhs, rhs, destination, length) => {
                        preform_scaler_operation(lhs, rhs, destination, length, &mut memory_ptr);
                    }
                    JobType::End => {
                        break;
                    },
                    _ => {
                        
                    }
                }
            },
            Err(_) => {
                break;
            }
        }
        let _ = to_equation_sender.send(JobPoke::Done(thread_number));
    }
}
