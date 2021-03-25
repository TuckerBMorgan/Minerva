extern crate stopwatch;
use stopwatch::{Stopwatch};

use std::sync::Arc;
use std::thread;
use blas::*;

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

fn bench_imperative_mul(matrix_size: usize) {
    println!("Benching imperative mul");
    let mut total_results = 0;
    
    for i in 0..25 {
        let result = imperative_mul(matrix_size);
        total_results += result;
        println!("Finished run # {}", i + 1);
    }
    println!("Average {}", total_results as f32 / 25.);     
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
    let number_of_rows = matrix_size / cpus;
    let number_of_rows_remainder = (matrix_size) % cpus;
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
                let mut total_result = 0.;
                let output_x = location % job.matrix_size;
                let output_y = location / job.matrix_size;

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
                    *memory.offset(location as isize) = total_result;
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

fn bench_jank_mul(matrix_size: usize) {
    println!("Benching Jank Mul");
    let mut total_results = 0;
    
    for i in 0..25 {
        let result = jank_mul(matrix_size);
        total_results += result;
        println!("Finished run # {}", i + 1);
    }
    println!("Average {}", total_results as f32 / 25.);     
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

pub fn bench_imperative_add(matrix_size: usize) {
    println!("Benching imperative Add");
    let mut total_results = 0;
    
    for i in 0..25 {
        let result = imperative_add(matrix_size);
        total_results += result;
        println!("Finished run # {}", i + 1);
    }
    println!("Average {}", total_results as f32 / 25.); 
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

pub fn bench_jank_add(matrix_size: usize) {
    println!("Benching Jank Add");
    let total_matrix_size = matrix_size * matrix_size;
    let mut total_results = 0;
    
    for i in 0..25 {
        let result = jank_add(matrix_size);
        total_results += result;
        println!("Finished run # {}", i + 1);
    }
    println!("Average {}", total_results as f32 / 25.); 
}

fn bench_add() {

    let matrix_size = 10000;

    println!("Starting Add Benchmark");
    bench_imperative_add(matrix_size);
    bench_jank_add(matrix_size);
}

fn bench_mul() {
    //Imperative has some trouble with larger sizes, so we reduce it by a factor of 10
    let matrix_size = 1000;
    println!("Starting Mul Benchmark");
    bench_imperative_mul(matrix_size);
    bench_jank_mul(matrix_size);
}

fn main() {
    bench_add();
    bench_mul();
}