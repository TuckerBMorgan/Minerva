pub struct AddJob {
    pub lhs_start: usize,
    pub rhs_start: usize,
    pub destination_start: usize,
    pub length: usize
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



/*
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
*/