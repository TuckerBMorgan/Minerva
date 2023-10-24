use crate::operations::*;

pub fn compile_copy_operation(from: MemoryToken, to: MemoryToken) -> Vec<JobType> {
    return vec![
        JobType::new_copy_type(from, to)
    ];
}

pub fn preform_copy_operation(from: MemoryToken, to: MemoryToken, memory_pointer: &mut JobPtr) {
    unsafe {
        let memory =  memory_pointer.memory_ptr;
        for i in 0..from.size {
            let from_offset = from.start + i;
            let to_offset = to.start + i;
            let from_value = *memory.offset(from_offset as isize);
            *memory.offset(to_offset as isize) = from_value;
        }
    }
}
