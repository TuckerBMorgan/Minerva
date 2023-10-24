#![allow(dead_code)]
mod operations;


#[cfg(test)]
pub mod tests {
    use std::collections::{HashMap};
    use std::hash::Hash;
    pub use crate::operations::*;
    use log::LevelFilter;
    use log::info;
    use rand::Rng;
    /*
//    pub use crate::equation::*;
*/
    #[test]
    fn main() {

        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add).unwrap();

        let d = equation.new_variable(2, 2);
        let e  = equation.new_operation_in_graph(vec![d, c], Operator::Add).unwrap();

        let f = equation.new_variable(2, 2);
        let g = equation.new_operation_in_graph(vec![e, f], Operator::MatrixMul).unwrap();

        let i = equation.new_variable(2, 2);
        let y = equation.new_operation_in_graph(vec![g, i], Operator::MatrixMul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0, 1.0, 1.0]);
        inputs.insert(b, vec![1.0, 2.0, 1.0, 1.0]);

        inputs.insert(d, vec![10.0, 2.0, 10.0, 2.0]);

        inputs.insert(f, vec![10.0, 10.0, 10.0, 10.0]);

        inputs.insert(i, vec![2.0, 2.0, 2.0, 2.0]);
        equation.compile();
        equation.evaluate(&mut inputs);
    }

    #[test]
    fn allocate_equation_test() {
        let _equation = Equation::new();
    }

    #[test]
    fn add_variable_to_equation_test() {
        let mut equation = Equation::new();
        let _a = equation.new_variable(2, 2);

    }

    #[test]
    fn add_operation_to_equation_test() {
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);

        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add);
        assert!(c.is_ok());
    }
    
    #[test]
    fn simple_addition_test() {
        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0,
                                   1.0, 1.0]);
        inputs.insert(b, vec![1.0, 1.0,
                                   2.0, 1.0]);

        equation.compile();
        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        assert!(result == vec![2.0, 2.0, 3.0, 2.0]);
    }
    
    //The large addition test, is our other addtion code path work the same as our
    //simple addition code path
    //
    #[test]
    fn large_addition_test() {
        let _ = simple_logging::log_to_file("server.log", LevelFilter::Info);
        let x_size = 10000;
        let y_size = 10000;
        let total_size = x_size * y_size;
        let mut equation = Equation::new();
        let a = equation.new_variable(y_size, x_size);
        let b = equation.new_variable(y_size, x_size);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add).unwrap();
        let mut a_value = vec![];
        let mut b_value = vec![];

        for i in 0..total_size {
            a_value.push(i as f32);
            b_value.push(i as f32);
        }

        let mut inputs = HashMap::new();
        inputs.insert(a, a_value);
        inputs.insert(b, b_value);
        equation.compile();
        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        for i in 0..total_size {
            assert!(result[i] == i as f32 * 2.0);
        }
    }

    #[test]
    fn simple_mul_test() {
        let _ = simple_logging::log_to_file("server.log", LevelFilter::Info);
        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::MatrixMul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0,
                                   3.0, 4.0]);

        inputs.insert(b, vec![5.0, 6.0,
                                   7.0, 8.0]);

        equation.compile();
        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        assert!(result == vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test] 
    fn unsquare_mul_test() {
        let mut equation = Equation::new();
        let a = equation.new_variable(3, 2);
        let b = equation.new_variable(2, 3);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::MatrixMul).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0]);

        inputs.insert(b, vec![7.0, 8.0,
                                   9.0, 10.0,
                                   11.0, 12.0]);

        equation.compile();
        equation.evaluate(&mut inputs);
        let result = equation.get_variable(c);
        println!("{:?}", result);
        assert!(result == vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn use_operation_result_test() {
        let _ = simple_logging::log_to_file("server.log", LevelFilter::Info);
        let mut equation = Equation::new();

        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Add).unwrap();

        let d = equation.new_variable(2, 2);
        let e = equation.new_operation_in_graph(vec![c, d], Operator::MatrixMul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0,
                              1.0, 1.0]);
        inputs.insert(b, vec![1.0, 1.0,
                              2.0, 1.0]);
        inputs.insert(d, vec![5.0, 6.0,
                              7.0, 8.0]);
        equation.compile();
        equation.evaluate(&mut inputs);

        let result = equation.get_variable(c);
        assert!(result == vec![2.0, 2.0,
                               3.0, 2.0]);
        let result = equation.get_variable(e);
        info!("{:?}", result);
        assert!(result == vec![24.0, 28.0,
                               29.0, 34.0]);
    }


    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }



    #[test]
    fn simple_mapping_test() {
        let _ = simple_logging::log_to_file("server.log", LevelFilter::Info);
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 2);
        let b = equation.new_mapping_operation(a, sigmoid).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 0.0, 5.0, 0.75]);

        equation.compile();
        equation.evaluate(&mut inputs);

        let result = equation.get_variable(b);
        let expected = vec![0.731, 0.500, 0.993, 0.679];

        let mut total_diff = 0.0f32;

        for i in 0..4 {
            total_diff += (result[i] - expected[i]).abs();
        }

        assert!(total_diff < 0.01f32);
    }

    fn sigmoid_prime(x: f32) -> f32 {
       x * (1.0f32 - x) 
    }
    #[test]
    fn simple_dif_operation_test() {
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::Diff).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0,
                             1.0, 1.0]);
        inputs.insert(b, vec![1.0, 1.0,
                              1.0, 1.0]);
        equation.compile();
        equation.evaluate(&mut inputs);

        let result = equation.get_variable(c);

        assert!(result == vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn simple_element_wise_mul_operation_test() {
        let mut equation = Equation::new();
        let a = equation.new_variable(2, 2);
        let b = equation.new_variable(2, 2);
        let c = equation.new_operation_in_graph(vec![a, b], Operator::ElementWiseMul).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0,
                             1.0, 1.0]);
        inputs.insert(b, vec![2.0, 2.0,
                              2.0, 2.0]);
        equation.compile();
        equation.evaluate(&mut inputs);

        let result = equation.get_variable(c);
        assert!(result == vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn big_test() { 
        let input_size = 2;
        let hidden_size1 = 5;
        let hidden_size2 = 5;
        let output_size = 1;
        let batch_size = 4;

        let mut feed_foward = Equation::new();
        // Forward Pass
        // if you make the Y(first dimenion) larger here, that is more or less batches
        let input = feed_foward.new_variable(input_size, batch_size);
        let learning_rate = feed_foward.new_variable(1, 1);
        
 
        let first_dense_weight =   feed_foward.new_variable(hidden_size1, input_size);
        feed_foward.random_init_variable(first_dense_weight);
        let first_forward_pass =   feed_foward.new_operation_in_graph(vec![input, first_dense_weight], Operator::MatrixMul).unwrap();

        let first_bias_weights =   feed_foward.new_variable(first_forward_pass.y_size, first_forward_pass.x_size);

        let first_bias_offset =    feed_foward.new_operation_in_graph(vec![first_forward_pass, first_bias_weights], Operator::Add).unwrap();
        let first_sigmoid_mapping = feed_foward.new_mapping_operation(first_bias_offset, sigmoid).unwrap();
        let second_dense_weights = feed_foward.new_variable(hidden_size1, hidden_size2);
        feed_foward.random_init_variable(second_dense_weights);

        let second_forward_pass =  feed_foward.new_operation_in_graph(vec![first_sigmoid_mapping, second_dense_weights], Operator::MatrixMul).unwrap();
        let second_bias_weights =  feed_foward.new_variable(second_forward_pass.y_size, second_forward_pass.x_size);
        let second_bias_offset = feed_foward.new_operation_in_graph(vec![second_forward_pass, second_bias_weights], Operator::Add).unwrap();
        let second_sigmoid_mapping = feed_foward.new_mapping_operation(second_bias_offset, sigmoid).unwrap();

        let third_dense_weights = feed_foward.new_variable(output_size, hidden_size2);
        feed_foward.random_init_variable(third_dense_weights);
        let third_forward_pass = feed_foward.new_operation_in_graph(vec![second_sigmoid_mapping, third_dense_weights], Operator::MatrixMul).unwrap();
        let third_bias_weight = feed_foward.new_variable(third_forward_pass.y_size, third_forward_pass.x_size);
        let third_bias_offset = feed_foward.new_operation_in_graph(vec![third_forward_pass, third_bias_weight], Operator::Add).unwrap();

        let third_sigmoid_mapping = feed_foward.new_mapping_operation(third_bias_offset, sigmoid).unwrap();

        let target = feed_foward.new_variable(output_size, batch_size);
        let target_error = feed_foward.new_operation_in_graph(vec![third_sigmoid_mapping, target], Operator::Diff).unwrap();

        
        let second_sigmoid_mapping_transpose = feed_foward.transpose(second_sigmoid_mapping);
        let third_dense_weight_delta = feed_foward.new_operation_in_graph(vec![second_sigmoid_mapping_transpose, target_error], Operator::MatrixMul).unwrap();
        
        // the first bias delta offset is target_error
        let third_weight_transpose = feed_foward.transpose(third_dense_weights);
        let second_sigmoid_derivative = feed_foward.new_mapping_operation(second_sigmoid_mapping, sigmoid_prime).unwrap();
        let second_layer_middle_opeartion = feed_foward.new_operation_in_graph(vec![target_error, third_weight_transpose], Operator::MatrixMul).unwrap();
        let second_dense_middle_operation = feed_foward.new_operation_in_graph(vec![second_layer_middle_opeartion, second_sigmoid_derivative], Operator::ElementWiseMul).unwrap();
        let first_sigmoid_tranpose = feed_foward.transpose(first_sigmoid_mapping);
        let second_dense_weight_delta = feed_foward.new_operation_in_graph(vec![first_sigmoid_tranpose, second_dense_middle_operation], Operator::MatrixMul).unwrap();
        // second_dense_middle_operation is the second_bias_delta

        let second_weight_transpose = feed_foward.transpose(second_dense_weights);
        let first_sigmoid_derivative = feed_foward.new_mapping_operation(first_sigmoid_mapping, sigmoid_prime).unwrap();
        let first_layer_middle_opeartion = feed_foward.new_operation_in_graph(vec![second_dense_middle_operation, second_weight_transpose], Operator::MatrixMul).unwrap();
        let first_dense_middle_operation = feed_foward.new_operation_in_graph(vec![first_layer_middle_opeartion, first_sigmoid_derivative], Operator::ElementWiseMul).unwrap();
        let input_tranpose = feed_foward.transpose(input);
        let first_dense_weight_delta = feed_foward.new_operation_in_graph(vec![input_tranpose, first_dense_middle_operation], Operator::MatrixMul).unwrap();

        let learning_rate_adjusted_third_weight_delta = feed_foward.new_operation_in_graph(vec![third_dense_weight_delta, learning_rate], Operator::Scalar).unwrap();
        let new_third_weights = feed_foward.new_operation_in_graph(vec![learning_rate_adjusted_third_weight_delta, third_dense_weights], Operator::Diff).unwrap();

        let learning_rate_adjusted_second_weight_delta = feed_foward.new_operation_in_graph(vec![second_dense_weight_delta, learning_rate], Operator::Scalar).unwrap();
        let new_second_weights = feed_foward.new_operation_in_graph(vec![learning_rate_adjusted_second_weight_delta, second_dense_weights], Operator::Diff).unwrap();

        let learning_rate_adjusted_first_weight_delta = feed_foward.new_operation_in_graph(vec![first_dense_weight_delta, learning_rate], Operator::Scalar).unwrap();
        let new_first_weights = feed_foward.new_operation_in_graph(vec![learning_rate_adjusted_first_weight_delta, first_dense_weight], Operator::Diff).unwrap();
        // first_dense_middle_opeartion is delta bias
        let _update_third = feed_foward.new_operation_in_graph(vec![new_third_weights, third_dense_weights], Operator::Copy).unwrap();
        let _update_second = feed_foward.new_operation_in_graph(vec![new_second_weights, second_dense_weights], Operator::Copy).unwrap();
        let _update_first = feed_foward.new_operation_in_graph(vec![new_first_weights, first_dense_weight], Operator::Copy).unwrap();

        feed_foward.compile();
        let mut inputs = HashMap::new();
        inputs.insert(input, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        let predicted = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        inputs.insert(target, vec![0.0, 1.0, 1.0, 0.0]);
        inputs.insert(learning_rate, vec![0.1]);

        for _ in 0..100 {
            feed_foward.evaluate(&mut inputs);
            let current_error = feed_foward.get_variable(target_error);
            
            let mut sum = 0.0;
            for (index, value) in current_error.iter().enumerate() {
                sum += predicted[index] - value;
            }

            println!("{:?}", sum);
        }
        println!("----");
    }
    

    #[test] 
    fn what_test() {
        let mut feed_foward = Equation::new();
        let a =   feed_foward.new_variable(3, 2);
        let b =   feed_foward.new_variable(2, 3);
        let a_t = feed_foward.transpose(a);
        let x = feed_foward.new_operation_in_graph(vec![a, b], Operator::MatrixMul).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        inputs.insert(b, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        feed_foward.compile();
        feed_foward.evaluate(&mut inputs);

        println!("{:?}", feed_foward.get_variable(x));
    }
}

pub mod prelude {
    pub use crate::operations::*;
}